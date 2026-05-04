"""Pure LLM+RAG baseline: skip graph construction, retrieve evidence directly, ask LLM.

This provides a lower bound: if graph-based memory doesn't beat raw evidence+LLM,
the MCoC/AMR architecture adds no value over a simple RAG approach.
"""
from __future__ import annotations

import json
from pathlib import Path

from longai.schema.models import EvidencePack, ReasoningResult
from longai.utils.io import read_jsonl


def _load_evidence_packs(session_dir: Path, session_id: str) -> list[EvidencePack]:
    """Load all evidence packs for a session."""
    pack_path = Path(session_dir) / "evidence_packs" / f"{session_id}.jsonl"
    if not pack_path.exists():
        return []
    rows = read_jsonl(pack_path)
    return [EvidencePack(**r) for r in rows]


def _retrieve_relevant(packs: list[EvidencePack], query: str, topk: int = 5) -> list[EvidencePack]:
    """Simple keyword-overlap retrieval against ASR text, scene, and event tags."""
    query_terms = set(query.lower().split())
    scored = []
    for p in packs:
        text = (p.asr_text + " " + p.scene_label + " " + " ".join(p.event_tags)).lower()
        text_terms = set(text.split())
        overlap = len(query_terms & text_terms)
        if overlap > 0 or len(p.asr_text) > 10:
            scored.append((overlap, p))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scored[:topk]]


def answer_query_rag(
    session_dir: Path,
    session_id: str,
    query: str,
    topk: int = 5,
) -> ReasoningResult:
    """Answer a query using pure LLM+RAG (no graph)."""
    from longai.utils.llm import generate

    packs = _load_evidence_packs(session_dir, session_id)
    if not packs:
        return ReasoningResult(
            query=query,
            query_type="status_query",
            answer="No evidence available for this session.",
            confidence=0.1,
            abstained=True,
        )

    relevant = _retrieve_relevant(packs, query, topk=topk)

    # Build context from retrieved evidence
    evidence_blocks = []
    for i, p in enumerate(relevant):
        ctx_parts = [f"[{i+1}] time={p.start_time:.0f}-{p.end_time:.0f}s"]
        if p.asr_text:
            ctx_parts.append(f'text="{p.asr_text}"')
        ctx_parts.append(f"speaker={p.speaker_role.value}")
        ctx_parts.append(f"scene={p.scene_label}")
        if p.event_tags:
            ctx_parts.append(f"events={p.event_tags}")
        evidence_blocks.append(" | ".join(ctx_parts))

    context = "\n".join(evidence_blocks)

    prompt = f"""You are a personal assistant with access to audio evidence from the user's day.
Answer the query based ONLY on the provided evidence snippets.

Query: {query}

Evidence snippets (time-ordered):
{context}

Return a JSON object with:
- "answer": your answer based on the evidence
- "confidence": float 0-1
- "abstained": true if evidence is insufficient, false otherwise

Return ONLY valid JSON."""

    try:
        text = generate(prompt, max_new_tokens=512)
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            result = json.loads(text[start:end])
            return ReasoningResult(
                query=query,
                query_type="rag_baseline",
                answer=result.get("answer", "Unable to determine."),
                graph_trace=[],
                evidence_trace=[{"type": "rag", "num_retrieved": len(relevant)}],
                confidence=float(result.get("confidence", 0.5)),
                abstained=bool(result.get("abstained", False)),
            )
    except Exception:
        pass

    # Fallback: simple answer from retrieved evidence
    texts = [p.asr_text for p in relevant if p.asr_text]
    answer = "Based on available evidence: " + ("; ".join(texts[:3]) if texts else "no clear speech detected.")
    return ReasoningResult(
        query=query,
        query_type="rag_baseline",
        answer=answer,
        confidence=0.4,
        abstained=len(texts) == 0,
    )
