from __future__ import annotations

from longai.reasoning.retrieval import retrieve
from longai.reasoning.router import route_query
from longai.reasoning.scaffold import build_scaffold
from longai.schema.models import ReasoningResult


def answer_query_rule(graph_snapshot: dict, query: str) -> ReasoningResult:
    query_type = route_query(query, backend="mock")
    scaffold = build_scaffold(query_type)
    graph_trace, evidence_trace = retrieve(graph_snapshot, query)

    if not graph_trace and not evidence_trace:
        return ReasoningResult(
            query=query, query_type=query_type,
            answer="Insufficient evidence to answer confidently.",
            graph_trace=[{"scaffold": scaffold}], evidence_trace=[],
            confidence=0.18, abstained=True,
        )

    short_nodes = [x.get("label", x.get("node_id", "node")) for x in graph_trace[:3]]
    answer = "Likely relevant intents: " + ", ".join(short_nodes)
    return ReasoningResult(
        query=query, query_type=query_type, answer=answer,
        graph_trace=[{"scaffold": scaffold}, *graph_trace],
        evidence_trace=evidence_trace, confidence=0.62, abstained=False,
    )


def answer_query_llm(graph_snapshot: dict, query: str) -> ReasoningResult:
    import json
    from longai.utils.llm import generate
    from longai.reasoning.router import route_query_rule

    query_type = route_query_rule(query)
    scaffold = build_scaffold(query_type)
    graph_trace, evidence_trace = retrieve(graph_snapshot, query)

    # Build context from retrieved nodes and evidence
    node_info = []
    for n in graph_trace:
        node_info.append(f"- [{n.get('node_type', '?')}] {n.get('label', n.get('node_id', '?'))} (attrs={n.get('attributes', {})})")

    evidence_info = []
    for e in evidence_trace[:3]:
        p = e.get("payload", {})
        evidence_info.append(f"- [{p.get('segment_id', '?')}] {p.get('asr_text', '')[:200]}")

    context = f"""Memory graph nodes:
{chr(10).join(node_info) if node_info else '(none)'}

Evidence snippets:
{chr(10).join(evidence_info) if evidence_info else '(none)'}"""

    prompt = f"""You are an adaptive memory reasoning agent. Answer the user's query based on the provided memory context.

Query: {query}
Query type: {query_type}

{context}

Return a JSON object with:
- "answer": your answer to the query
- "confidence": a float between 0 and 1
- "abstained": true if there is insufficient evidence, false otherwise

Return ONLY valid JSON."""

    try:
        text = generate(prompt, max_new_tokens=512)
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            result = json.loads(text[start:end])
            return ReasoningResult(
                query=query, query_type=query_type,
                answer=result.get("answer", "Unable to determine."),
                graph_trace=[{"scaffold": scaffold}, *graph_trace],
                evidence_trace=evidence_trace,
                confidence=float(result.get("confidence", 0.5)),
                abstained=bool(result.get("abstained", False)),
            )
    except Exception:
        pass

    return answer_query_rule(graph_snapshot, query)


def answer_query(graph_snapshot: dict, query: str, backend: str = "mock") -> ReasoningResult:
    if backend == "local_hf":
        return answer_query_llm(graph_snapshot, query)
    return answer_query_rule(graph_snapshot, query)
