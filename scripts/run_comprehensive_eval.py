#!/usr/bin/env python
"""Comprehensive evaluation without GT: statistics, coverage, quality checks."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from longai.utils.io import ensure_dir, read_json, read_jsonl, write_json


def eval_tool_stats(session_ids: list[str], artifacts_dir: Path) -> dict:
    """Compute perception tool statistics."""
    stats = {"sessions": len(session_ids), "vad": {}, "asr": {}, "scene": {}, "speaker": {}, "events": {}}
    total_vad = 0
    total_asr_nonempty = 0
    total_speech_duration = 0.0
    total_audio_duration = 0.0

    sm_rows = read_jsonl(Path("data/processed/session_manifest.jsonl"))
    sm_map = {r["session_id"]: r for r in sm_rows}

    for sid in session_ids:
        row = sm_map.get(sid, {})
        total_audio_duration += row.get("duration_sec", 0)

        vad = read_json(artifacts_dir / "perception/vad" / f"{sid}.json")
        asr = read_json(artifacts_dir / "perception/asr" / f"{sid}.json")
        scene = read_json(artifacts_dir / "perception/scene" / f"{sid}.json")
        spk = read_json(artifacts_dir / "perception/speaker_role" / f"{sid}.json")
        events = read_json(artifacts_dir / "perception/events" / f"{sid}.json")

        n_seg = len(vad["segments"])
        total_vad += n_seg
        total_speech_duration += sum(s["end"] - s["start"] for s in vad["segments"])

        nonempty = sum(1 for s in asr["segments"] if s.get("text", "").strip() not in ("", "..."))
        total_asr_nonempty += nonempty

        # Scene distribution
        scene_labels = Counter(s["label"] for s in scene["segments"])
        stats["scene"][sid] = dict(scene_labels)

        # Speaker role distribution
        spk_roles = Counter(s["role"] for s in spk["segments"])
        stats["speaker"][sid] = dict(spk_roles)

        # Event distribution
        all_events = Counter()
        for s in events["segments"]:
            for t in s.get("tags", []):
                all_events[t] += 1
        stats["events"][sid] = dict(all_events.most_common(5))

        # ASR confidence stats
        asr_confs = [s.get("confidence", 0) for s in asr["segments"]]
        stats["asr"][sid] = {
            "segments": n_seg,
            "nonempty": nonempty,
            "nonempty_ratio": round(nonempty / n_seg, 3) if n_seg else 0,
            "avg_confidence": round(sum(asr_confs) / len(asr_confs), 3) if asr_confs else 0,
        }

    stats["vad"] = {
        "total_segments": total_vad,
        "total_speech_duration_s": round(total_speech_duration, 1),
        "total_audio_duration_s": round(total_audio_duration, 1),
        "speech_ratio": round(total_speech_duration / total_audio_duration, 3) if total_audio_duration else 0,
    }
    return stats


def eval_memory_stats(session_ids: list[str], artifacts_dir: Path) -> dict:
    """Compute memory graph statistics."""
    stats = {}
    for sid in session_ids:
        graph = read_json(artifacts_dir / "memory/session_graphs" / f"{sid}.json")
        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])
        node_types = Counter(n.get("node_type", "?") for n in nodes)
        edge_relations = Counter(e.get("relation", "?") for e in edges)
        stats[sid] = {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "node_types": dict(node_types),
            "edge_relations": dict(edge_relations),
            "intent_count": node_types.get("Intent", 0),
            "episode_count": node_types.get("Episode", 0),
            "entity_count": node_types.get("Entity", 0),
        }
    return stats


def eval_reasoning_stats(session_ids: list[str], artifacts_dir: Path) -> dict:
    """Compute reasoning output statistics."""
    stats = {}
    for sid in session_ids:
        result = read_json(artifacts_dir / "reasoning/amr_full" / f"{sid}.json")
        items = result.get("results", [])
        stats[sid] = {
            "num_queries": len(items),
            "avg_confidence": round(sum(r["confidence"] for r in items) / len(items), 3) if items else 0,
            "abstain_rate": round(sum(1 for r in items if r.get("abstained")) / len(items), 3) if items else 0,
            "avg_graph_trace_len": round(sum(len(r.get("graph_trace", [])) for r in items) / len(items), 1) if items else 0,
            "avg_evidence_trace_len": round(sum(len(r.get("evidence_trace", [])) for r in items) / len(items), 1) if items else 0,
            "queries": [{"query": r["query"], "type": r["query_type"], "answer_preview": r["answer"][:150]} for r in items],
        }
    return stats


def qualitative_analysis(session_ids: list[str], artifacts_dir: Path) -> list[dict]:
    """Extract interesting evidence packs for qualitative analysis."""
    cases = []
    for sid in session_ids:
        packs = read_jsonl(artifacts_dir / "evidence_packs" / f"{sid}.jsonl")
        for p in packs:
            text = p.get("asr_text", "").strip()
            if len(text) > 30 and text != "...":
                cases.append({
                    "session_id": sid,
                    "segment_id": p["segment_id"],
                    "start": p["start_time"],
                    "end": p["end_time"],
                    "speaker": p["speaker_role"],
                    "text": text,
                    "scene": p["scene_label"],
                    "events": p["event_tags"],
                })
        if len(cases) >= 20:
            break
    return cases[:20]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-manifest", type=Path, default=Path("data/processed/session_manifest.jsonl"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/eval/comprehensive_eval.json"))
    args = parser.parse_args()

    rows = read_jsonl(args.session_manifest)
    session_ids = [r["session_id"] for r in rows]

    report = {
        "pipeline": "LongAI Phase 1 — End-to-End Evaluation",
        "num_sessions": len(session_ids),
        "session_ids": session_ids,
        "tool_stats": eval_tool_stats(session_ids, args.artifacts_dir),
        "memory_stats": eval_memory_stats(session_ids, args.artifacts_dir),
        "reasoning_stats": eval_reasoning_stats(session_ids, args.artifacts_dir),
        "qualitative_cases": qualitative_analysis(session_ids, args.artifacts_dir),
    }

    ensure_dir(args.output.parent)
    write_json(args.output, report)
    print(f"Comprehensive evaluation written to {args.output}")
    print(f"  Sessions: {len(session_ids)}")
    print(f"  Tool stats: VAD={report['tool_stats']['vad']}")
    print(f"  Memory stats: {report['memory_stats']}")
    print(f"  Reasoning stats: {report['reasoning_stats']}")
    print(f"  Qualitative cases: {len(report['qualitative_cases'])}")


if __name__ == "__main__":
    main()
