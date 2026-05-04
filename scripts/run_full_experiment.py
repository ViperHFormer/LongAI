#!/usr/bin/env python
"""Run complete experiment: pipeline configs + LLM+RAG baseline + ablation comparison."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from longai.memory.manager import build_memory_from_evidence
from longai.reasoning.amr import answer_query
from longai.reasoning.rag_baseline import answer_query_rag
from longai.utils.io import read_jsonl, write_json, ensure_dir

QUERIES = [
    "What is the current status of user intent?",
    "What is the next intent?",
    "What changed or was canceled?",
]

# Experiment configs: (name, mcoc_backend, amr_backend, description)
EXPERIMENTS = {
    # ── Pipeline configs ──
    "full": {
        "mcoc": "mock", "amr": "mock",
        "desc": "Full pipeline: all tools, rule-based MCoC + AMR",
    },
    "transcript_only": {
        "mcoc": "mock", "amr": "mock",
        "desc": "Transcript-only ablation (−scene, −events, −speaker_role)",
        "use_alt_packs": "transcript_only",
    },
    "no_scene": {
        "mcoc": "mock", "amr": "mock",
        "desc": "Without scene classification",
        "use_alt_packs": "no_scene",
    },
    "no_events": {
        "mcoc": "mock", "amr": "mock",
        "desc": "Without event detection",
        "use_alt_packs": "no_events",
    },
    "llm_mcoc": {
        "mcoc": "local_hf", "amr": "mock",
        "desc": "LLM MCoC + rule AMR",
    },
    "llm_amr": {
        "mcoc": "mock", "amr": "local_hf",
        "desc": "Rule MCoC + LLM AMR",
    },
    "llm_full": {
        "mcoc": "local_hf", "amr": "local_hf",
        "desc": "LLM MCoC + LLM AMR",
    },
    # ── Baselines ──
    "llm_rag": {
        "mcoc": None, "amr": "rag",
        "desc": "Pure LLM+RAG baseline (no graph construction)",
    },
}


def run_pipeline_config(name: str, cfg: dict, base_dir: Path, session_ids: list[str]) -> dict:
    """Run one pipeline config (perception → memory → reasoning)."""
    artifacts_dir = ensure_dir(base_dir / name)

    total_nodes = total_edges = total_intents = total_entities = 0
    total_conf = total_abstain = total_trace = 0
    n_queries = 0

    for sid in session_ids:
        # Determine evidence packs source
        packs_src = base_dir / (cfg.get("use_alt_packs", name))
        pack_path = packs_src / "evidence_packs" / f"{sid}.jsonl"
        if not pack_path.exists():
            pack_path = base_dir / name / "evidence_packs" / f"{sid}.jsonl"
        if not pack_path.exists():
            print(f"  SKIP {sid}: no evidence packs")
            continue

        # Build memory
        mem_dir = ensure_dir(artifacts_dir / "memory")
        snap = build_memory_from_evidence(
            pack_path,
            mem_dir / "session_graphs" / f"{sid}.json",
            mem_dir / "update_logs" / f"{sid}.jsonl",
            backend=cfg["mcoc"],
        )

        n_nodes = len(snap["nodes"])
        n_edges = len(snap["edges"])
        n_int = sum(1 for n in snap["nodes"] if n["node_type"] == "Intent")
        n_ent = sum(1 for n in snap["nodes"] if n["node_type"] == "Entity")
        total_nodes += n_nodes
        total_edges += n_edges
        total_intents += n_int
        total_entities += n_ent

        # Run reasoning
        for q in QUERIES:
            result = answer_query(snap, q, backend=cfg["amr"])
            n_queries += 1
            total_conf += result.confidence
            if result.abstained:
                total_abstain += 1
            total_trace += len(result.graph_trace)

    return {
        "nodes": total_nodes,
        "edges": total_edges,
        "intents": total_intents,
        "entities": total_entities,
        "reasoning_conf": round(total_conf / n_queries, 3) if n_queries else 0,
        "abstain_rate": round(total_abstain / n_queries, 3) if n_queries else 0,
        "trace_len": round(total_trace / n_queries, 1) if n_queries else 0,
    }


def run_rag_baseline(name: str, cfg: dict, base_dir: Path, session_ids: list[str]) -> dict:
    """Run pure LLM+RAG baseline (no MCoC graph)."""
    packs_src = base_dir / "full"  # Use full perception data
    total_conf = total_abstain = 0
    n_queries = 0

    for sid in session_ids:
        for q in QUERIES:
            result = answer_query_rag(packs_src, sid, q, topk=5)
            n_queries += 1
            total_conf += result.confidence
            if result.abstained:
                total_abstain += 1

    return {
        "nodes": 0, "edges": 0, "intents": 0, "entities": 0,
        "reasoning_conf": round(total_conf / n_queries, 3) if n_queries else 0,
        "abstain_rate": round(total_abstain / n_queries, 3) if n_queries else 0,
        "trace_len": 0,
        "note": "No graph; keyword retrieval + LLM only",
    }


def print_comparison(results: dict) -> None:
    """Print formatted comparison table."""
    headers = ["Config", "Nodes", "Edges", "Intents", "Entities", "Reas.Conf", "Abstain%"]
    print()
    print("=" * 110)
    print("FULL EXPERIMENT RESULTS")
    print("=" * 110)
    header_line = "".join(f"{h:>15s}" for h in headers)
    print(header_line)
    print("-" * 110)

    for name in EXPERIMENTS:
        r = results.get(name, {})
        vals = [
            name[:14],
            str(r.get("nodes", "-")),
            str(r.get("edges", "-")),
            str(r.get("intents", "-")),
            str(r.get("entities", "-")),
            f"{r.get('reasoning_conf', 0):.3f}",
            f"{r.get('abstain_rate', 0):.3f}",
        ]
        print("".join(f"{v:>15s}" for v in vals))

    print("-" * 110)

    # Deltas vs full
    baseline = results.get("full", {})
    print("\nDeltas vs FULL pipeline baseline:")
    print()
    for name in EXPERIMENTS:
        if name == "full":
            continue
        r = results.get(name, {})
        deltas = []
        for key, label in [("intents", "I"), ("edges", "E"), ("reasoning_conf", "Conf"), ("abstain_rate", "Abst")]:
            bv = baseline.get(key, 1)
            cv = r.get(key, 0)
            if isinstance(bv, (int, float)) and bv != 0 and cv != "-":
                dp = (cv - bv) / abs(bv) * 100 if abs(bv) > 0 else 0
                deltas.append(f"{label}:{dp:+.1f}%")
            else:
                deltas.append(f"{label}:N/A")
        print(f"  {name:20s} | {' | '.join(deltas)}")

    print("=" * 110)


def main():
    parser = argparse.ArgumentParser(description="Run full LongAI experiment suite")
    parser.add_argument("--base-dir", type=Path, default=Path("artifacts/experiment"))
    parser.add_argument("--session-ids", type=str, default="session_0000,session_0001,session_0002,session_0003,session_0004")
    parser.add_argument("--configs", type=str, default="",
                        help="Comma-separated config names (default: all)")
    parser.add_argument("--output", type=Path, default=Path("artifacts/experiment/comparison.json"))
    args = parser.parse_args()

    session_ids = [s.strip() for s in args.session_ids.split(",")]
    config_names = [n.strip() for n in args.configs.split(",") if n.strip()] if args.configs else list(EXPERIMENTS)

    print(f"Experiment: {len(config_names)} configs × {len(session_ids)} sessions")
    print(f"Output: {args.base_dir}")
    print()

    results = {}
    for i, name in enumerate(config_names):
        cfg = EXPERIMENTS[name]
        print(f"[{i+1}/{len(config_names)}] {name}: {cfg['desc']}")
        t0 = time.time()

        if cfg["amr"] == "rag":
            r = run_rag_baseline(name, cfg, args.base_dir, session_ids)
        else:
            r = run_pipeline_config(name, cfg, args.base_dir, session_ids)

        results[name] = r
        elapsed = time.time() - t0
        print(f"  → N={r['nodes']} E={r['edges']} I={r['intents']} En={r['entities']} "
              f"| Conf={r['reasoning_conf']:.3f} Abst={r['abstain_rate']:.3f} "
              f"| {elapsed:.0f}s")
        print()

    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "configs": {n: EXPERIMENTS[n]["desc"] for n in config_names},
        "results": results,
    }
    write_json(ensure_dir(args.output.parent) / args.output.name, output)
    print(f"Results saved to {args.output}")

    # Print comparison
    print_comparison(results)


if __name__ == "__main__":
    main()
