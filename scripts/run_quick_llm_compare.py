#!/usr/bin/env python
"""Quick run: LLM configs on session_0000 only for fast comparison."""
from __future__ import annotations

import json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from longai.memory.manager import build_memory_from_evidence
from longai.reasoning.amr import answer_query

QUERIES = [
    "What is the current status of user intent?",
    "What is the next intent?",
    "What changed or was canceled?",
]

BASE_DIR = Path("artifacts/ablation_full")
SID = "session_0000"

def run_one(name, mcoc_backend, amr_backend):
    pack_path = BASE_DIR / name / "evidence_packs" / f"{SID}.jsonl"
    if not pack_path.exists():
        pack_path = BASE_DIR / "full" / "evidence_packs" / f"{SID}.jsonl"
    if not pack_path.exists():
        print(f"  {name}: no evidence packs, SKIP")
        return None

    t0 = time.time()
    snap = build_memory_from_evidence(
        pack_path,
        BASE_DIR / name / "memory" / "session_graphs" / f"{SID}.json",
        BASE_DIR / name / "memory" / "update_logs" / f"{SID}.jsonl",
        backend=mcoc_backend,
    )

    total_conf = total_abstain = 0
    for q in QUERIES:
        r = answer_query(snap, q, backend=amr_backend)
        total_conf += r.confidence
        if r.abstained:
            total_abstain += 1

    dt = time.time() - t0
    n_nodes = len(snap["nodes"])
    n_edges = len(snap["edges"])
    n_int = sum(1 for n in snap["nodes"] if n["node_type"] == "Intent")
    n_ent = sum(1 for n in snap["nodes"] if n["node_type"] == "Entity")

    result = {
        "nodes": n_nodes, "edges": n_edges,
        "intents": n_int, "entities": n_ent,
        "reasoning_conf": round(total_conf / 3, 3),
        "abstain_rate": round(total_abstain / 3, 3),
        "time_s": round(dt, 0),
    }
    print(f"  {name}: N={n_nodes} E={n_edges} I={n_int} En={n_ent} "
          f"Conf={result['reasoning_conf']:.3f} Abst={result['abstain_rate']:.3f} "
          f"({dt:.0f}s)")
    return result


def main():
    configs = [
        ("llm_mcoc", "local_hf", "mock"),
        ("llm_amr", "mock", "local_hf"),
        ("llm_full", "local_hf", "local_hf"),
    ]
    results = {}
    for name, mcoc, amr in configs:
        print(f"\nRunning {name} (mcoc={mcoc}, amr={amr})...")
        results[name] = run_one(name, mcoc, amr)

    # Save
    out_path = BASE_DIR / "comparison_llm_session0000.json"
    json.dump(results, open(out_path, "w"), indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
