#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from longai.reasoning.amr import answer_query
from longai.utils.io import ensure_dir, read_json, read_jsonl, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-manifest", type=Path, default=Path("data/processed/session_manifest.jsonl"))
    parser.add_argument("--queries", type=Path, default=Path("data/annotations/reasoning_queries.jsonl"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--experiment-name", type=str, default="amr_full")
    parser.add_argument("--backend", type=str, default="mock")
    args = parser.parse_args()

    rows = read_jsonl(args.session_manifest)
    query_rows = read_jsonl(args.queries)

    def _default_queries(session_id: str) -> list[dict]:
        return [
            {"session_id": session_id, "query": "What is the current status of user intent?"},
            {"session_id": session_id, "query": "What is the next intent?"},
            {"session_id": session_id, "query": "What changed or was canceled?"},
        ]

    if not query_rows:
        query_rows = []
        for row in rows:
            query_rows.extend(_default_queries(row["session_id"]))

    out_dir = ensure_dir(args.artifacts_dir / "reasoning" / args.experiment_name)
    graph_dir = args.artifacts_dir / "memory/session_graphs"

    grouped: dict[str, list[dict]] = {}
    for item in query_rows:
        grouped.setdefault(item["session_id"], []).append(item)

    for row in rows:
        sid = row["session_id"]
        if sid not in grouped:
            grouped[sid] = _default_queries(sid)

    for sid, items in grouped.items():
        graph_path = graph_dir / f"{sid}.json"
        if not graph_path.exists():
            continue
        graph = read_json(graph_path)
        results = []
        for item in items:
            result = answer_query(graph, item["query"], backend=args.backend)
            results.append(result.model_dump())
        write_json(out_dir / f"{sid}.json", {"session_id": sid, "results": results})

    print(f"reasoning_done sessions={len(grouped)} -> {out_dir}")


if __name__ == "__main__":
    main()
