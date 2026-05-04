#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from longai.eval.memory_metrics import evaluate_memory_graph
from longai.utils.io import read_json, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-id", type=str, required=True)
    parser.add_argument("--pred-graph-dir", type=Path, default=Path("artifacts/memory/session_graphs"))
    parser.add_argument("--gt-graph", type=Path, default=Path("data/annotations/graph_level_gt.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/eval/memory_eval.json"))
    args = parser.parse_args()

    if not args.gt_graph.exists():
        write_json(args.output, {"warning": "GT graph not found. Skip memory eval."})
        print("GT graph not found. memory eval skipped.")
        return

    pred = read_json(args.pred_graph_dir / f"{args.session_id}.json")
    gt = read_json(args.gt_graph)
    out = evaluate_memory_graph(pred, gt)
    write_json(args.output, out)
    print(f"memory_eval -> {args.output}")


if __name__ == "__main__":
    main()
