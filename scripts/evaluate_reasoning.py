#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from longai.eval.reasoning_metrics import evaluate_reasoning
from longai.utils.io import read_json, read_jsonl, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-id", type=str, required=True)
    parser.add_argument("--reasoning-output", type=Path, default=Path("artifacts/reasoning/amr_full"))
    parser.add_argument("--gt", type=Path, default=Path("data/annotations/reasoning_gt.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/eval/reasoning_eval.json"))
    args = parser.parse_args()

    if not args.gt.exists():
        write_json(args.output, {"warning": "Reasoning GT not found. Skip reasoning eval."})
        print("Reasoning GT not found. reasoning eval skipped.")
        return

    pred_bundle = read_json(args.reasoning_output / f"{args.session_id}.json")
    results = pred_bundle.get("results", [])
    gt_rows = read_jsonl(args.gt)
    out = evaluate_reasoning(results, gt_rows)
    write_json(args.output, out)
    print(f"reasoning_eval -> {args.output}")


if __name__ == "__main__":
    main()
