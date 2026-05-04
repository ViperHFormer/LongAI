#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from longai.train.writer_data import (
    convert_graph_gt_to_ops,
    convert_update_logs_to_writer_data,
    split_writer_data,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["graph-gt", "update-logs"], default="update-logs")
    parser.add_argument("--graph-gt", type=Path, default=Path("data/annotations/graph_level_gt.json"))
    parser.add_argument("--update-logs-dir", type=Path, default=Path("artifacts/memory/update_logs"))
    parser.add_argument("--evidence-pack-dir", type=Path, default=Path("artifacts/evidence_packs"))
    parser.add_argument("--all-output", type=Path, default=Path("data/annotations/writer_all.jsonl"))
    parser.add_argument("--train-output", type=Path, default=Path("data/annotations/writer_train.jsonl"))
    parser.add_argument("--eval-output", type=Path, default=Path("data/annotations/writer_eval.jsonl"))
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    if args.mode == "graph-gt":
        count = convert_graph_gt_to_ops(args.graph_gt, args.all_output)
    else:
        count = convert_update_logs_to_writer_data(
            update_logs_dir=args.update_logs_dir,
            evidence_pack_dir=args.evidence_pack_dir,
            output_jsonl=args.all_output,
            max_samples=args.max_samples,
        )

    train_n, eval_n = split_writer_data(
        input_jsonl=args.all_output,
        train_jsonl=args.train_output,
        eval_jsonl=args.eval_output,
        eval_ratio=args.eval_ratio,
        seed=args.seed,
    )
    print(
        {
            "built_examples": count,
            "train_examples": train_n,
            "eval_examples": eval_n,
            "all_output": str(args.all_output),
            "train_output": str(args.train_output),
            "eval_output": str(args.eval_output),
        }
    )


if __name__ == "__main__":
    main()
