#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from longai.train.writer_eval import evaluate_writer_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-jsonl", type=Path, default=Path("data/annotations/writer_eval.jsonl"))
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("artifacts/train/mcoc_sft/checkpoint-final"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/eval/writer_sft_eval.json"))
    parser.add_argument("--max-eval-samples", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    args = parser.parse_args()

    summary = evaluate_writer_model(
        eval_jsonl=args.eval_jsonl,
        output_json=args.output,
        base_model=args.base_model,
        checkpoint_dir=args.checkpoint_dir,
        max_eval_samples=args.max_eval_samples,
        max_new_tokens=args.max_new_tokens,
    )
    print(summary)


if __name__ == "__main__":
    main()
