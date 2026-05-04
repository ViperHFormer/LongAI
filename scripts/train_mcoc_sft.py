#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from longai.train.writer_sft import WriterSFTConfig, run_writer_sft


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", type=Path, default=Path("data/annotations/writer_train.jsonl"))
    parser.add_argument("--eval-jsonl", type=Path, default=Path("data/annotations/writer_eval.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/train/mcoc_sft"))
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    args = parser.parse_args()

    cfg = WriterSFTConfig(
        model_name=args.model_name,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        max_seq_length=args.max_seq_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        seed=args.seed,
        load_in_4bit=args.load_in_4bit,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
    )
    eval_path = args.eval_jsonl if args.eval_jsonl.exists() else None
    summary = run_writer_sft(args.train_jsonl, args.output_dir, eval_path=eval_path, config=cfg)
    print(summary)


if __name__ == "__main__":
    main()
