#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from longai.train.reasoner_sft import run_reasoner_sft


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", type=Path, default=Path("data/annotations/reasoner_train.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/train/amr_sft"))
    args = parser.parse_args()

    summary = run_reasoner_sft(args.train_jsonl, args.output_dir)
    print(summary)


if __name__ == "__main__":
    main()
