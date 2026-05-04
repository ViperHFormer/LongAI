#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from longai.train.reasoner_rl import run_reasoner_rl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/train/amr_rl"))
    args = parser.parse_args()

    summary = run_reasoner_rl(args.output_dir)
    print(summary)


if __name__ == "__main__":
    main()
