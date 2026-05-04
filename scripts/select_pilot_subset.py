#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from longai.data.subset import select_pilot_subset
from longai.utils.io import write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/raw/pilot_subset_manifest.jsonl"))
    parser.add_argument("--max-sessions", type=int, default=24)
    parser.add_argument("--min-duration-sec", type=float, default=300.0)
    parser.add_argument("--max-duration-sec", type=float, default=3600.0)
    args = parser.parse_args()

    rows = select_pilot_subset(
        source_dir=args.source_dir,
        max_sessions=args.max_sessions,
        min_duration_sec=args.min_duration_sec,
        max_duration_sec=args.max_duration_sec,
    )
    write_jsonl(args.output, rows)
    print(f"selected={len(rows)} -> {args.output}")


if __name__ == "__main__":
    main()
