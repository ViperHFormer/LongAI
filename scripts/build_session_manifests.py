#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from longai.data.session_manifest import build_session_manifest_entry
from longai.utils.io import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-manifest", type=Path, default=Path("data/processed/audio_manifest.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/session_manifest.jsonl"))
    args = parser.parse_args()

    rows = read_jsonl(args.audio_manifest)
    session_rows = []
    for row in rows:
        session_rows.append(
            build_session_manifest_entry(
                session_id=row["session_id"],
                source_path=row["source_path"],
                wav_path=row["wav_path"],
                wearer_id=row.get("wearer_id", "unknown"),
                clip_boundaries=row.get("clip_boundaries", []),
            )
        )

    write_jsonl(args.output, session_rows)
    print(f"sessions={len(session_rows)} -> {args.output}")


if __name__ == "__main__":
    main()
