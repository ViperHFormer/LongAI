#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from longai.data.audio_extract import extract_audio
from longai.utils.io import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/raw/pilot_subset_manifest.jsonl"))
    parser.add_argument("--audio-dir", type=Path, default=Path("data/processed/audio"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/audio_manifest.jsonl"))
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    rows = read_jsonl(args.manifest)
    out_rows = []
    for row in rows:
        sid = row["session_id"]
        src = Path(row["source_path"])
        wav = args.audio_dir / f"{sid}.wav"
        extract_audio(src, wav, sample_rate=args.sample_rate, force=args.force)
        new_row = dict(row)
        new_row["wav_path"] = str(wav)
        out_rows.append(new_row)

    write_jsonl(args.output, out_rows)
    print(f"audio_extracted={len(out_rows)} -> {args.output}")


if __name__ == "__main__":
    main()
