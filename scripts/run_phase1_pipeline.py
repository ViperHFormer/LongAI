#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=str, required=True)
    parser.add_argument("--max-sessions", type=int, default=24)
    args = parser.parse_args()

    run(["python", "scripts/select_pilot_subset.py", "--source-dir", args.source_dir, "--max-sessions", str(args.max_sessions)])
    run(["python", "scripts/extract_audio.py"])
    run(["python", "scripts/build_session_manifests.py"])
    run(["python", "scripts/init_annotation_templates.py"])
    run(["python", "scripts/run_perception.py"])
    run(["python", "scripts/build_memory.py"])
    run(["python", "scripts/run_reasoning.py"])


if __name__ == "__main__":
    main()
