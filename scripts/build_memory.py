#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from longai.memory.manager import build_memory_from_evidence
from longai.utils.io import ensure_dir, read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-manifest", type=Path, default=Path("data/processed/session_manifest.jsonl"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--backend", type=str, default="mock")
    args = parser.parse_args()

    rows = read_jsonl(args.session_manifest)
    graph_dir = ensure_dir(args.artifacts_dir / "memory/session_graphs")
    log_dir = ensure_dir(args.artifacts_dir / "memory/update_logs")
    packs_dir = args.artifacts_dir / "evidence_packs"

    for row in rows:
        sid = row["session_id"]
        pack_path = packs_dir / f"{sid}.jsonl"
        build_memory_from_evidence(
            evidence_pack_path=pack_path,
            graph_out=graph_dir / f"{sid}.json",
            log_out=log_dir / f"{sid}.jsonl",
            backend=args.backend,
        )

    print(f"memory_built sessions={len(rows)}")


if __name__ == "__main__":
    main()
