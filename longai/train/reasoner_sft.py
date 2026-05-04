from __future__ import annotations

from pathlib import Path

from longai.utils.io import read_jsonl, write_json


def run_reasoner_sft(train_path: Path, output_dir: Path) -> dict:
    rows = read_jsonl(train_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "placeholder",
        "message": "Reasoner SFT scaffold ready.",
        "num_examples": len(rows),
    }
    write_json(output_dir / "reasoner_sft_summary.json", summary)
    return summary
