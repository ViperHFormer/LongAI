from __future__ import annotations

from pathlib import Path

from longai.utils.io import write_json


def run_reasoner_rl(output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "placeholder",
        "message": "Lightweight GRPO/DPO hook ready. Implement reward funcs before training.",
    }
    write_json(output_dir / "reasoner_rl_summary.json", summary)
    return summary
