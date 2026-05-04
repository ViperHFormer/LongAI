from __future__ import annotations

import subprocess
from pathlib import Path


def extract_audio(
    source_path: Path,
    output_wav: Path,
    sample_rate: int = 16000,
    force: bool = False,
) -> None:
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    if output_wav.exists() and not force:
        return
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(source_path),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        str(output_wav),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
