from __future__ import annotations

import subprocess
from pathlib import Path


def wav_duration_sec(wav_path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(wav_path),
    ]
    raw = subprocess.check_output(cmd, text=True).strip()
    return float(raw) if raw else 0.0


def build_session_manifest_entry(
    session_id: str,
    source_path: str,
    wav_path: str,
    wearer_id: str = "unknown",
    clip_boundaries: list[dict] | None = None,
) -> dict:
    wav = Path(wav_path)
    return {
        "session_id": session_id,
        "source_path": source_path,
        "wav_path": wav_path,
        "duration_sec": round(wav_duration_sec(wav), 3),
        "wearer_id": wearer_id,
        "clip_boundaries": clip_boundaries or [],
    }
