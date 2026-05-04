from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class VideoMeta:
    path: Path
    duration: float
    has_audio: bool


def _run_ffprobe(path: Path) -> tuple[float, bool]:
    duration_cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    audio_cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a",
        "-show_entries",
        "stream=codec_type",
        "-of",
        "csv=p=0",
        str(path),
    ]
    try:
        duration_raw = subprocess.check_output(duration_cmd, text=True).strip()
        duration = float(duration_raw) if duration_raw else 0.0
    except Exception:
        duration = 0.0
    try:
        audio_raw = subprocess.check_output(audio_cmd, text=True).strip()
        has_audio = bool(audio_raw)
    except Exception:
        has_audio = False
    return duration, has_audio


def scan_videos(source_dir: Path, suffixes: tuple[str, ...] = (".mp4", ".mov", ".mkv")) -> list[VideoMeta]:
    metas: list[VideoMeta] = []
    for path in sorted(source_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in suffixes:
            duration, has_audio = _run_ffprobe(path)
            metas.append(VideoMeta(path=path, duration=duration, has_audio=has_audio))
    return metas


def select_pilot_subset(
    source_dir: Path,
    max_sessions: int = 24,
    min_duration_sec: float = 300.0,
    max_duration_sec: float = 3600.0,
) -> list[dict]:
    selected: list[VideoMeta] = []
    for path in sorted(source_dir.rglob("*")):
        if not (path.is_file() and path.suffix.lower() in (".mp4", ".mov", ".mkv")):
            continue
        duration, has_audio = _run_ffprobe(path)
        if has_audio and duration >= min_duration_sec and duration <= max_duration_sec:
            selected.append(VideoMeta(path=path, duration=duration, has_audio=has_audio))
        if len(selected) >= max_sessions:
            break

    rows = []
    for idx, item in enumerate(selected):
        rows.append(
            {
                "session_id": f"session_{idx:04d}",
                "source_path": str(item.path),
                "duration_sec": round(item.duration, 3),
                "wearer_id": "unknown",
            }
        )
    return rows
