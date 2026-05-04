from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from longai.tools.base import BaseTool, ToolConfig
from longai.utils.io import read_json, write_json


class VADTool(BaseTool):
    def __init__(self, config: ToolConfig, frame_sec: float = 0.5, threshold: float = 0.01):
        super().__init__(config)
        self.frame_sec = frame_sec
        self.threshold = threshold
        self._silero_model = None

    def _get_silero_model(self):
        if self._silero_model is None:
            from silero_vad import load_silero_vad
            self._silero_model = load_silero_vad()
        return self._silero_model

    def _run_mock(self, wav_path: Path) -> dict:
        data, sr = sf.read(wav_path)
        if data.ndim > 1:
            data = data.mean(axis=1)
        frame = max(1, int(sr * self.frame_sec))
        energies = []
        for i in range(0, len(data), frame):
            chunk = data[i : i + frame]
            if len(chunk) == 0:
                continue
            energies.append(float(np.sqrt(np.mean(np.square(chunk)))))

        segments = []
        active = False
        start_t = 0.0
        for i, e in enumerate(energies):
            t0 = i * self.frame_sec
            t1 = (i + 1) * self.frame_sec
            if e >= self.threshold and not active:
                active = True
                start_t = t0
            if e < self.threshold and active:
                active = False
                segments.append({"start": round(start_t, 3), "end": round(t0, 3), "confidence": 0.7})
            if i == len(energies) - 1 and active:
                segments.append({"start": round(start_t, 3), "end": round(t1, 3), "confidence": 0.7})

        if not segments:
            duration = len(data) / float(sr)
            segments = [{"start": 0.0, "end": round(duration, 3), "confidence": 0.3}]
        return segments

    def _run_silero(self, wav_path: Path) -> dict:
        from silero_vad import get_speech_timestamps, read_audio
        model = self._get_silero_model()
        wav = read_audio(str(wav_path), sampling_rate=16000)
        raw = get_speech_timestamps(wav, model, return_seconds=True)
        segments = [
            {"start": round(s["start"], 3), "end": round(s["end"], 3), "confidence": 0.85}
            for s in raw
        ]
        if not segments:
            duration = len(wav) / 16000.0
            segments = [{"start": 0.0, "end": round(duration, 3), "confidence": 0.3}]
        return segments

    def run(self, session_id: str, wav_path: Path, force: bool = False) -> dict:
        cache_path = self.config.cache_dir / f"{session_id}.json" if self.config.cache_dir else None
        if cache_path and cache_path.exists() and not force:
            return read_json(cache_path)

        if self.config.backend == "local_hf":
            segments = self._run_silero(wav_path)
        else:
            segments = self._run_mock(wav_path)

        out = {"session_id": session_id, "segments": segments}
        if cache_path:
            write_json(cache_path, out)
        return out
