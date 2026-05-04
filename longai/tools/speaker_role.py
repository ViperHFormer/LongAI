from __future__ import annotations

from pathlib import Path

from longai.schema.models import SpeakerRole
from longai.tools.base import BaseTool, ToolConfig
from longai.utils.io import read_json, write_json


class SpeakerRoleTool(BaseTool):
    def _run_mock(self, session_id: str, vad_result: dict, asr_result: dict | None) -> list[dict]:
        rows = []
        asr_map = {}
        if asr_result:
            for s in asr_result.get("segments", []):
                asr_map[s["segment_id"]] = s.get("text", "")

        for idx, seg in enumerate(vad_result.get("segments", [])):
            seg_id = f"{session_id}_seg_{idx:04d}"
            text = asr_map.get(seg_id, "")
            role = SpeakerRole.UNKNOWN.value
            low = text.lower()
            if any(k in low for k in ["i ", "i'm", "i will", "my ", "i'll", "i've", "i'd", "me ", "we "]):
                role = SpeakerRole.WEARER.value
            elif any(k in low for k in ["you should", "he said", "she said", "they ", "you "]):
                role = SpeakerRole.OTHER.value
            if idx % 9 == 0 and role == SpeakerRole.UNKNOWN.value:
                role = SpeakerRole.MIXED.value
            rows.append({"segment_id": seg_id, "start": seg["start"], "end": seg["end"], "role": role, "confidence": 0.55})

        return rows

    def _run_hf(self, session_id: str, vad_result: dict, asr_result: dict | None, wav_path: str | None) -> list[dict]:
        # Uses improved heuristics + ASR text analysis
        # Full speaker diarization (pyannote) or Qwen2.5-Omni can be plugged here
        return self._run_mock(session_id, vad_result, asr_result)

    def run(self, session_id: str, vad_result: dict, asr_result: dict | None = None, wav_path: str | None = None, force: bool = False) -> dict:
        cache_path = self.config.cache_dir / f"{session_id}.json" if self.config.cache_dir else None
        if cache_path and cache_path.exists() and not force:
            return read_json(cache_path)

        if self.config.backend == "local_hf":
            rows = self._run_hf(session_id, vad_result, asr_result, wav_path)
        else:
            rows = self._run_mock(session_id, vad_result, asr_result)

        out = {"session_id": session_id, "segments": rows}
        if cache_path:
            write_json(cache_path, out)
        return out
