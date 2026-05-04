from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from longai.tools.base import BaseTool, ToolConfig
from longai.utils.io import read_json, write_json


EVENT_TAGS_POOL = [
    "typing", "door", "footsteps", "traffic",
    "kitchen_activity", "appliance", "crowd_chatter", "silence_quiet",
]

AUDIOSET_EVENT_MAP = {
    "typing": ["Typing", "Computer keyboard"],
    "door": ["Door", "Doorbell", "Sliding door"],
    "footsteps": ["Footsteps", "Walking"],
    "traffic": ["Traffic noise", "Vehicle", "Car"],
    "kitchen_activity": ["Cooking", "Frying", "Chopping"],
    "appliance": ["Microwave oven", "Vacuum cleaner", "Dishwasher"],
    "crowd_chatter": ["Chatter", "Conversation", "Crowd", "Babbling"],
    "silence_quiet": ["Silence", "Quiet"],
}


class EventTool(BaseTool):
    def __init__(self, config: ToolConfig):
        super().__init__(config)
        self._classifier = None

    def _get_classifier(self):
        if self._classifier is None:
            from transformers import pipeline
            import torch
            device = 0 if torch.cuda.is_available() else -1
            self._classifier = pipeline(
                "audio-classification",
                model="MIT/ast-finetuned-audioset-10-10-0.4593",
                device=device,
            )
        return self._classifier

    def _map_audioset_to_events(self, predictions: list[dict], threshold: float = 0.15) -> list[str]:
        scores = {tag: 0.0 for tag in EVENT_TAGS_POOL}
        for pred in predictions:
            label_low = pred["label"].lower()
            for tag, audioset_keys in AUDIOSET_EVENT_MAP.items():
                if any(k.lower() in label_low for k in audioset_keys):
                    scores[tag] = max(scores[tag], pred["score"])
        result = sorted([t for t, s in scores.items() if s >= threshold], key=lambda t: scores[t], reverse=True)
        return result[:3] if result else ["silence_quiet"]

    def _run_mock(self, session_id: str, wav_path: Path, vad_result: dict) -> list[dict]:
        wav, sr = sf.read(wav_path)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)

        rows = []
        for idx, seg in enumerate(vad_result.get("segments", [])):
            a = int(seg["start"] * sr)
            b = int(seg["end"] * sr)
            chunk = wav[a:b] if b > a else wav
            energy = float(np.sqrt(np.mean(np.square(chunk)))) if len(chunk) else 0.0
            if energy < 0.004:
                tags = ["silence_quiet"]
            elif energy > 0.04:
                tags = ["crowd_chatter", "traffic"]
            elif seg["end"] - seg["start"] > 6:
                tags = ["typing"]
            else:
                tags = ["appliance"]
            rows.append({"segment_id": f"{session_id}_seg_{idx:04d}", "tags": tags, "confidence": 0.42})

        return rows

    def _run_hf(self, session_id: str, wav_path: Path, vad_result: dict, context_sec: float = 5.0) -> list[dict]:
        classifier = self._get_classifier()
        wav, sr = sf.read(str(wav_path))
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        duration = len(wav) / float(sr)

        rows = []
        for idx, seg in enumerate(vad_result.get("segments", [])):
            seg_id = f"{session_id}_seg_{idx:04d}"
            start = max(0.0, seg["start"] - context_sec)
            end = min(duration, seg["end"] + context_sec)
            a = int(start * sr)
            b = int(end * sr)
            chunk = wav[a:b] if b > a else wav

            if len(chunk) < sr * 0.5:
                rows.append({"segment_id": seg_id, "tags": ["silence_quiet"], "confidence": 0.3})
                continue

            try:
                preds = classifier(chunk.astype(np.float32), sampling_rate=sr, top_k=15)
                tags = self._map_audioset_to_events(preds)
                conf = 0.5
            except Exception:
                tags, conf = ["silence_quiet"], 0.3

            rows.append({"segment_id": seg_id, "tags": tags, "confidence": conf})

        return rows

    def run(self, session_id: str, wav_path: Path, vad_result: dict, context_sec: float = 5.0, force: bool = False) -> dict:
        cache_path = self.config.cache_dir / f"{session_id}.json" if self.config.cache_dir else None
        if cache_path and cache_path.exists() and not force:
            return read_json(cache_path)

        if self.config.backend == "local_hf":
            rows = self._run_hf(session_id, wav_path, vad_result, context_sec)
        else:
            rows = self._run_mock(session_id, wav_path, vad_result)

        out = {"session_id": session_id, "segments": rows}
        if cache_path:
            write_json(cache_path, out)
        return out
