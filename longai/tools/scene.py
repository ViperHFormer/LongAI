from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from longai.tools.base import BaseTool, ToolConfig
from longai.utils.io import read_json, write_json


SCENE_LABELS = [
    "office", "meeting_room", "home", "kitchen",
    "street_outdoor", "vehicle", "cafe_shop", "other",
]

AUDIOSET_SCENE_MAP = {
    "office": ["Office", "Typing", "Computer keyboard"],
    "meeting_room": ["Speech", "Conversation", "Lecture", "Meeting"],
    "home": ["Domestic sounds", "Home", "Television"],
    "kitchen": ["Cooking", "Frying (food)", "Microwave oven"],
    "street_outdoor": ["Traffic noise", "Outside", "Urban or manmade"],
    "vehicle": ["Vehicle", "Car", "Bus", "Train"],
    "cafe_shop": ["Cafeteria", "Restaurant", "Crowd"],
}


class SceneTool(BaseTool):
    labels = SCENE_LABELS

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

    def _map_audioset_to_scene(self, predictions: list[dict]) -> tuple[str, float]:
        scores = {label: 0.0 for label in SCENE_LABELS}
        for pred in predictions:
            label_low = pred["label"].lower()
            for scene_label, audioset_keys in AUDIOSET_SCENE_MAP.items():
                if any(k.lower() in label_low for k in audioset_keys):
                    scores[scene_label] += pred["score"]
        best = max(scores, key=scores.get)
        conf = round(min(scores[best], 0.95), 3)
        if scores[best] == 0.0:
            return "other", 0.3
        return best, conf

    def _run_mock(self, session_id: str, wav_path: Path, vad_result: dict, context_sec: float = 5.0) -> list[dict]:
        wav, sr = sf.read(wav_path)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        duration = len(wav) / float(sr)

        items = []
        for idx, seg in enumerate(vad_result.get("segments", [])):
            start = max(0.0, seg["start"] - context_sec)
            end = min(duration, seg["end"] + context_sec)
            a = int(start * sr)
            b = int(end * sr)
            chunk = wav[a:b] if b > a else wav
            energy = float(np.sqrt(np.mean(np.square(chunk)))) if len(chunk) else 0.0
            zcr = float(np.mean(np.abs(np.diff(np.sign(chunk))))) if len(chunk) > 1 else 0.0
            if energy < 0.005:
                label = "office"
            elif zcr > 0.5:
                label = "street_outdoor"
            elif energy > 0.03:
                label = "cafe_shop"
            else:
                label = "other"
            items.append({"segment_id": f"{session_id}_seg_{idx:04d}", "start": start, "end": end, "label": label, "confidence": 0.45})

        return items

    def _run_hf(self, session_id: str, wav_path: Path, vad_result: dict, context_sec: float = 5.0) -> list[dict]:
        classifier = self._get_classifier()
        wav, sr = sf.read(str(wav_path))
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        duration = len(wav) / float(sr)

        items = []
        for idx, seg in enumerate(vad_result.get("segments", [])):
            seg_id = f"{session_id}_seg_{idx:04d}"
            start = max(0.0, seg["start"] - context_sec)
            end = min(duration, seg["end"] + context_sec)
            a = int(start * sr)
            b = int(end * sr)
            chunk = wav[a:b] if b > a else wav

            if len(chunk) < sr * 0.5:
                items.append({"segment_id": seg_id, "start": start, "end": end, "label": "other", "confidence": 0.3})
                continue

            try:
                preds = classifier(chunk.astype(np.float32), sampling_rate=sr, top_k=10)
                label, conf = self._map_audioset_to_scene(preds)
            except Exception:
                label, conf = "other", 0.3

            items.append({"segment_id": seg_id, "start": start, "end": end, "label": label, "confidence": conf})

        return items

    def run(self, session_id: str, wav_path: Path, vad_result: dict, context_sec: float = 5.0, force: bool = False) -> dict:
        cache_path = self.config.cache_dir / f"{session_id}.json" if self.config.cache_dir else None
        if cache_path and cache_path.exists() and not force:
            return read_json(cache_path)

        if self.config.backend == "local_hf":
            items = self._run_hf(session_id, wav_path, vad_result, context_sec)
        else:
            items = self._run_mock(session_id, wav_path, vad_result, context_sec)

        out = {"session_id": session_id, "segments": items}
        if cache_path:
            write_json(cache_path, out)
        return out
