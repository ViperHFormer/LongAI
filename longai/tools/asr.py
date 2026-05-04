from __future__ import annotations

from pathlib import Path

from longai.tools.base import BaseTool, ToolConfig
from longai.utils.io import read_json, write_json


class ASRTool(BaseTool):
    def __init__(self, config: ToolConfig, model_name: str | None = None):
        super().__init__(config)
        import os
        self.model_name = model_name or config.asr_model_size or os.environ.get("LONGAI_ASR_MODEL_SIZE", "tiny")
        self._whisper_model = None

    def _get_whisper_model(self):
        if self._whisper_model is None:
            from faster_whisper import WhisperModel
            import torch
            self._whisper_model = WhisperModel(
                self.model_name,
                device="cuda" if torch.cuda.is_available() else "cpu",
                compute_type="float16" if torch.cuda.is_available() else "int8",
            )
        return self._whisper_model

    def _run_mock(self, vad_result: dict) -> dict:
        segments = []
        for idx, seg in enumerate(vad_result.get("segments", [])):
            seg_id = f"{vad_result.get('session_id', 's')}_seg_{idx:04d}"
            duration = seg["end"] - seg["start"]
            if duration > 8:
                text = "I will finish the task and prepare for the next meeting"
            elif duration > 3:
                text = "Need to check schedule and call teammate"
            else:
                text = "ok"
            segments.append({
                "segment_id": seg_id,
                "start": seg["start"],
                "end": seg["end"],
                "text": text,
                "confidence": 0.5,
            })
        return segments

    def _run_whisper(self, session_id: str, vad_result: dict, wav_path: Path) -> dict:
        model = self._get_whisper_model()
        segments_full, info = model.transcribe(str(wav_path), beam_size=5, word_timestamps=True)
        duration = info.duration

        # Build word timeline
        words = []
        for seg in segments_full:
            if seg.words:
                for w in seg.words:
                    words.append({"start": w.start, "end": w.end, "word": w.word, "prob": w.probability})

        # Align to VAD segments
        out_segments = []
        for idx, vad_seg in enumerate(vad_result.get("segments", [])):
            seg_id = f"{session_id}_seg_{idx:04d}"
            s0, s1 = vad_seg["start"], vad_seg["end"]
            seg_words = [w for w in words if w["start"] >= s0 - 0.1 and w["end"] <= s1 + 0.1]
            text = " ".join(w["word"] for w in seg_words).strip()
            avg_conf = sum(w["prob"] for w in seg_words) / len(seg_words) if seg_words else 0.5
            if not text:
                # Fallback: get any words overlapping this segment
                overlapping = [w for w in words if w["end"] > s0 and w["start"] < s1]
                text = " ".join(w["word"] for w in overlapping).strip()
                avg_conf = sum(w["prob"] for w in overlapping) / len(overlapping) if overlapping else 0.3

            out_segments.append({
                "segment_id": seg_id,
                "start": vad_seg["start"],
                "end": vad_seg["end"],
                "text": text if text else "...",
                "confidence": round(avg_conf, 3),
            })

        return out_segments

    def run(self, session_id: str, vad_result: dict, wav_path: Path, force: bool = False) -> dict:
        cache_path = self.config.cache_dir / f"{session_id}.json" if self.config.cache_dir else None
        if cache_path and cache_path.exists() and not force:
            return read_json(cache_path)

        if self.config.backend == "local_hf":
            segments = self._run_whisper(session_id, vad_result, wav_path)
        else:
            segments = self._run_mock(vad_result)

        out = {
            "session_id": session_id,
            "backend": self.config.backend,
            "model": self.model_name if self.config.backend == "local_hf" else "mock-asr",
            "segments": segments,
        }
        if cache_path:
            write_json(cache_path, out)
        return out
