from __future__ import annotations

from pathlib import Path

from longai.schema.models import EvidencePack, SpeakerRole
from longai.utils.io import write_jsonl


def build_evidence_packs(
    session_id: str,
    wav_path: str,
    vad_result: dict,
    asr_result: dict,
    speaker_role_result: dict,
    scene_result: dict,
    event_result: dict,
    output_path: Path,
) -> list[EvidencePack]:
    role_map = {x["segment_id"]: x for x in speaker_role_result.get("segments", [])}
    asr_map = {x["segment_id"]: x for x in asr_result.get("segments", [])}
    scene_map = {x["segment_id"]: x for x in scene_result.get("segments", [])}
    event_map = {x["segment_id"]: x for x in event_result.get("segments", [])}

    packs: list[EvidencePack] = []
    for idx, seg in enumerate(vad_result.get("segments", [])):
        seg_id = f"{session_id}_seg_{idx:04d}"
        role = role_map.get(seg_id, {}).get("role", "unknown")
        pack = EvidencePack(
            segment_id=seg_id,
            session_id=session_id,
            start_time=seg["start"],
            end_time=seg["end"],
            waveform_path=wav_path,
            speaker_role=SpeakerRole(role),
            asr_text=asr_map.get(seg_id, {}).get("text", ""),
            scene_label=scene_map.get(seg_id, {}).get("label", "other"),
            event_tags=event_map.get(seg_id, {}).get("tags", []),
            affective_tags=[],
            tool_confidences={
                "vad": seg.get("confidence", 0.5),
                "asr": asr_map.get(seg_id, {}).get("confidence", 0.5),
                "speaker": role_map.get(seg_id, {}).get("confidence", 0.5),
                "scene": scene_map.get(seg_id, {}).get("confidence", 0.5),
                "events": event_map.get(seg_id, {}).get("confidence", 0.5),
            },
            context_window_refs=[
                {
                    "left": scene_map.get(seg_id, {}).get("start", seg["start"]),
                    "right": scene_map.get(seg_id, {}).get("end", seg["end"]),
                }
            ],
        )
        packs.append(pack)

    write_jsonl(output_path, [p.model_dump() for p in packs])
    return packs
