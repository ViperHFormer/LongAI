#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from longai.construction.evidence_pack import build_evidence_packs
from longai.tools.asr import ASRTool
from longai.tools.base import ToolConfig
from longai.tools.events import EventTool
from longai.tools.scene import SceneTool
from longai.tools.speaker_role import SpeakerRoleTool
from longai.tools.vad import VADTool
from longai.utils.io import ensure_dir, read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-manifest", type=Path, default=Path("data/processed/session_manifest.jsonl"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--backend", type=str, default="local_hf")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-tools", type=str, default="",
                        help="Comma-separated tools to skip: vad,asr,speaker_role,scene,events")
    parser.add_argument("--tool-backends", type=str, default="",
                        help="Per-tool backend overrides, e.g. 'speaker_role=mock,scene=mock'")
    parser.add_argument("--asr-model-size", type=str, default=None,
                        help="ASR model size: tiny, base, small (default: tiny)")
    args = parser.parse_args()

    skip = set(t.strip() for t in args.skip_tools.split(",") if t.strip())
    tool_backends = {}
    for item in args.tool_backends.split(","):
        item = item.strip()
        if "=" in item:
            k, v = item.split("=", 1)
            tool_backends[k.strip()] = v.strip()

    rows = read_jsonl(args.session_manifest)

    _b = lambda tool: tool_backends.get(tool, args.backend)
    vad_tool = VADTool(ToolConfig(backend=_b("vad"), cache_dir=ensure_dir(args.artifacts_dir / "perception/vad")))
    asr_tool = ASRTool(ToolConfig(backend=_b("asr"), asr_model_size=args.asr_model_size, cache_dir=ensure_dir(args.artifacts_dir / "perception/asr")))
    spk_tool = SpeakerRoleTool(ToolConfig(backend=_b("speaker_role"), cache_dir=ensure_dir(args.artifacts_dir / "perception/speaker_role")))
    scene_tool = SceneTool(ToolConfig(backend=_b("scene"), cache_dir=ensure_dir(args.artifacts_dir / "perception/scene")))
    event_tool = EventTool(ToolConfig(backend=_b("events"), cache_dir=ensure_dir(args.artifacts_dir / "perception/events")))

    SKIP_MOCK = {"segments": [], "confidence": 0.0}

    # Cache SKIP_MOCK results so downstream scripts can read them
    def _run_or_skip(tool, sid, skip_flag, args_list):
        if skip_flag in skip:
            # Always write SKIP_MOCK for skipped tools (ablation: remove tool's contribution)
            if tool.config.cache_dir:
                from longai.utils.io import write_json
                write_json(tool.config.cache_dir / f"{sid}.json", SKIP_MOCK)
            return SKIP_MOCK
        return tool.run(sid, *args_list, force=args.force)

    pack_dir = ensure_dir(args.artifacts_dir / "evidence_packs")
    for row in rows:
        sid = row["session_id"]
        wav = Path(row["wav_path"])

        vad = _run_or_skip(vad_tool, sid, "vad", [wav])
        asr = _run_or_skip(asr_tool, sid, "asr", [vad, wav])
        spk = _run_or_skip(spk_tool, sid, "speaker_role", [vad, asr])
        scene = _run_or_skip(scene_tool, sid, "scene", [wav, vad])
        events = _run_or_skip(event_tool, sid, "events", [wav, vad])

        build_evidence_packs(
            session_id=sid,
            wav_path=str(wav),
            vad_result=vad,
            asr_result=asr,
            speaker_role_result=spk,
            scene_result=scene,
            event_result=events,
            output_path=pack_dir / f"{sid}.jsonl",
        )

    print(f"perception_done sessions={len(rows)}")


if __name__ == "__main__":
    main()
