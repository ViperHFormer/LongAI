#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from longai.eval.tool_metrics import eval_asr, eval_simple_label_accuracy, eval_speaker_role, eval_vad
from longai.utils.io import read_json, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-id", type=str, required=True)
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--gt-tool", type=Path, default=Path("data/annotations/tool_level_gt.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/eval/tools_eval.json"))
    args = parser.parse_args()

    if not args.gt_tool.exists():
        write_json(args.output, {"warning": "GT not found. Skip tool eval."})
        print("GT not found. tools eval skipped.")
        return

    gt = read_json(args.gt_tool)
    pred_vad = read_json(args.artifacts_dir / "perception/vad" / f"{args.session_id}.json")["segments"]
    pred_asr = read_json(args.artifacts_dir / "perception/asr" / f"{args.session_id}.json")["segments"]
    pred_spk = read_json(args.artifacts_dir / "perception/speaker_role" / f"{args.session_id}.json")["segments"]
    pred_scene = read_json(args.artifacts_dir / "perception/scene" / f"{args.session_id}.json")["segments"]

    out = {
        "vad": eval_vad(pred_vad, gt.get("vad_regions", [])),
        "asr": eval_asr(pred_asr, gt.get("asr", [])),
        "speaker_role": eval_speaker_role(pred_spk, gt.get("speaker_role", [])),
        "scene": eval_simple_label_accuracy(pred_scene, gt.get("scene", []), key="label"),
    }
    write_json(args.output, out)
    print(f"tool_eval -> {args.output}")


if __name__ == "__main__":
    main()
