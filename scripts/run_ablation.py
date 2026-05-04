#!/usr/bin/env python
"""Run ablation studies: systematically vary tool/config combinations and compare results."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime

from longai.utils.io import ensure_dir, read_json, write_json

ABLATION_CONFIGS = {
    # --- Tool presence ablations ---
    "full": {
        "desc": "Full pipeline (baseline): all 5 tools, base ASR, keyword MCoC",
        "skip_tools": "",
        "tool_backends": "",
        "backend": "local_hf",
        "mcoc_backend": "mock",
        "amr_backend": "mock",
    },
    "transcript_only": {
        "desc": "Transcript-only: VAD + ASR, skip scene/events/speaker_role",
        "skip_tools": "scene,events,speaker_role",
        "tool_backends": "",
        "backend": "local_hf",
        "mcoc_backend": "mock",
        "amr_backend": "mock",
    },
    "no_speaker_role": {
        "desc": "Without speaker role (−speaker_role)",
        "skip_tools": "speaker_role",
        "tool_backends": "",
        "backend": "local_hf",
        "mcoc_backend": "mock",
        "amr_backend": "mock",
    },
    "no_scene": {
        "desc": "Without scene classification (−scene)",
        "skip_tools": "scene",
        "tool_backends": "",
        "backend": "local_hf",
        "mcoc_backend": "mock",
        "amr_backend": "mock",
    },
    "no_events": {
        "desc": "Without event detection (−events)",
        "skip_tools": "events",
        "tool_backends": "",
        "backend": "local_hf",
        "mcoc_backend": "mock",
        "amr_backend": "mock",
    },
    "no_scene_events": {
        "desc": "Without scene and events (−scene −events)",
        "skip_tools": "scene,events",
        "tool_backends": "",
        "backend": "local_hf",
        "mcoc_backend": "mock",
        "amr_backend": "mock",
    },

    # --- ASR model size ablation ---
    "asr_tiny": {
        "desc": "ASR: faster-whisper-tiny (39M) instead of base",
        "skip_tools": "",
        "tool_backends": "",
        "backend": "local_hf",
        "mcoc_backend": "mock",
        "amr_backend": "mock",
        "asr_model_size": "tiny",
    },

    # --- Backend ablations ---
    "mock_all": {
        "desc": "All mock backends (rule-based VAD/ASR/MCoC/AMR)",
        "skip_tools": "",
        "tool_backends": "",
        "backend": "mock",
        "mcoc_backend": "mock",
        "amr_backend": "mock",
    },
    "llm_mcoc": {
        "desc": "LLM-based MCoC (Qwen2.5-7B) with real perception",
        "skip_tools": "",
        "tool_backends": "",
        "backend": "local_hf",
        "mcoc_backend": "local_hf",
        "amr_backend": "mock",
    },
    "llm_amr": {
        "desc": "LLM-based AMR reasoning with real perception",
        "skip_tools": "",
        "tool_backends": "",
        "backend": "local_hf",
        "mcoc_backend": "mock",
        "amr_backend": "local_hf",
    },
    "llm_full": {
        "desc": "LLM-based MCoC + AMR with real perception",
        "skip_tools": "",
        "tool_backends": "",
        "backend": "local_hf",
        "mcoc_backend": "local_hf",
        "amr_backend": "local_hf",
    },
}


def run_cmd(cmd: list[str], desc: str = "") -> bool:
    label = f" [{desc}]" if desc else ""
    print(f"  RUN{'=' + label[1:] if desc else ''}: {' '.join(str(x) for x in cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if result.returncode != 0:
        print(f"  FAILED: {result.stderr[:300]}")
        return False
    for line in result.stdout.strip().splitlines():
        print(f"    | {line}")
    return True


def run_ablation_config(
    name: str,
    cfg: dict,
    session_manifest: Path,
    base_dir: Path,
    max_sessions: int = 0,
) -> dict:
    """Run the full pipeline for one ablation config. Returns eval stats."""
    artifacts_dir = ensure_dir(base_dir / name)
    _rel = lambda sub: artifacts_dir / sub  # noqa

    python = sys.executable
    scripts = Path("scripts")

    skip = cfg["skip_tools"]
    tool_be = cfg["tool_backends"]
    be = cfg["backend"]

    # Build perception args
    perc_args = [
        python, str(scripts / "run_perception.py"),
        "--session-manifest", str(session_manifest),
        "--artifacts-dir", str(artifacts_dir),
        "--backend", be,
    ]
    if skip:
        perc_args += ["--skip-tools", skip]
    if tool_be:
        perc_args += ["--tool-backends", tool_be]
    if cfg.get("asr_model_size"):
        perc_args += ["--asr-model-size", cfg["asr_model_size"]]

    if not run_cmd(perc_args, f"{name}/perception"):
        return {"error": "perception failed"}

    # Build memory
    mem_args = [
        python, str(scripts / "build_memory.py"),
        "--session-manifest", str(session_manifest),
        "--artifacts-dir", str(artifacts_dir),
        "--backend", cfg["mcoc_backend"],
    ]
    if not run_cmd(mem_args, f"{name}/memory"):
        return {"error": "memory failed"}

    # Run reasoning
    reason_args = [
        python, str(scripts / "run_reasoning.py"),
        "--session-manifest", str(session_manifest),
        "--artifacts-dir", str(artifacts_dir),
        "--experiment-name", "amr_full",
        "--backend", cfg["amr_backend"],
    ]
    if not run_cmd(reason_args, f"{name}/reasoning"):
        return {"error": "reasoning failed"}

    # Run comprehensive eval
    eval_out = artifacts_dir / "eval" / "comprehensive_eval.json"
    eval_args = [
        python, str(scripts / "run_comprehensive_eval.py"),
        "--session-manifest", str(session_manifest),
        "--artifacts-dir", str(artifacts_dir),
        "--output", str(eval_out),
    ]
    if not run_cmd(eval_args, f"{name}/eval"):
        return {"error": "eval failed"}

    return read_json(eval_out)


def collect_metrics(eval_data: dict) -> dict:
    """Extract key metrics from comprehensive eval output."""
    ts = eval_data.get("tool_stats", {})
    ms = eval_data.get("memory_stats", {})
    rs = eval_data.get("reasoning_stats", {})

    # Aggregate tool stats
    asr_vals = list(ts.get("asr", {}).values())
    total_nonempty = sum(a.get("nonempty", 0) for a in asr_vals)
    total_segs = sum(a.get("segments", 0) for a in asr_vals)
    avg_asr_conf = sum(a.get("avg_confidence", 0) for a in asr_vals) / len(asr_vals) if asr_vals else 0

    # Aggregate memory stats
    mem_vals = list(ms.values())
    total_nodes = sum(m.get("total_nodes", 0) for m in mem_vals)
    total_edges = sum(m.get("total_edges", 0) for m in mem_vals)
    total_intents = sum(m.get("intent_count", 0) for m in mem_vals)
    total_entities = sum(m.get("entity_count", 0) for m in mem_vals)

    # Aggregate reasoning stats
    reason_vals = list(rs.values())
    avg_confidence = sum(r.get("avg_confidence", 0) for r in reason_vals) / len(reason_vals) if reason_vals else 0
    avg_abstain = sum(r.get("abstain_rate", 0) for r in reason_vals) / len(reason_vals) if reason_vals else 0
    avg_trace = sum(r.get("avg_graph_trace_len", 0) for r in reason_vals) / len(reason_vals) if reason_vals else 0

    return {
        "vad_segments": ts.get("vad", {}).get("total_segments", 0),
        "speech_ratio": ts.get("vad", {}).get("speech_ratio", 0),
        "asr_nonempty_ratio": round(total_nonempty / total_segs, 3) if total_segs else 0,
        "asr_avg_confidence": round(avg_asr_conf, 3),
        "memory_total_nodes": total_nodes,
        "memory_total_edges": total_edges,
        "memory_total_intents": total_intents,
        "memory_total_entities": total_entities,
        "reasoning_avg_confidence": round(avg_confidence, 3),
        "reasoning_avg_abstain_rate": round(avg_abstain, 3),
        "reasoning_avg_trace_len": round(avg_trace, 1),
    }


def print_comparison_table(results: dict[str, dict]) -> None:
    """Print a formatted comparison table."""
    metrics = [
        "vad_segments", "speech_ratio", "asr_nonempty_ratio", "asr_avg_confidence",
        "memory_total_nodes", "memory_total_edges", "memory_total_intents", "memory_total_entities",
        "reasoning_avg_confidence", "reasoning_avg_abstain_rate", "reasoning_avg_trace_len",
    ]
    headers = ["Config", "VAD Seg", "Speech%", "ASR NonEmpty%", "ASR Conf",
               "Nodes", "Edges", "Intents", "Entities",
               "Reas.Conf", "Abstain%", "TraceLen"]

    print("\n" + "=" * 140)
    print("ABLATION COMPARISON TABLE")
    print("=" * 140)
    header_line = "".join(f"{h:>13s}" for h in headers)
    print(header_line)
    print("-" * len(header_line))

    config_names = [n for n in ABLATION_CONFIGS if n in results and "error" not in results[n]]
    baseline = results.get("full", {})
    baseline_vals = {m: baseline.get(m, 0) for m in metrics}

    for name in config_names:
        r = results[name]
        vals = [name[:12]]
        for m in metrics:
            v = r.get(m, 0)
            if isinstance(v, float):
                vals.append(f"{v:.3f}")
            else:
                vals.append(str(v))
        line = "".join(f"{v:>13s}" for v in vals)
        print(line)

    print("-" * len(header_line))
    print("Delta rows (vs full baseline):")
    print()

    for name in config_names:
        if name == "full":
            continue
        r = results[name]

        deltas = []
        for m in metrics:
            bv = baseline_vals.get(m, 0)
            cv = r.get(m, 0)
            if isinstance(bv, float) and bv != 0:
                delta_pct = (cv - bv) / abs(bv) * 100
                deltas.append(f"{delta_pct:+.1f}%")
            elif isinstance(bv, int) and bv != 0:
                delta_pct = (cv - bv) / bv * 100
                deltas.append(f"{delta_pct:+.1f}%")
            else:
                deltas.append("N/A")

        delta_line = "".join(f"{d:>13s}" for d in [name[:12]] + deltas)
        print(delta_line)

    print("=" * 140)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LongAI ablation studies")
    parser.add_argument("--session-manifest", type=Path, default=Path("data/processed/session_manifest.jsonl"))
    parser.add_argument("--base-dir", type=Path, default=Path("artifacts/ablation"))
    parser.add_argument("--configs", type=str, default="",
                        help="Comma-separated config names to run (default: all)")
    parser.add_argument("--max-sessions", type=int, default=0,
                        help="Limit to first N sessions (0=all)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip configs that already have eval output")
    args = parser.parse_args()

    # Filter manifest if max_sessions specified
    manifest_path = args.session_manifest
    if args.max_sessions > 0:
        from longai.utils.io import read_jsonl, write_jsonl
        rows = read_jsonl(args.session_manifest)
        filtered = rows[:args.max_sessions]
        tmp_manifest = args.base_dir / "session_manifest_filtered.jsonl"
        ensure_dir(args.base_dir)
        write_jsonl(tmp_manifest, filtered)
        manifest_path = tmp_manifest
        print(f"Limited to {len(filtered)} sessions via {tmp_manifest}")

    config_names = [n.strip() for n in args.configs.split(",") if n.strip()] if args.configs else list(ABLATION_CONFIGS)

    print(f"Ablation study: {len(config_names)} configurations on {manifest_path}")
    print(f"Output base: {args.base_dir}")
    print()

    results = {}
    for i, name in enumerate(config_names):
        cfg = ABLATION_CONFIGS[name]
        existing_eval = args.base_dir / name / "eval" / "comprehensive_eval.json"
        if args.skip_existing and existing_eval.exists():
            print(f"[{i+1}/{len(config_names)}] {name}: {cfg['desc']}  (cached)")
            results[name] = read_json(existing_eval)
            results[name]["_metrics"] = collect_metrics(results[name])
            continue

        print(f"[{i+1}/{len(config_names)}] {name}: {cfg['desc']}")
        eval_data = run_ablation_config(name, cfg, manifest_path, args.base_dir, args.max_sessions)
        eval_data["_metrics"] = collect_metrics(eval_data)
        results[name] = eval_data
        print()

    # Build comparison
    comparison = {
        "timestamp": datetime.now().isoformat(),
        "configs": {name: ABLATION_CONFIGS[name]["desc"] for name in config_names},
        "metrics": {name: results[name].get("_metrics", results[name]) for name in config_names},
    }
    out_path = args.base_dir / "ablation_comparison.json"
    write_json(ensure_dir(args.base_dir) / "ablation_comparison.json", comparison)
    print(f"\nComparison saved to {out_path}")

    # Print comparison table
    metrics_map = {name: results[name].get("_metrics", results[name]) for name in config_names}
    print_comparison_table(metrics_map)


if __name__ == "__main__":
    main()
