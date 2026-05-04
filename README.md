# LongAI

LongAI is an audio-first research codebase for building a streaming personal intent memory and answering memory-grounded queries.

The implementation follows three stages:

1. Phase 1 (run-first): data subset -> audio extraction -> perception -> EvidencePack -> MCoC heuristic writer -> memory graph -> AMR baseline.
2. Phase 2 (writer training): convert graph GT to write operations and train a 7B/8B construction writer.
3. Phase 3 (reasoner training): SFT reasoning trajectories, then optional lightweight GRPO/DPO.

## Core Design Constraints

- Audio-first, no video dependency at inference.
- Speech segment is the primary semantic unit.
- Persistent memory graph is separated from query-time reasoning scaffold.
- Compact schema with explicit observability and confidence.

## Project Layout

```text
LongAI/
├── configs/
├── data/
├── artifacts/
├── longai/
├── scripts/
└── tests/
```

## Environment

Conda env: `longai`

Install deps:

```bash
pip install -r requirements.txt
```

Optional tools:

- `ffmpeg` and `ffprobe` must be available in PATH.
- GPU execution can be controlled via `CUDA_VISIBLE_DEVICES`.

## Phase 1: Run End-to-End (No Training)

Ego4D root (provided by user):

`/aiot-hdd-nas-hk01/helixing/ego4d_data/v2/video_540ss`

Step-by-step:

```bash
python scripts/select_pilot_subset.py \
	--source-dir "/aiot-hdd-nas-hk01/helixing/ego4d_data/v2/video_540ss" \
	--max-sessions 24

python scripts/extract_audio.py
python scripts/build_session_manifests.py
python scripts/init_annotation_templates.py
python scripts/run_perception.py --backend local_hf
python scripts/build_memory.py
python scripts/run_reasoning.py --experiment-name amr_full
```

One-command runner:

```bash
python scripts/run_phase1_pipeline.py \
	--source-dir "/aiot-hdd-nas-hk01/helixing/ego4d_data/v2/video_540ss" \
	--max-sessions 24
```

Outputs:

- `artifacts/perception/`
- `artifacts/evidence_packs/`
- `artifacts/memory/`
- `artifacts/reasoning/`
- `artifacts/eval/`
- `artifacts/train/`

## Evaluation

Tool-level (requires GT file):

```bash
python scripts/evaluate_tools.py --session-id session_0000
```

Memory-level (requires GT file):

```bash
python scripts/evaluate_memory.py --session-id session_0000
```

Reasoning-level (requires GT file):

```bash
python scripts/evaluate_reasoning.py --session-id session_0000
```

## Training Scaffolds

Phase2 writer SFT data prep (from phase1 artifacts/update logs):

```bash
python scripts/prepare_writer_sft_data.py \
	--mode update-logs \
	--update-logs-dir artifacts/memory/update_logs \
	--evidence-pack-dir artifacts/evidence_packs \
	--all-output data/annotations/writer_all.jsonl \
	--train-output data/annotations/writer_train.jsonl \
	--eval-output data/annotations/writer_eval.jsonl
```

If you already have graph GT and want strict GT-to-ops conversion:

```bash
python scripts/prepare_writer_sft_data.py \
	--mode graph-gt \
	--graph-gt data/annotations/graph_level_gt.json
```

Writer SFT training (7B LoRA):

```bash
python scripts/train_mcoc_sft.py \
	--train-jsonl data/annotations/writer_train.jsonl \
	--eval-jsonl data/annotations/writer_eval.jsonl \
	--model-name Qwen/Qwen2.5-7B-Instruct \
	--batch-size 1 \
	--grad-accum 8 \
	--num-train-epochs 1.0 \
	--output-dir artifacts/train/mcoc_sft_7b
```

Writer SFT smoke run (small model, quick verification):

```bash
python scripts/train_mcoc_sft.py \
	--train-jsonl data/annotations/writer_train.jsonl \
	--eval-jsonl data/annotations/writer_eval.jsonl \
	--model-name Qwen/Qwen2.5-0.5B-Instruct \
	--batch-size 1 \
	--grad-accum 2 \
	--max-steps 10 \
	--max-seq-length 1024 \
	--lora-r 8 \
	--lora-alpha 16 \
	--output-dir artifacts/train/mcoc_sft_smoke
```

Writer evaluation:

```bash
python scripts/evaluate_mcoc_writer.py \
	--eval-jsonl data/annotations/writer_eval.jsonl \
	--base-model Qwen/Qwen2.5-7B-Instruct \
	--checkpoint-dir artifacts/train/mcoc_sft_7b/checkpoint-final \
	--output artifacts/eval/writer_sft_eval.json
```

Reasoner SFT scaffold:

```bash
python scripts/train_amr_sft.py
```

Reasoner RL scaffold:

```bash
python scripts/train_amr_rl.py
```

## Baselines and Ablations

Implemented baseline-ready pipeline modes (config/script level):

- Construction: `rule_text_only`, `llm_text_only`, `llm_text_plus_acoustic_tags`, `mcoc_full`.
- Reasoning: `transcript_rag`, `static_graph_retrieval`, `naive_graph_rag`, `amr_full`.

Use separate config files and `--experiment-name` tags to run ablations such as:

- construction: no observability typing, no merge/dedup, no state update.
- audio: transcript-only, no scene/events/speaker.
- reasoning: no query DAG, no rolling memory, no self-check.

## Current Implementation Status

Implemented now:

- end-to-end phase-1 runnable pipeline.
- deterministic artifact writing and caching by session.
- schema, memory graph writer, and AMR baseline.
- evaluation module interfaces and metric implementations for pilot use.
- training entrypoints as scaffolds.

Planned next:

- plug real local HF ASR/VAD/scene/event models as defaults.
- reasoner SFT then lightweight GRPO/DPO.
- stronger phase2 operation-level and graph-level writer benchmarking with full GT.
- reasoner SFT then lightweight GRPO/DPO.
