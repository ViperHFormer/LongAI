# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

LongAI is an **audio-first** research codebase that builds a **streaming personal intent memory** from long egocentric audio and reasons over it. The core contribution is two methods:

- **MCoC** (Multimodal Chain-of-Construction): decomposes graph construction into structured, evidence-aware write operations with explicit observability typing (EXTRACTED/INFERRED/AMBIGUOUS).
- **AMR** (Adaptive Memory Reasoning): builds a temporary query-time reasoning DAG over the persistent memory graph, rather than using static graph traversal.

Three-stage pipeline: **Audio Perception → Streaming Memory Construction (MCoC) → Adaptive Memory Reasoning (AMR)**.

## Environment

Conda env: `longai`. Install: `pip install -r requirements.txt`. Requires `ffmpeg`/`ffprobe` in PATH.

## Common commands

```bash
# Full phase-1 pipeline (no training needed)
python scripts/run_phase1_pipeline.py --source-dir <ego4d_video_dir> --max-sessions 24

# Individual steps
python scripts/select_pilot_subset.py --source-dir <dir> --max-sessions 24
python scripts/extract_audio.py
python scripts/build_session_manifests.py
python scripts/init_annotation_templates.py
python scripts/run_perception.py --backend local_hf
python scripts/build_memory.py
python scripts/run_reasoning.py --experiment-name amr_full

# Evaluation (requires GT annotations)
python scripts/evaluate_tools.py --session-id session_0000
python scripts/evaluate_memory.py --session-id session_0000
python scripts/evaluate_reasoning.py --session-id session_0000

# Writer SFT training (Phase 2)
python scripts/prepare_writer_sft_data.py --mode update-logs --update-logs-dir artifacts/memory/update_logs --evidence-pack-dir artifacts/evidence_packs
python scripts/train_mcoc_sft.py --train-jsonl data/annotations/writer_train.jsonl --eval-jsonl data/annotations/writer_eval.jsonl

# Tests
pytest tests/ -v
```

## Architecture

```
longai/
├── schema/models.py      # Canonical pydantic models: EvidencePack, MemoryNode, MemoryEdge,
│                          #   GraphWriteOperation, ReasoningResult, enums (SpeakerRole,
│                          #   Observability, OperationKind)
├── data/                 # Subset selection, audio extraction (ffmpeg), session manifests
├── tools/                # Perception wrappers: VAD, ASR, speaker_role, scene, events
│   └── base.py           #   BaseTool + ToolConfig (backend: local_hf | api)
├── construction/         # MCoC: evidence_pack builder, candidate spotting, observability,
│   ├── mcoc.py           #   ops generation, orchestrator (generate_write_operations)
│   ├── candidates.py     #   Keyword-based candidate spotting (currently rule-based)
│   └── observability.py  #   Rule-based EXTRACTED/INFERRED/AMBIGUOUS classification
├── memory/               # Graph store (networkx MultiDiGraph) + streaming manager
│   ├── graph_store.py    #   MemoryGraphStore: apply_operation, snapshot, save
│   └── manager.py        #   build_memory_from_evidence: EvidencePacks → graph
├── reasoning/            # AMR: query router, scaffold builder, retrieval, answer
│   ├── router.py         #   Keyword-based query type classification
│   ├── scaffold.py       #   Builds subgoal DAG per query type
│   ├── retrieval.py      #   Simple token-match retrieval from graph nodes + evidence
│   └── amr.py            #   Top-level answer_query orchestrator
├── eval/                 # tool_metrics, memory_metrics, reasoning_metrics
├── train/                # writer_sft (Qwen LoRA via trl), reasoner_sft/rl scaffolds
└── utils/                # io (JSON/JSONL), config (YAML), hashing, logging
```

## Key design points

- **Speech-centric, not fixed-window**: VAD segments are the primary semantic unit. Non-speech context windows around speech provide scene/event augmentation only.
- **Two graphs are separate**: persistent memory graph (construction) vs. query-time reasoning DAG (AMR) — never conflate them.
- **All tools have stub/mock defaults**. VAD uses energy threshold; ASR returns hardcoded text. Each tool wrapper supports `backend: local_hf | api` and caches results to `artifacts/perception/<tool>/<session_id>.json`.
- **Graph store** uses `networkx.MultiDiGraph`. Nodes are `(node_id, {attrs})`. Edges are `(src, tgt, {relation, confidence})`. Evidence is stored in a separate dict keyed by segment_id.
- **State machine for intents**: `tentative → planned → ongoing → done | canceled | dropped`. Updated via `UPDATE_STATE` operations.
- **Schema is the source of truth**: all pipeline stages use the pydantic models from `longai.schema.models`. Changes to the schema propagate everywhere.
- **Artifact layout**: `artifacts/perception/`, `artifacts/evidence_packs/`, `artifacts/memory/`, `artifacts/reasoning/`, `artifacts/eval/`, `artifacts/train/`. Every stage writes deterministic, cacheable outputs keyed by session_id.
- **Oracle factorization** is mandatory for evaluation: support GT/pred tool output cross-combinations to isolate bottlenecks.
- **Training is staged and optional**: Phase 1 runs without training. Phase 2 SFTs the writer. Phase 3 adds reasoner SFT + optional RL. RL must never block the rest of the pipeline.

## Current implementation status

- Phase 1 pipeline: **runnable end-to-end** with mock tools and rule-based MCoC/AMR.
- Real ASR/VAD/scene models: **not yet plugged in** — tool wrappers exist but return stub outputs.
- Writer SFT: scaffolded with full LoRA training via TRL, but depends on quality GT data.
- Reasoner SFT/RL: stub entrypoints only.
- Tests: minimal (schema roundtrip, graph store create, router classification).
