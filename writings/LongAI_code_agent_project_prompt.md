# Prompt for Code Agent — Understand and Scaffold the LongAI Project

You are a senior research engineer helping build **LongAI**, an audio-first long-term memory agent for egocentric audio.

Your task in this phase is **NOT** to optimize every model or finish every experiment immediately.
Your task is to understand the research design, create a clean project structure, and implement a robust, minimal but extensible first version.

---

## 1. Project goal

Build a research codebase that studies the following problem:

> Given a long egocentric audio stream, construct a **self-evolving personal intent memory** and use it for downstream reasoning such as current intent status query, next-intent prediction, conflict/update reasoning, and short-horizon planning.

This project is inspired by the separation of **memorization** and **control** in M3-Agent, but differs in four important ways:

1. the primary modality is **audio**, not video;
2. the core memory target is **personal intent**, not generic world knowledge;
3. memory construction must be **streaming and evidence-aware**;
4. reasoning should use a **query-time reasoning scaffold** rather than only static graph traversal.

---

## 2. Non-negotiable design principles

### 2.1 Audio-first, not video-first
The project may use data that originally comes from video files, but LongAI itself should operate on **audio only**.
The pipeline should extract and use waveform audio as the core input.

### 2.2 Speech-driven, not fixed-window-only
Do **not** make fixed 30-second clips the primary semantic unit.
Use a **speech-centric processing design**:

- detect speech segments with VAD;
- optionally segment by speaker turn;
- use non-speech windows around speech as supporting context for scene/event estimation.

### 2.3 Persistent self-evolving memory
The graph is not rebuilt from scratch for every query.
It is incrementally updated while scanning audio from left to right.

### 2.4 Compact graph schema
Use a compact schema.
Avoid overly broad ontologies.

Required node types:
- `Intent`
- `Episode`
- `Entity`
- `Evidence`

Required edge types:
- `subgoal_of`
- `realized_by`
- `involves`
- `located_at`
- `before`
- `updates`
- `grounded_by`

### 2.5 Confidence and observability are first-class
Each node/edge/write operation should support:
- `confidence`
- `observability`

Use three observability states:
- `EXTRACTED`
- `INFERRED`
- `AMBIGUOUS`

### 2.6 Two reasoning layers
The project must distinguish between:
1. **persistent memory graph** built during memorization;
2. **temporary query-time reasoning scaffold** built during control.

---

## 3. High-level research architecture

Implement the project around three main stages.

### Stage A. Audio Perception
Input: raw audio sessions.
Output: timestamped multimodal evidence packs.

Suggested components:
- VAD
- speaker-role attribution (`wearer`, `other`, `mixed`, `unknown`)
- ASR
- coarse scene understanding (ASC or audio caption)
- optional coarse event understanding (SED or event caption)

### Stage B. Streaming Memory Construction
Input: evidence packs in temporal order.
Output: persistent Personal Intent Memory graph.

Use **MCoC — Multimodal Chain-of-Construction**:
- candidate spotting
- observability typing
- graph-write operation generation
- confidence scoring
- merge / dedup / state update

### Stage C. Adaptive Memory Reasoning
Input: query + current graph.
Output: answer / prediction / plan.

Use **AMR — Adaptive Memory Reasoning**:
- query routing
- query-time subgoal DAG
- topological execution
- selective retrieval of graph nodes + evidence
- rolling reasoning memory
- self-check before final answer

---

## 4. Initial scope constraints

This repository must be designed for **limited GPU resources** (a few A6000s).
Therefore:

- prefer modular, swappable model wrappers;
- allow both local HuggingFace deployment and API-based backends;
- use small or medium open models by default for testing;
- keep all training pipelines optional and incremental.

The first milestone is a **working research prototype**, not a fully optimized large-scale system.

---

## 5. Data assumptions

The user already has a local Ego4D download.
It is very large and stored as MP4.
For LongAI, you only need a **small pilot subset** for testing.

The codebase should support:

1. selecting a small subset of source videos;
2. extracting audio only from them;
3. creating manageable sessions for pilot experiments;
4. loading human annotations or placeholder annotation templates for graph GT and QA GT.

The benchmark design should be **real-data first**.
Synthetic data may exist only as an auxiliary debugging option.

---

## 6. What the codebase must contain

Create a clean research repository with the following logical areas.
You do not have to use these exact names, but the structure should be close.

### 6.1 `configs/`
- dataset configs
- model configs
- training configs
- evaluation configs
- experiment presets

### 6.2 `data/`
- raw metadata manifests
- processed manifests
- annotation templates
- benchmark splits

### 6.3 `longai/`
Main python package.

Suggested subpackages:

#### `longai/data/`
- dataset selection
- audio extraction
- session building
- manifest generation
- data loaders

#### `longai/tools/`
Wrappers for perception tools:
- vad
- speaker_attribution
- asr
- asc
- sed_or_audio_caption

Each wrapper must support:
- local backend
- optional API backend
- cached outputs

#### `longai/schema/`
- graph schema definitions
- pydantic/dataclass models for nodes, edges, evidence, operations

#### `longai/memory/`
- graph store
- node merge
- temporal update logic
- confidence aggregation
- self-evolving memory manager

#### `longai/construction/`
- evidence pack builder
- candidate spotting
- observability typing
- graph write operation generation
- MCoC orchestrator

#### `longai/reasoning/`
- query router
- reasoning scaffold / DAG builder
- retrieval engine
- rolling memory summarizer
- answer / prediction / planning module

#### `longai/eval/`
- VAD metrics
- ASR metrics
- ASC metrics
- SED metrics
- graph metrics
- reasoning metrics
- oracle factorization

#### `longai/train/`
- SFT for construction writer
- SFT for reasoning policy
- optional RL / DPO / preference optimization scaffolds

#### `longai/utils/`
- logging
- I/O
- seed control
- caching
- retry logic

### 6.4 `scripts/`
Executable scripts for:
- subset creation
- audio extraction
- preprocessing
- graph construction
- reasoning demo
- evaluation
- training

### 6.5 `notebooks/`
Only lightweight exploratory notebooks.
The core pipeline must live in python modules and scripts.

### 6.6 `tests/`
Unit tests for:
- schema validity
- graph merge behavior
- parsing and I/O
- reasoning scaffold construction

---

## 7. Schema requirements

Implement these canonical classes.

### 7.1 EvidencePack
Represents one timestamped input unit for memory construction.
Fields should include:
- `segment_id`
- `session_id`
- `start_time`
- `end_time`
- `waveform_path` or audio reference
- `speaker_role`
- `asr_text`
- `scene_label`
- `event_tags`
- `affective_tags`
- `tool_confidences`
- `context_window_refs`

### 7.2 MemoryNode
Required fields:
- `node_id`
- `node_type`
- `label`
- `attributes`
- `confidence`
- `created_at`
- `last_updated_at`

### 7.3 MemoryEdge
Required fields:
- `edge_id`
- `source`
- `target`
- `relation`
- `confidence`
- `attributes`

### 7.4 EvidenceRef
Required fields:
- `evidence_id`
- `modality`
- `source_tool`
- `start_time`
- `end_time`
- `payload`
- `observability`
- `confidence`

### 7.5 GraphWriteOperation
Required operation kinds:
- `CREATE_NODE`
- `UPDATE_NODE`
- `MERGE_NODE`
- `CREATE_EDGE`
- `UPDATE_STATE`
- `ATTACH_EVIDENCE`
- `ARCHIVE_NODE`

---

## 8. Streaming memory behavior

The memory manager must support left-to-right streaming updates.

Pseudo requirement:

- read evidence packs chronologically
- emit graph-write operations for each pack
- apply operations to persistent graph
- store provenance and evidence links
- keep active memory and archived memory separated

Intent states should support at least:
- `tentative`
- `planned`
- `ongoing`
- `done`
- `canceled`
- `dropped`

People entities must distinguish:
- `USER`
- `OTHER_PERSON_n`

---

## 9. Reasoning requirements

Implement a reasoning module that is not just naive GraphRAG.

### Required query types
- current status query
- what is coming up next
- what changed / was canceled
- what should the user likely do next
- short-horizon schedule suggestion

### Required reasoning flow
1. route query type
2. build a small query DAG / scaffold
3. retrieve relevant nodes and evidence in dependency order
4. keep a rolling memory summary
5. self-check completeness
6. output answer with evidence references

The result format should include:
- final answer
- graph trace
- evidence trace
- confidence

---

## 10. Learning philosophy

Do not over-engineer the first version.

### Phase 1
No training required.
Just run frozen tools and a prompted or heuristic construction module.

### Phase 2
SFT the construction writer on annotated graph-write trajectories.

### Phase 3
SFT the reasoning policy on solved query trajectories.

### Phase 4
Optional RL or preference optimization for reasoning efficiency and faithfulness.

Important: if implementing RL scaffolding, keep it modular and optional.
The project should still run without RL.

---

## 11. Benchmark philosophy

Support three evaluation layers.

### A. Tool-layer evaluation
- VAD
- speaker attribution
- ASR
- ASC
- SED/event cues

### B. Memory-layer evaluation
- node/edge F1
- state accuracy
- update/conflict accuracy
- evidence attribution accuracy
- compactness ratio

### C. Reasoning-layer evaluation
- QA correctness
- faithfulness
- next-intent prediction accuracy
- planning validity
- token/latency efficiency

Also support **oracle factorization** by swapping in GT tool outputs.

---

## 12. Engineering expectations

The repository should be:
- clean and reproducible
- configuration-driven
- able to cache intermediate outputs
- able to resume from any stage
- able to run a tiny pilot experiment end to end

Please create:
- clear README
- environment setup notes for conda env `longai`
- example configs
- example small-run commands
- lightweight test fixtures

---

## 13. What to avoid

Do **not**:
- hard-code everything into one giant script
- overcomplicate the ontology
- make visual processing a hidden dependency
- mix persistent graph and query-time reasoning graph into one thing
- force every component to require huge GPUs
- make RL mandatory in the initial version

---

## 14. Deliverables for this phase

Please generate a repository scaffold that includes:

1. all major modules and placeholders;
2. clean data model definitions;
3. working command-line entrypoints;
4. clear TODO markers where training or annotation is still needed;
5. a tiny end-to-end demo path over mock or pilot data.

The result should look like a serious research codebase that can grow into a paper-ready project.

