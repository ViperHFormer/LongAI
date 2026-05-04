# Prompt for Code Agent — Implement the LongAI Research Project End-to-End

You are implementing **LongAI**, an audio-first research system for building a self-evolving personal intent memory from long egocentric audio and reasoning over it.

This prompt is for the **engineering execution phase**.
Assume that repository scaffolding exists or that you are expected to create it.
Focus on concrete implementation details: data usage, model wrapping, training, evaluation, baselines, ablations, and pilot experiments.

---

# 1. Overall mission

Implement a first research-ready version of LongAI that can:

1. select a small real-world subset from local Ego4D-style data;
2. extract audio and build session manifests;
3. run audio perception tools;
4. construct a streaming personal intent memory graph;
5. answer memory-grounded queries and run prediction/planning tasks;
6. evaluate all stages with benchmark-style metrics;
7. support baseline and ablation experiments.

The first target is **small-scale end-to-end correctness**, not large-scale performance.

---

# 2. Environment assumptions

- Conda environment name: `longai`
- GPU resources: only a few A6000 GPUs are available
- Therefore prioritize:
  - modular model wrappers
  - small default local models
  - batchable offline preprocessing
  - caching
  - resume support

The implementation must support **two backend modes** for LLM-like modules:

1. **local HuggingFace deployment** (default for initial testing)
2. **API backend** (optional, for later stronger upper-bound experiments)

Design all wrappers so the backend can be switched from config.

---

# 3. Dataset implementation plan

## 3.1 Data source
The user has local Ego4D data already downloaded as MP4.
Use only a **small subset** initially.

## 3.2 Required dataset scripts

### Script A: subset selection
Create a script that:
- scans the local source directory for candidate videos;
- filters for videos with audio and reasonable duration;
- selects a small pilot subset;
- writes `data/raw/pilot_subset_manifest.jsonl`.

Initial pilot target:
- 20–30 sessions total, or the equivalent number of clips that can be merged into sessions;
- total extracted audio should stay manageable, ideally well below a few tens of GB.

### Script B: audio extraction
Create a script using ffmpeg to extract:
- mono wav
- 16 kHz or configurable rate
- per selected video/session

Write outputs under something like:
- `data/processed/audio/{session_id}.wav`

### Script C: session manifest builder
Build a canonical manifest file per session containing:
- session id
- source file path
- extracted wav path
- duration
- wearer id if available
- optional clip boundaries if source video is merged from multiple clips

## 3.3 Ground truth structure
You must support three GT layers.

### Layer A: tool-level GT
For pilot data, support placeholders or annotation import for:
- VAD regions
- speaker role
- ASR
- coarse scene
- optional coarse event tags

### Layer B: graph-level GT
Support human annotation JSON for:
- intents
- episodes
- entities
- relations
- states
- evidence references

### Layer C: reasoning GT
Support JSON/JSONL for:
- QA items
- next-intent labels
- planning references

## 3.4 Annotation format design
Create annotation templates and schemas.
Use machine-readable JSON.
Every graph annotation should include provenance.

Example graph GT object:
- `session_id`
- `nodes`
- `edges`
- `evidence`
- `timeline_updates`

---

# 4. Perception pipeline implementation

Implement a full preprocessing stage that outputs **EvidencePacks**.

## 4.1 VAD
Create a wrapper module for VAD.

Requirements:
- configurable backend
- default: strong open VAD model or existing Silero-like backend
- output timestamped speech segments
- cache results to disk

Expected output file:
- `artifacts/perception/vad/{session_id}.json`

## 4.2 Speaker role attribution
Do not implement full TSE.
Instead, implement or approximate:
- `wearer`
- `other`
- `mixed`
- `unknown`

This can be a heuristic or weakly supervised first version.
If robust wearer attribution is unavailable, provide a pluggable interface and a baseline heuristic.

Expected output file:
- `artifacts/perception/speaker_role/{session_id}.json`

## 4.3 ASR
Implement a wrapper that supports:
- local HuggingFace model
- optional API backend

For initial local testing, support small-to-medium models to ensure the pipeline runs.
Later configs can swap in larger models.

Expected output:
- per speech segment transcript
- optional timestamps
- confidence if available

Cache under:
- `artifacts/perception/asr/{session_id}.json`

## 4.4 ASC / audio context understanding
Implement a coarse scene classifier or audio caption wrapper.
This should run on context windows around speech segments, not only on speech content.

Expected output labels such as:
- office
- meeting_room
- home
- kitchen
- street_outdoor
- vehicle
- cafe_shop
- other

Cache under:
- `artifacts/perception/scene/{session_id}.json`

## 4.5 SED / event cue extraction
This is optional but should be implemented as a pluggable module.
A coarse event ontology is enough initially.
Examples:
- typing
- door
- footsteps
- traffic
- kitchen_activity
- appliance
- crowd_chatter
- silence_quiet

If strong fine-grained SED is too unstable, permit an audio-caption fallback that emits event tags.

Cache under:
- `artifacts/perception/events/{session_id}.json`

## 4.6 EvidencePack builder
Merge all tool outputs into timestamp-aligned evidence units.

Design principle:
- speech segment is the primary unit;
- scene/event windows are aligned back to that segment;
- store both direct speech evidence and surrounding context evidence.

Output:
- `artifacts/evidence_packs/{session_id}.jsonl`

Each line should represent a canonical EvidencePack.

---

# 5. Streaming memory construction implementation

Implement **MCoC** as a real module, even if first version is partly prompt- or rule-based.

## 5.1 Candidate spotting
Given an EvidencePack, generate candidates for:
- intents
- episodes
- entities
- time expressions

Implement two versions:
1. rule / heuristic baseline
2. LLM-backed candidate spotter

## 5.2 Observability typing
Classify each candidate as:
- `EXTRACTED`
- `INFERRED`
- `AMBIGUOUS`

This must be implemented explicitly, not just implicit in free-form text.

## 5.3 Graph write operation generator
Implement a structured generation target.
Use JSON or pydantic schema.

Required operation kinds:
- `CREATE_NODE`
- `UPDATE_NODE`
- `MERGE_NODE`
- `CREATE_EDGE`
- `UPDATE_STATE`
- `ATTACH_EVIDENCE`
- `ARCHIVE_NODE`

## 5.4 Memory manager
Implement a graph store that can:
- apply operations in order
- merge similar nodes
- track node history
- update intent states
- preserve evidence provenance
- serialize to disk

You may use `networkx` for the first version.
Design the interface so another backend could be substituted later.

## 5.5 Streaming processor
Create a pipeline that scans EvidencePacks chronologically and applies MCoC incrementally.

Output files:
- session-level graph snapshot
- full update log
- optional intermediate graph snapshots per step

Recommended paths:
- `artifacts/memory/session_graphs/{session_id}.json`
- `artifacts/memory/update_logs/{session_id}.jsonl`

---

# 6. Query-time reasoning implementation

Implement **Adaptive Memory Reasoning (AMR)**.

## 6.1 Query router
Classify queries into:
- status query
- temporal multi-hop query
- update/conflict query
- next-intent prediction
- short-horizon planning
- acoustic disambiguation query

## 6.2 Reasoning scaffold builder
Build a temporary DAG or dependency structure over subgoals.
This is inspired by LogicRAG-like dynamic reasoning structures, but must operate over the persistent memory graph plus evidence.

## 6.3 Retrieval engine
Support retrieval from:
- graph nodes
- graph neighborhoods
- evidence nodes
- transcript snippets
- optional raw audio segment references

## 6.4 Rolling reasoning memory
Maintain a compact summary over retrieved evidence to prevent exploding context length.

## 6.5 Self-check
Before final answer output, verify whether all required subgoals are resolved.
If not, perform one of:
- re-retrieve
- answer with uncertainty
- abstain

## 6.6 Output format
The reasoning result must include:
- `answer`
- `graph_trace`
- `evidence_trace`
- `confidence`
- `abstained` boolean

Save outputs under:
- `artifacts/reasoning/{experiment_name}/{session_id or query_set}.json`

---

# 7. Training implementation

Training must be staged and optional.
Do not make the whole project depend on training being complete.

## 7.1 SFT for construction writer
Implement a trainer that learns to produce graph-write operations from EvidencePacks + local memory context.

### Training example format
Input:
- evidence pack
- optional current local graph context

Output:
- structured graph-write operations

### Data source
Use human graph annotations converted into write-operation trajectories.

## 7.2 SFT for reasoning policy
Implement supervised training for AMR trajectories.

### Training target
A trajectory may include:
- query type
- reasoning scaffold
- retrieval requests
- final answer

## 7.3 Optional RL / preference optimization
Support an optional training stage for reasoning only.

Potential reward components:
- correctness
- evidence faithfulness
- efficiency (turn count / token count)
- correct abstention when evidence is insufficient

Important:
- keep this optional
- expose hooks for reward calculation and rollout logging
- do not let RL complexity block the rest of the project

---

# 8. Evaluation implementation

Implement stage-wise evaluation scripts and reusable evaluators.

## 8.1 Tool-level evaluation

### VAD
Metrics:
- frame precision / recall / F1
- segment F1
- IoU

### Speaker role attribution
Metrics:
- macro-F1
- confusion matrix

### ASR
Metrics:
- WER
- CER

### ASC
Metrics:
- accuracy
- macro-F1

### Event cues / SED
Metrics:
- multilabel F1 or event-F1

---

## 8.2 Graph evaluation

Metrics:
- node precision / recall / F1 by type
- edge precision / recall / F1 by relation
- intent-state accuracy
- update/conflict accuracy
- evidence attribution hit / IoU
- compactness ratio

Need support fuzzy label matching and canonical-id matching when available.

---

## 8.3 Reasoning evaluation

### QA
- exact / normalized match where appropriate
- LLM-as-judge hook for open-ended cases
- faithfulness / evidence correctness

### Next-intent prediction
- top-1 accuracy
- top-3 accuracy
- MRR

### Planning
- validity
- consistency with known constraints
- conflict avoidance
- optional LLM-as-judge rubric

### Efficiency
- number of retrieval turns
- latency
- prompt tokens if available

---

## 8.4 Oracle factorization

This is mandatory.
Implement evaluation modes where the system can replace predicted tool outputs with GT.

At minimum support:
- GT transcript + GT acoustic tags
- Pred transcript + GT acoustic tags
- GT transcript + Pred acoustic tags
- Pred transcript + Pred acoustic tags

This will isolate bottlenecks.

---

# 9. Baselines

Implement at least the following baselines.

## 9.1 Construction baselines
1. `rule_text_only`
2. `llm_text_only`
3. `llm_text_plus_acoustic_tags`
4. `mcoc_full`
5. `mcoc_oracle_tools`

## 9.2 Reasoning baselines
1. `transcript_rag`
2. `static_graph_retrieval`
3. `memory_agent_text_only`
4. `naive_graph_rag`
5. `amr_full`
6. `oracle_memory`

Each baseline should have a config and a runnable script.

---

# 10. Ablation experiments

Create experiment configs and scripts for these ablations.

## Construction ablations
- no observability typing
- no confidence-aware operations
- no merge / dedup
- no state updates
- no evidence links

## Audio ablations
- no speaker role
- no scene
- no event cues
- transcript only
- speech-only without surrounding context

## Reasoning ablations
- no query DAG
- no rolling memory
- no self-check
- no acoustic evidence recall
- answer after first retrieval only

---

# 11. Minimal pilot benchmark implementation

Create a small but complete pilot evaluation package.

## Required contents
- small selected sessions
- extracted audio
- session manifests
- annotation templates
- example filled annotations for a few sessions
- example QA / next-intent / planning tasks
- configs for running all baselines on this pilot set

## Goal
A reviewer or collaborator should be able to run:
1. preprocessing
2. memory construction
3. reasoning
4. evaluation

on a small pilot split.

---

# 12. Commands and scripts to provide

Provide CLI entrypoints roughly like:

- `python scripts/select_pilot_subset.py ...`
- `python scripts/extract_audio.py ...`
- `python scripts/build_session_manifests.py ...`
- `python scripts/run_perception.py ...`
- `python scripts/build_memory.py ...`
- `python scripts/run_reasoning.py ...`
- `python scripts/evaluate_tools.py ...`
- `python scripts/evaluate_memory.py ...`
- `python scripts/evaluate_reasoning.py ...`
- `python scripts/train_mcoc_sft.py ...`
- `python scripts/train_amr_sft.py ...`
- `python scripts/train_amr_rl.py ...` (optional)

Include example shell commands in the README.

---

# 13. Logging and artifact policy

Every stage must save artifacts in deterministic locations.
Use config hashes or experiment names.
Make reruns resumable.

Recommended top-level artifact structure:
- `artifacts/perception/`
- `artifacts/evidence_packs/`
- `artifacts/memory/`
- `artifacts/reasoning/`
- `artifacts/eval/`
- `artifacts/train/`

---

# 14. README requirements

The repository README must explain:
- research problem
- architecture overview
- project layout
- setup in conda env `longai`
- how to run pilot subset extraction
- how to run the full pipeline on pilot data
- how to train writer / reasoning models
- how to run baselines and ablations
- what is currently implemented vs placeholder

---

# 15. Important research-aware implementation notes

1. **Do not collapse the project into generic GraphRAG.**
   The main contribution is streaming intent memory construction plus adaptive reasoning.

2. **Do not assume all non-speech audio is useless.**
   Use it as context, not as the primary memory unit.

3. **Do not treat every detected proposition as a durable intent.**
   Observability typing and confidence are required to control graph quality.

4. **Do not force expensive RL early.**
   First ensure tool outputs, graph writes, and reasoning traces are all inspectable and evaluable.

5. **Make everything inspectable.**
   Save intermediate JSONs, operation logs, traces, and graph snapshots.

---

# 16. Final deliverables from the code agent

At the end of implementation, the repository should support:

1. a tiny real-audio pilot subset;
2. a working perception pipeline;
3. a working streaming memory builder;
4. a working reasoning module;
5. complete evaluation scripts;
6. baseline and ablation configs;
7. optional training scaffolds;
8. enough structure to support a publishable research project.

Your implementation should optimize for:
- clarity
- modularity
- reproducibility
- inspectability
- smooth future scaling

