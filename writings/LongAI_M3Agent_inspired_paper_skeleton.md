# LongAI: Streaming Personal Long-Audio Intent Memory from Long Egocentric Audio

## Working title candidates

1. **LongAI: Streaming Personal Long-Audio Intent Memory from Long Egocentric Audio**
2. **Listen, Remember, Intend, and Reason: Audio-Grounded Long-Term Memory for Personal Intent Modeling**
3. **From Continuous Egocentric Audio to Self-Evolving Intent Memory**
4. **Streaming Intent Memory for Wearable Audio Agents**

---

## 0. Core positioning

This paper should **not** be framed as “a large pipeline with many modules”.
It should be framed as a **research problem + benchmark + two core methods**:

- **Problem**: long-form egocentric audio contains sparse, distributed, time-dependent evidence about a user’s plans, obligations, preferences, and future actions.
- **Goal**: build a **self-evolving personal intent memory** from streaming audio and use it for downstream reasoning.
- **Method novelty 1**: **MCoC — Multimodal Chain-of-Construction** for streaming, evidence-aware intent graph writing.
- **Method novelty 2**: **AMR — Adaptive Memory Reasoning** for query-time reasoning over persistent intent memory with graph- and evidence-level retrieval.

---

# 1. Introduction

## 1.1 Problem setting

Always-on wearable assistants naturally receive **continuous egocentric audio** rather than neatly segmented task descriptions. In realistic long audio streams, information related to a user’s future plans and intents is:

- **sparse**: only a small fraction of time contains intent-relevant evidence;
- **distributed**: the evidence for a single intent may be spread across multiple distant moments;
- **heterogeneous**: transcript, speaker role, scene, acoustic cues, and non-speech events contribute complementary signals;
- **dynamic**: intents are created, refined, delayed, canceled, or fulfilled over time.

Existing long-context or memory-agent work mostly studies video-centric long-term memory, textual memory, or generic QA. It does not directly solve the problem of **streaming intent memory construction from long egocentric audio**.

## 1.2 Why transcript-only is insufficient

A transcript-only system is an essential baseline, but it is not sufficient for the target problem because:

- **speaker role matters**: whether the wearer said something, overheard it, or was being addressed changes whether it should update personal intent memory;
- **acoustic evidence matters**: urgency, hesitation, emphasis, interruption, and background context often help disambiguate intent strength and status;
- **non-speech context matters**: environmental audio can provide location and activity cues that are not explicitly verbalized.

Therefore, LongAI should be positioned as an **audio-first, transcript-centered, evidence-aware memory agent**.

## 1.3 Key gap in current benchmarks

There is currently no standard benchmark that jointly evaluates:

1. long egocentric audio perception,
2. streaming personal intent memory construction,
3. evidence-grounded multi-hop reasoning over that memory.

Existing first-person datasets can provide real audio, but they do not directly provide **personal intent graph** ground truth. This paper therefore introduces a benchmark layer on top of real egocentric audio.

## 1.4 Main idea of the paper

We propose **LongAI**, an audio-first memory agent that transforms long egocentric audio into a **self-evolving personal intent memory**.

LongAI has three stages:

1. **Audio Perception**: detect speech turns, assign speaker role, transcribe speech, and derive acoustic context from surrounding windows.
2. **Streaming Memory Construction**: use **MCoC** to convert incremental multimodal evidence into graph-write operations over a compact personal intent memory.
3. **Adaptive Memory Reasoning**: use **AMR** to build a query-time reasoning scaffold over the persistent memory and selectively retrieve graph nodes, transcript evidence, and acoustic evidence.

## 1.5 Claimed contributions

Keep this section concise and sharp. Recommended version:

1. We formulate **streaming personal intent memory construction from long egocentric audio** as a new task.
2. We introduce a compact, evidence-aware **Personal Intent Memory** representation and **MCoC**, a structured chain-of-construction for incremental graph writing.
3. We propose **Adaptive Memory Reasoning (AMR)**, a query-time reasoning framework that performs dependency-aware retrieval over graph memory and supporting evidence.
4. We build **LongAI-Bench**, a real-audio benchmark derived primarily from egocentric datasets (e.g., Ego4D subsets) with additional human annotation for intent memory and downstream QA/prediction tasks.
5. We show that audio-grounded memory construction and adaptive reasoning outperform transcript-only and static-memory baselines, especially on state updates, conflict resolution, and future-intent prediction.

---

# 2. Related Work

## 2.1 Long-term memory for agents
- LLM memory agents
- multimodal memory agents
- distinction from generic dialogue memory and execution trajectory memory
- how M3-Agent motivates the “memorization/control” decomposition

## 2.2 Long audio and spoken language understanding
- VAD, diarization/speaker attribution, ASR
- acoustic scene and event understanding
- why long audio is different from short utterance SLU

## 2.3 Knowledge graph / memory graph construction
- text KG extraction
- streaming graph construction
- confidence-aware extraction and uncertainty labeling
- distinction between generic KG and **personal intent memory**

## 2.4 Retrieval-augmented reasoning
- vanilla RAG
- GraphRAG
- reasoning-guided retrieval
- M3-Agent-style multi-turn control
- LogicRAG-style query-time reasoning structures

## 2.5 Positioning statement

A very important paragraph:

- M3-Agent contributes the **memorization/control split**, entity-centric persistent multimodal memory, and learning-based control.
- LogicRAG contributes the insight that **query-time reasoning structures** can be more appropriate than relying only on a fixed pre-built graph.
- LongAI combines these ideas but targets a different problem: **audio-first, streaming, personal intent memory**, where memory construction itself must reason about observability, speaker role, and evolving status.

---

# 3. Task Definition

## 3.1 Input
A continuous egocentric audio stream for a single wearer.

## 3.2 Output
A persistent, timestamped, self-evolving **Personal Intent Memory** consisting of:

- intent nodes,
- episode nodes,
- entity nodes,
- evidence nodes,
- typed edges and confidence/observability labels,
- incremental state updates over time.

## 3.3 Downstream tasks

1. **Intent memory QA**
2. **Current status / pending-intent query**
3. **Conflict/update reasoning**
4. **Next-intent prediction**
5. **Short-horizon planning / schedule suggestion**

## 3.4 Evaluation regimes

- **offline session-level** evaluation
- **streaming prefix-to-future** evaluation
- **oracle-factorized** evaluation (GT vs predicted tools)

---

# 4. Method

# 4.1 Overall system

LongAI runs in two coupled loops:

- **Memorization loop** (online / left-to-right): update personal intent memory as audio arrives.
- **Reasoning loop** (on-demand): answer queries or predict next intents using the current memory state.

This mirrors M3-Agent’s memorization/control split, but adapted to audio-only intent modeling.

---

## 4.2 Audio perception as tool-mediated evidence extraction

### Design principle
Do **not** use fixed 30s clips as the primary semantic unit.
Instead, use **dual timescales**:

- **speech-centric units** for intent-bearing content;
- **context windows** around speech for scene and environment evidence.

### 4.2.1 Speech timeline
Use VAD and turn segmentation to produce speech segments.
Each speech segment has:

- start/end time
- speaker role (`wearer`, `other`, `mixed`, `unknown`)
- ASR transcript

### 4.2.2 Acoustic context timeline
Construct windows centered on or surrounding speech segments, e.g.:

- left context: 5–10s before speech
- right context: 5–10s after speech
- optional global window: 15–20s

From these windows derive:

- scene label (coarse ASC)
- event tags (coarse SED / audio caption)
- optional affective cues (urgency, calmness, hesitation, stress)

### 4.2.3 Why this is better than M3-Agent’s clip-by-clip scheme

For long egocentric audio, intent-relevant information is much denser in speech than in arbitrary fixed windows.
However, non-speech context is still useful as **supporting evidence** rather than primary memory content.
So the right design is not purely clip-by-clip, but **speech-driven with context-aware acoustic augmentation**.

This is a central design choice and should be explicitly defended in the paper.

---

## 4.3 Personal Intent Memory schema

Keep the schema compact.

### Node types

1. **Intent**
   - future-directed or preference-related personal states
   - e.g. attend meeting, buy coffee, call landlord, prepare slides
2. **Episode**
   - localized observed interaction/event/action
   - e.g. wearer agrees to a meeting, hears reminder, discusses schedule
3. **Entity**
   - user, other people, places, objects, organizations, topics
4. **Evidence**
   - transcript spans, acoustic tags, timestamps, surrounding window summaries

### Edge types

1. `subgoal_of` — Intent -> Intent
2. `realized_by` — Intent -> Episode
3. `involves` — Intent/Episode -> Entity
4. `located_at` — Intent/Episode -> Entity(place)
5. `before` — Intent/Episode -> Intent/Episode
6. `updates` — Intent/Episode -> Intent/Episode
7. `grounded_by` — any node -> Evidence

### Attributes

#### Intent attributes
- `owner` (`user` or other agent)
- `state` in `{tentative, planned, ongoing, done, canceled, dropped}`
- `time_text`
- `start_time_est`
- `end_time_est`
- `recurrence`
- `priority`

#### Evidence attributes
- `modality` in `{text, audio, text_audio}`
- `source_tool` (`vad`, `speaker_attrib`, `asr`, `asc`, `sed`, `audio_caption`, `llm`)
- `observability` in `{extracted, inferred, ambiguous}`
- `confidence` in `[0,1]`

### Important entity distinction
People must distinguish:
- `USER`
- `OTHER_PERSON_i`
- optional canonical names if available

This directly encodes the ego-centric nature of the task.

---

## 4.4 MCoC: Multimodal Chain-of-Construction

This is the main novelty.

### High-level idea
Instead of directly asking an LLM to “output a graph”, MCoC decomposes graph construction into a sequence of constrained write operations.

### Step 1: Candidate spotting
Input:
- transcript of a speech segment
- speaker role
- local acoustic context summary
- optional prior local memory neighborhood

Output:
- candidate intents
- candidate episodes
- candidate entities
- candidate time expressions

### Step 2: Observability typing
Every candidate item is assigned one of:
- **EXTRACTED**: directly grounded in speech/audio evidence
- **INFERRED**: plausible higher-level conclusion
- **AMBIGUOUS**: uncertain, keep but downweight or mark for review

This should be explicitly inspired by confidence-aware graph construction ideas, but tailored to audio intent memory.

### Step 3: Graph-write operations
The model emits a sequence of structured operations, e.g.:

- `CREATE_INTENT`
- `CREATE_EPISODE`
- `CREATE_ENTITY`
- `LINK_REALIZED_BY`
- `LINK_INVOLVES`
- `LINK_LOCATED_AT`
- `LINK_SUBGOAL`
- `LINK_BEFORE`
- `LINK_UPDATES`
- `ATTACH_EVIDENCE`
- `MERGE_NODE`
- `UPDATE_INTENT_STATE`

### Step 4: Confidence calibration
Each operation gets:
- confidence score
- observability label
- evidence references

Confidence is a function of:
- transcript explicitness
- speaker-role confidence
- acoustic support
- temporal coherence with memory
- contradiction with prior graph state

### Step 5: Streaming merge
Operations are applied to the persistent graph.
Existing nodes are updated rather than duplicated.
A weighted memory policy similar in spirit to M3-Agent’s weight mechanism can be adapted here, but now at the level of intent state and evidence support.

### Step 6: State transition and conflict handling
Intent states evolve over time based on new evidence.

Example:
- “I’ll meet Sam tomorrow” -> `planned`
- “I’m on my way to meet Sam” -> `ongoing`
- “The meeting is canceled” -> `updates` + `canceled`

### Why MCoC is meaningful
The benefit is not merely interpretability.
It addresses three real problems:

1. **over-extraction** from noisy audio,
2. **latent-vs-observable confusion**,
3. **streaming consistency** across long time spans.

---

## 4.5 Self-evolving intent memory

This section should formalize the streaming nature.

### Definition
At time t, the memory is `G_t`.
Each new segment produces an update operator `Δ_t`, and:

`G_t = Apply(G_{t-1}, Δ_t)`

### Design requirements
- online
- append-and-update, not rebuild-from-scratch
- bounded memory growth through merge and archival
- provenance-preserving

### Archival policy
Old episodes can be archived, but intents and high-value entities remain active.
This supports lifelong use without memory explosion.

---

## 4.6 Adaptive Memory Reasoning (AMR)

Name it something like:

- **Adaptive Memory Reasoning (AMR)**
- **Memory-Grounded Adaptive Reasoning**
- **Reasoning over Self-Evolving Intent Memory**

### Key idea
Use the persistent intent memory as a substrate, but build a **query-time reasoning scaffold** rather than relying only on static graph traversal.

This is where LogicRAG’s influence enters.

### 4.6.1 Query routing
Given a question, first classify it into:
- factual status query
- temporal / multi-hop query
- conflict/update query
- next-intent prediction
- short-horizon planning
- acoustic disambiguation query

### 4.6.2 Reasoning scaffold construction
Build a small query-time DAG over subgoals such as:
- identify relevant user intent
- resolve time
- resolve current state
- inspect conflicting updates
- gather support evidence

This DAG is **not** the persistent memory graph.
It is a temporary reasoning structure used to orchestrate retrieval.

### 4.6.3 Retrieval order
Use topological ordering over the query DAG.
At each step retrieve from:
- memory nodes
- local graph neighborhoods
- evidence nodes
- optionally raw transcript/audio snippets

### 4.6.4 Rolling evidence memory
Keep a compact rolling summary of already retrieved evidence, inspired by LogicRAG’s rolling memory.
This prevents context blow-up.

### 4.6.5 Self-check and abstention
Before final answer generation:
- verify that required subgoals are resolved
- if missing evidence remains, either re-retrieve or abstain / output uncertainty

### Why AMR matters
This prevents the system from being either:
- a naive graph traversal engine, or
- an unconstrained multi-turn RAG agent.

It gives you a principled middle ground.

---

## 4.7 Learning

This section must be realistic given limited A6000 GPUs.

### 4.7.1 Stage A: tool modules
Mostly frozen or lightly adapted.

- VAD: off-the-shelf
- speaker attribution: off-the-shelf or weakly supervised
- ASR: off-the-shelf / optionally LoRA
- ASC/SED/audio caption: frozen or few-shot prompted

### 4.7.2 Stage B: MCoC writer learning
Use **SFT first**, not RL-first.

#### Training target
Supervise the model to output:
- candidate extraction
- observability labels
- graph-write operations
- state updates
- evidence links

#### Why SFT first
Construction is structurally constrained and supervision-friendly.
RL at this stage is expensive and unstable.

### 4.7.3 Stage C: AMR policy learning
This is the best place to use RL.

#### Policy outputs
At each reasoning turn:
- choose retrieval target
- issue structured subquery
- decide whether more evidence is needed
- answer or abstain

#### Reward design
Potential reward components:
- answer correctness
- evidence faithfulness
- retrieval efficiency
- contradiction avoidance
- proper abstention when evidence is insufficient

#### Practical recommendation
Start with:
- SFT on expert trajectories / heuristic traces
- then lightweight RL or DPO-style refinement for the reasoning policy only

This is much more feasible than jointly RL-training memorization and reasoning from scratch.

---

# 5. LongAI-Bench

## 5.1 Data source philosophy
Real-world first.

### Main data source
- Ego4D-derived egocentric audio subset

### Optional auxiliary data
- a small synthetic audio set only for debugging and stress testing
- optionally public long-video memory benchmarks for pretraining the reasoning policy, but not as the core benchmark

## 5.2 Recommended benchmark construction

### Stage 1: pilot benchmark
- 20–30 sessions total
- each session 15–30 minutes
- total audio within a few tens of GB or less

### Stage 2: extended benchmark
- 50–100 sessions
- broader people / locations / activities

## 5.3 Ground truth layers

### Layer A: tool-level GT
- VAD boundaries
- speaker role / attribution
- ASR transcript
- coarse scene labels
- optional coarse event labels

### Layer B: memory GT
- intent nodes
- episode nodes
- entities
- relations
- state transitions
- confidence / observability labels
- supporting evidence references

### Layer C: reasoning GT
- QA pairs
- next-intent labels
- short-horizon planning references

## 5.4 Annotation philosophy
Only annotate **observable or well-supported** intents.
Do not annotate arbitrary latent internal states that cannot be recovered from audio.

This is essential for a fair benchmark.

---

# 6. Experiments

## 6.1 Research questions

RQ1. Does audio-grounded memory construction outperform transcript-only memory construction?

RQ2. Does MCoC reduce graph over-extraction while improving state and update accuracy?

RQ3. Does AMR outperform static retrieval and naive graph reasoning for downstream intent queries?

RQ4. Which acoustic signals actually matter: scene, events, speaker role, affect, or only transcript?

RQ5. How much of final performance is bottlenecked by tools versus memory/reasoning policy?

---

## 6.2 Metrics

### Tool-level
- VAD: frame F1, segment F1, IoU
- speaker attribution: macro-F1
- ASR: WER, CER
- ASC: accuracy, macro-F1
- SED / acoustic events: event-F1 or multilabel F1

### Memory construction
- node precision / recall / F1 by type
- edge precision / recall / F1 by relation
- intent-state accuracy
- update/conflict detection accuracy
- evidence attribution accuracy
- compactness ratio (predicted / GT)

### Reasoning
- answer correctness
- faithfulness
- evidence correctness
- next-intent top-k accuracy
- planning validity / conflict avoidance
- efficiency: turns, latency, token cost

---

## 6.3 Baselines

### Construction baselines
1. rule-based transcript extraction
2. transcript-only LLM construction
3. transcript + scene/event tags construction
4. full MCoC
5. oracle tools + MCoC upper bound

### Reasoning baselines
1. transcript-only RAG
2. static graph retrieval
3. M3-Agent-style multi-turn retrieval over textual memory
4. naive GraphRAG over intent graph
5. full AMR
6. oracle memory upper bound

---

## 6.4 Ablation study

### MCoC ablations
- no observability typing
- no confidence-aware write policy
- no state update module
- no merge / dedup
- no evidence links

### AMR ablations
- no query DAG
- no rolling evidence memory
- no self-check
- no acoustic evidence recall
- direct answer after first retrieval

### Tool ablations
- no speaker role
- no scene
- no event cues
- no affective cues
- transcript-only

### Oracle factorization
- GT transcript + GT acoustic tags
- Pred transcript + GT acoustic tags
- GT transcript + Pred acoustic tags
- Pred transcript + Pred acoustic tags

This is mandatory.

---

## 6.5 Qualitative analysis

Include 3–5 case studies:

1. transcript suffices
2. transcript ambiguous, speaker role resolves it
3. transcript ambiguous, acoustic context resolves it
4. conflict/update case
5. next-intent prediction case

---

# 7. Discussion

## 7.1 What M3-Agent teaches us

Useful lessons to acknowledge:
- separate memorization and control
- use persistent multimodal entity grounding
- semantic memory is crucial, not optional
- reasoning policy benefits from learning, not only prompting

## 7.2 What LongAI must do differently

- speech-driven, not naive fixed clips
- intent memory, not generic world memory
- observability-aware construction
- streaming state evolution
- audio-specific disambiguation
- persistent graph plus query-time reasoning scaffold

## 7.3 Practical constraints

With limited A6000 GPUs, the project should prioritize:
- compact schema
- frozen or lightly adapted tool models
- SFT for writer
- RL only for reasoning policy after the pipeline is stable

---

# 8. Limitations

- difficult to infer truly latent user intent from audio alone
- annotation cost of intent memory is high
- scene/event tools may still be noisy
- future planning evaluation can be partly subjective
- privacy-sensitive nature of egocentric audio

---

# 9. Conclusion

Re-emphasize that the contribution is not “many modules”, but a principled solution to:

**streaming personal intent memory construction and reasoning from long egocentric audio**.

---

# Strong paper narrative in one sentence

> We turn continuous egocentric audio into a self-evolving personal intent memory through structured multimodal graph writing, and we reason over that memory with adaptive query-time reasoning structures.

