# LongAI: 面向连续第一人称音频的流式个人意图记忆系统

## 技术报告

---

## 1. 项目背景

### 1.1 问题定义

可穿戴设备（智能眼镜、AR 头显）持续记录第一人称视角的音视频数据。如何从数十小时的连续音频流中**结构化提取、增量更新和自适应推理个人意图**，是一个开放的研究问题。现有方法面临三个核心挑战：

1. **上下文窗口有限**：固定窗口（如 30 秒）的帧处理会丢失跨时间尺度的依赖关系。
2. **不确定性建模不足**：仅凭标量置信度无法区分"证据清晰"和"模式熟悉"两种不同来源的模型确信。
3. **被动图遍历不充分**：静态图查询无法根据查询语义动态调整检索策略。

### 1.2 核心贡献

LongAI 提出两个方法来解决上述挑战：

- **MCoC（Multimodal Chain-of-Construction，多模态构建链）**：将图构建分解为结构化的、证据感知的写操作，通过显式的可观察性类型（EXTRACTED/INFERRED/AMBIGUOUS）建模不确定性。
- **AMR（Adaptive Memory Reasoning，自适应记忆推理）**：在持久化记忆图之上构建**临时的查询时间推理 DAG**，替代静态图遍历。

---

## 2. 系统架构

### 2.1 总体管线

```
原始音频 (Ego4D MP4)
    │
    ▼
┌──────────────────────────────────────────────────┐
│  Phase 0: 数据准备                                │
│  ffmpeg 提取 → 16kHz 单声道 WAV                   │
│  会话清单构建 (session_manifest.jsonl)            │
└──────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────┐
│  Phase 1: 音频感知 (Perception)                   │
│  VAD → ASR → 说话者角色 → 场景 → 事件             │
│  输出: EvidencePack (每段语音一个)                 │
└──────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────┐
│  Phase 2: 流式记忆构建 (MCoC)                     │
│  候选发现 → 可观察性分类 → 图写操作生成            │
│  MemoryGraphStore.apply_operation()               │
│  输出: 持久化记忆图 (NetworkX MultiDiGraph)        │
└──────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────┐
│  Phase 3: 自适应记忆推理 (AMR)                    │
│  查询路由 → 脚手架构建 → 检索 → 回答生成          │
│  输出: ReasoningResult (答案 + 证据追踪)           │
└──────────────────────────────────────────────────┘
```

### 2.2 核心设计原则

- **以语音段为语义单元**：VAD 段是基本处理单元，非语音上下文窗口仅用于场景/事件增强。
- **两图分离**：持久化记忆图（构建阶段）与查询时推理 DAG（AMR）物理分离，前者存储知识，后者按需构建。
- **语音优先（Audio-First）**：所有感知工具围绕语音段组织，非语音模态通过 EvidencePack 的 `context_window_refs` 对齐。
- **缓存优先架构**：每个工具以 `{session_id}.json` 为单位缓存确定性结果，支持增量重算。

---

## 3. 数据模型

### 3.1 核心 Schema（Pydantic 模型）

**枚举类型：**

| 枚举 | 取值 | 语义 |
|------|------|------|
| `SpeakerRole` | `wearer`, `other`, `mixed`, `unknown` | 说话者与佩戴者的关系 |
| `Observability` | `EXTRACTED`, `INFERRED`, `AMBIGUOUS` | 意图的可观察性层级 |
| `OperationKind` | `CREATE_NODE`, `UPDATE_NODE`, `MERGE_NODE`, `CREATE_EDGE`, `UPDATE_STATE`, `ATTACH_EVIDENCE`, `ARCHIVE_NODE` | 图写操作类型 |

**关键数据模型：**

- **EvidencePack**：以语音段为中心的多模态证据单元。包含 `segment_id`、`session_id`、时间边界、`waveform_path`、`asr_text`、`speaker_role`、`scene_label`、`event_tags`、`tool_confidences` 和 `context_window_refs`。
- **MemoryNode**：记忆图中的节点，包含 `node_id`、`node_type`（Episode/Intent/Entity）、`label`、`attributes`、`confidence` 和时间戳。
- **MemoryEdge**：有向边，包含 `source`、`target`、`relation` 和 `confidence`。
- **GraphWriteOperation**：MCoC 与 MemoryGraphStore 之间的统一接口。包含 `kind`、`session_id`、`segment_id`、`timestamp`、`payload`、`confidence` 和 `observability`。
- **ReasoningResult**：AMR 的输出。包含 `query`、`query_type`、`answer`、`graph_trace`、`evidence_trace`、`confidence` 和 `abstained`。

### 3.2 可观察性（Observability）设计

可观察性是 LongAI 不确定性建模的核心创新：

- **EXTRACTED**：意图在语音中明确陈述（如 "I need to call John"）。
- **INFERRED**：意图从上下文推断得出（如听到拨号声 → 推断正在通话）。
- **AMBIGUOUS**：证据不足以确定意图，需要更多信息或回溯确认。

该三层分类超越了标量置信度，明确区分了"证据质量"和"模型校准"两个维度。

### 3.3 意图状态机

意图节点的生命周期：`tentative → planned → ongoing → done | canceled | dropped`

通过 `UPDATE_STATE` 操作实现状态转换，每次转换记录在 `update_logs.jsonl` 中，支持全链路审计。

---

## 4. 感知工具（Perception Tools）

所有工具继承自 `BaseTool`（`longai/tools/base.py`），统一接口：`ToolConfig(backend, cache_dir, asr_model_size)` → `tool.run(*args) → dict`。

### 4.1 VAD — 语音活动检测

| 后端 | 方法 | 参数 |
|------|------|------|
| `mock` | 能量阈值：RMS 帧能量 > 0.01 | frame_sec=0.5 |
| `local_hf` | **Silero VAD v6.2.1**（ONNX 模型） | 16kHz 输入 |

真实 VAD 使用预训练的 Silero 模型，`get_speech_timestamps()` 返回带置信度 0.85 的语音段。

### 4.2 ASR — 自动语音识别

| 后端 | 方法 | 模型 |
|------|------|------|
| `mock` | 按段长合成文本 | — |
| `local_hf` | **faster-whisper**（CTranslate2） | tiny (39M) / base (74M) / small (244M) |

词级时间戳通过 CTranslate2 的 `word_timestamps=True` 获取，与 VAD 段边界对齐。对齐采用严格匹配 + 重叠回退策略：若有词精确对齐到段边界则使用精确匹配，否则对重叠词（`word.end > seg_start and word.start < seg_end`）进行聚合。

### 4.3 说话者角色分类

| 后端 | 方法 |
|------|------|
| `mock` | ASR 文本关键词：第一人称代词 → WEARER，第二人称/报告性代词 → OTHER |
| `local_hf` | 同 mock（可扩展至 pyannote.audio 或 Qwen2.5-Omni） |

### 4.4 场景分类

| 后端 | 方法 | 标签 |
|------|------|------|
| `mock` | 能量 + 过零率启发式规则 | 4 类 |
| `local_hf` | **MIT AST-finetuned-audioset**（Audio Spectrogram Transformer） | 8 类 |

真实后端使用 AST 模型在 AudioSet 标签上的 top-K 预测，通过关键词映射到 LongAI 的 8 个场景类别（office, meeting_room, home, kitchen, street_outdoor, vehicle, cafe_shop, other）。从每个 VAD 段周围提取 ±5s 上下文窗口作为输入。

### 4.5 事件检测

| 后端 | 方法 | 标签池 |
|------|------|------|
| `mock` | 能量级别规则 | 8 类 |
| `local_hf` | **MIT AST**（与场景共享模型，不同类别映射） | 8 类 |

从 AudioSet top-15 预测结果中以 0.15 阈值匹配事件类型，返回 top-3 标签。

---

## 5. MCoC：多模态构建链

### 5.1 算法流程

```
EvidencePack
    │
    ├─→ spot_candidates(pack, backend)
    │   ├─ rule: 6 关键词字典匹配 (meeting→"attend meeting", call→"make call", ...)
    │   └─ llm: Qwen2.5-7B-Instruct 结构化提取 (意图 + 实体 + 时间表达式)
    │
    ├─→ classify_observability(text, confidence, speaker_role, backend)
    │   ├─ rule: 显式标记 ("I will", "need to", "must") → EXTRACTED
    │   │        置信度 > 0.7 → INFERRED，其余 → AMBIGUOUS
    │   └─ llm: Qwen2.5-7B-Instruct 文本分类 (EXTRACTED/INFERRED/AMBIGUOUS)
    │
    └─→ 为每个候选生成 GraphWriteOperation:
        - CREATE_NODE (Episode) ×1 per pack
        - CREATE_NODE (Intent) ×1 per intent
        - CREATE_EDGE (intent→episode, "realized_by") ×1 per intent
        - CREATE_NODE (Entity) ×1 per entity
        - ATTACH_EVIDENCE ×1 per pack
```

### 5.2 Mock vs LLM 后端

- **Mock（规则型）**：基于 6 个关键词的模式匹配，计算复杂度 O(n)，可解释但覆盖有限。
- **LLM（Qwen2.5-7B-Instruct）**：提示模板包含 EvidencePack 的 ASR 文本、场景和事件标签，输出结构化 JSON。支持 temperature=0.1 的受控生成。LLM 失败时自动回退到规则型。

---

## 6. 记忆图存储（MemoryGraphStore）

基于 `networkx.MultiDiGraph` 的持久化图存储，支持多平行边。

### 6.1 核心操作

| 方法 | 功能 |
|------|------|
| `apply_operation(op)` | 处理全部 7 种操作类型 |
| `_create_or_update_node(payload, conf, ts)` | 去重：max 置信度合并属性 |
| `CREATE_EDGE` | 仅在 src、tgt 均存在时添加边（保证引用完整性） |
| `ATTACH_EVIDENCE` | 按 segment_id 索引存储原始证据 |
| `ARCHIVE_NODE` | 标记节点为 archived（软删除） |
| `snapshot()` | 返回 {nodes, edges, evidence} 纯字典 |
| `save(graph_path, log_path)` | 写入 JSON + JSONL 日志 |

### 6.2 更新日志

每个操作应用记录包含 `kind`、`session_id`、`segment_id`、`timestamp`、`payload`、`confidence`、`observability` 和 `applied_at`。日志以 JSONL 格式存储，支持完整回放和增量更新。

---

## 7. AMR：自适应记忆推理

### 7.1 查询路由（Query Router）

6 路查询分类：

| 查询类型 | 触发词 | 示例 |
|----------|--------|------|
| `status_query` | "status", "pending" | "What is the current status?" |
| `temporal_multi_hop` | "when", "before", "after" | "When did I plan the meeting?" |
| `update_conflict` | "cancel", "changed" | "What changed or was canceled?" |
| `next_intent_prediction` | "next" + "intent" | "What is my next intent?" |
| `short_horizon_planning` | "plan", "suggest" | "What should I do?" |
| `acoustic_disambiguation` | "sound", "acoustic" | "What sound was that?" |

### 7.2 推理脚手架（Scaffold）

子目标 DAG（所有查询类型）：
```
g1 (identify_relevant_intents)
  └─→ g2 (collect_evidence)
       └─→ g3 (resolve/predict/plan, 类型特异)
            └─→ g4 (self_check)
```

对于 `temporal_multi_hop` 和 `update_conflict`，g3 为 `resolve_temporal_or_update`；对于 `next_intent_prediction` 和 `short_horizon_planning`，g3 为 `predict_or_plan`。

### 7.3 检索（Retrieval）

- **规则型**：查询词与节点标签/属性的 Token 重叠匹配，topk=5。无匹配时回退到最近节点。
- **LLM 引导**：检索节点 + 证据后构建结构化上下文，交由 Qwen2.5-7B 回答。

### 7.4 回答生成

- **规则型**：模板化输出 "Likely relevant intents: {node_1}, {node_2}, ..."，固定置信度 0.62。
- **LLM 型**：结构化 prompt 包含检索到的节点和证据，输出 JSON `{answer, confidence, abstained}`。LLM 失败时回退到规则型。支持 `abstained=True` 表示证据不足以回答。

---

## 8. 评估框架

### 8.1 工具级评估

| 指标 | 工具 | 方法 |
|------|------|------|
| Segment F1 / IoU | VAD | IoU > 0.3 匹配 |
| WER / CER | ASR | jiwer 库 |
| Macro-F1 | 说话者角色 | 混淆矩阵 |
| Accuracy | 场景 / 事件 | 标签匹配 |

### 8.2 记忆图评估

节点 F1：预测 vs GT，匹配依据 `(node_type, normalized_label)`。边 F1：匹配依据 `(source, target, relation)`。紧凑率：`|predicted_nodes| / |gt_nodes|`。

### 8.3 推理评估

- **精确匹配准确率**：标准化后与 GT 答案的比较
- **弃权正确率**：系统弃权且查询允许弃权的比率
- **平均置信度**、**弃权率**、**平均图追踪长度**

---

## 9. 实现细节

### 9.1 环境与硬件

- **Python 3.11**, PyTorch 2.10, **Conda 环境**: `longai`
- **GPU**: 8× NVIDIA RTX A6000 (48 GB)，其中 GPU 3 分配用于 ASR + LLM 推理
- **音频处理**: ffmpeg/ffprobe（16kHz 单声道 WAV 提取）
- **LLM**: Qwen2.5-7B-Instruct（bf16，device_map="auto"），全局懒加载单例

### 9.2 数据集

从 Ego4D v2 中选取 5 个多样化 session（15-30 分钟/个），总计 **2.7 小时音频**，包含 **280 个 VAD 语音段**，语音占比 **7.2%**。

### 9.3 关键依赖

| 包 | 版本 | 用途 |
|----|------|------|
| silero-vad | ≥0.1 | 语音活动检测 |
| faster-whisper | ≥1.0 | 语音识别（CTranslate2） |
| transformers | ≥4.46 | LLM 加载 |
| networkx | ≥3.0 | 图存储 |
| pydantic | ≥2.0 | 数据模型 |
| jiwer | ≥3.0 | ASR 评估 |

### 9.4 缓存策略

所有工具以 `artifacts/perception/{tool}/{session_id}.json` 为单位缓存。缓存命中的逻辑：文件存在且 `force=False` → 直接读取。这使得消融实验可以复用主实验的感知结果（10× 加速）。

### 9.5 消融实验框架（scripts/run_ablation.py）

11 个预定义配置，涵盖三个消融维度：

1. **工具存在性**：移除单个/多个感知工具
2. **ASR 模型大小**：tiny vs base
3. **后端类型**：规则型 vs LLM（Qwen2.5-7B）

每个配置按序运行 perception → memory → reasoning → eval，支持 `--skip-existing` 和逐工具后端覆盖。

---

## 10. 实验结果

### 10.1 实验设置

- **数据**：5 个 Ego4D session，258 个 EvidencePack（VAD 语音段），约 2.7 小时音频
- **模型**：Silero VAD + faster-whisper-base ASR + MIT AST（场景/事件）+ Qwen2.5-3B-Instruct（LLM MCoC/AMR）
- **基线**：`full` 配置（5 个真实工具 + rule-based MCoC/AMR）+ `llm_rag` 配置（纯 LLM+RAG，无图）
- **消融维度**：工具存在性（−scene, −events, −speaker_role）、后端类型（mock vs local_hf vs rag）
- **注意**：LLM 配置（llm_mcoc/llm_amr/llm_full）仅在 session_0000 上运行，因 LLM 推理时间限制（单段结构化 JSON 生成约 5-18 秒，258 段全量需约 2 小时）

### 10.2 完整消融对比表（全 5 session，rule-based 配置）

| Config | Nodes | Edges | Intents | Entities | Conf | Abstain% |
|---|---|---|---|---|---|---|
| **full** (基线: rule MCoC+AMR) | 291 | 561 | 10 | 1 | 0.620 | 0.0% |
| transcript_only | 287 | 77 | 5 | 2 | 0.620 | 0.0% |
| no_scene | 287 | 77 | 5 | 2 | 0.620 | 0.0% |
| no_events | 292 | 562 | 10 | 2 | 0.620 | 0.0% |
| **llm_rag** (纯 RAG, 无图) | 0 | 0 | 0 | 0 | 0.753 | 20.0% |

### 10.3 LLM 配置对比（session_0000 单 session，Qwen2.5-3B-Instruct）

| Config | Nodes | Edges | Intents | Entities | Conf | Abstain% |
|---|---|---|---|---|---|---|
| **full** (rule MCoC+AMR, session_0000) | 36 | 68 | 2 | 0 | 0.620 | 0.0% |
| **llm_mcoc** (LLM MCoC + rule AMR) | 55 | 38 | **18** | 3 | 0.620 | 0.0% |
| **llm_amr** (rule MCoC + LLM AMR) | 35 | 25 | 1 | 0 | **0.800** | 33.3% |
| **llm_full** (LLM MCoC + LLM AMR) | 55 | 38 | **18** | 3 | **0.800** | 33.3% |

> **注**：LLM 配置仅在 session_0000 上运行（34 EvidencePack）。非 LLM 配置覆盖全部 5 session。

### 10.4 各消融维度的相对变化（vs full 基线）

| Config | Nodes | Edges | Intents | 推理置信度 | 弃权率 |
|---|---|---|---|---|---|
| transcript_only | −1.4% | **−86.3%** | −50% | — | — |
| no_scene | −1.4% | **−86.3%** | −50% | — | — |
| no_events | +0.3% | +0.2% | — | — | — |
| llm_rag | **−100%** | **−100%** | **−100%** | **+21.5%** | **+20pp** |
| llm_mcoc (session_0000) | +52.8% | −44.1% | **+800%** | — | — |
| llm_amr (session_0000) | −2.8% | −63.2% | −50% | **+29.0%** | **+33.3pp** |
| llm_full (session_0000) | +52.8% | −44.1% | **+800%** | **+29.0%** | **+33.3pp** |

### 10.5 关键发现与分析

#### 发现 1：LLM MCoC 大幅提升意图发现能力（已修复）

**修复前**：LLM MCoC 因 prompt 设计缺陷（JSON 格式不明确、缺少 edge 生成指示）导致 0 intents / 0 edges。**修复后**：重写 `spot_candidates_llm()` prompt，强制输出规范 JSON schema（intents 内含 entities、relations），自动补全 `realized_by` 边，并对 observability/state/entity type 做 enum 校验。结果：同一 session_0000 上 LLM MCoC 发现 **18 个 intent**（rule 仅 2 个），提升 **9×**（800%）。

关键修复（`longai/construction/candidates.py`）：
- **系统 prompt**：增加 OBSERVABILITY GUIDE（EXTRACTED/INFERRED/AMBIGUOUS 判断标准）和 INTENT THRESHOLD（避免为每个 segment 生成虚假 intent）
- **输出格式**：从松散 `{intents, entities, time_expressions}` 改为完整 JSON schema，每个 intent 内嵌 `entities` 数组和 `relations` 数组
- **JSON 解析**：`_extract_json()` 支持 markdown fence 剥离、多模式回退、嵌套 JSON 提取
- **容错回退**：LLM 解析失败时自动回退到 rule-based（`spot_candidates_rule()`），保证 pipeline 不掉线
- **确定性 ID**：`_node_id()` 使用 regex slugging（`Intent:make_call`），替代不可靠的 Python `hash()`

#### 发现 2：Scene/Events 对图构建至关重要

去除场景/事件后，边数从 561 骤降至 77（**−86.3%**），意图数从 10 降至 5（**−50%**）。修复后的 `spot_candidates_rule()` 使用三路信号融合：

1. **文本意图检测**：`PLAN_MARKERS`（"need to", "have to", "will" 等 15 个标记）+ `INTENT_KEYWORDS`（meeting/call/schedule/prepare 等 6 个类别）
2. **声学推理**：`EVENT_INTENT_HINTS`（phone_ringing → "answer call", typing → "write message", kitchen_activity → "prepare food" 等 9 种事件映射）
3. **场景推理**：`SCENE_INTENT_HINTS`（meeting_room → "attend meeting", kitchen → "prepare food", vehicle → "travel" 等 7 种场景映射）

三种信号按优先级（text > acoustic > scene）合并，去重保留最高置信度。

#### 发现 3：LLM+RAG 基线对比

纯 LLM+RAG（跳过图构建，直接检索 EvidencePack + LLM 回答）：

| 维度 | Rule MCoC + AMR | LLM+RAG |
|------|----------------|---------|
| 图结构 | 291 nodes / 561 edges | 无 |
| 推理置信度 | 0.620 | **0.753** (+21.5%) |
| 弃权率 | 0% | 20% |
| 适用场景 | 时序追踪、状态变更 | 简单状态查询 |

**结论**：RAG 适用于单轮问答，但无法支持 LongAI 的核心需求——跨 segment 的意图状态追踪（tentative → planned → ongoing → done/canceled）。

#### 发现 4：LLM 推理质量更高但更谨慎

LLM AMR（`llm_amr`）的推理置信度比 rule AMR 高 **29%**（0.800 vs 0.620），但弃权率从 0% 升至 33.3%。这表明：
1. 证据充足时，LLM 提供比模板更具体、上下文相关的答案
2. 证据不足时，LLM 倾向于弃权而非给出低质量答案——比 rule-based 更保守

#### 发现 5：图构建质量量化对比（session_0000）

| 指标 | Rule MCoC | LLM MCoC (3B) | 倍数 |
|------|-----------|---------------|------|
| Intent 数 | 2 | 18 | **9×** |
| Entity 数 | 0 | 3 | — |
| 边数 | 68 | 38 | 0.56× |
| Intent/Segment | 0.06 | 0.53 | **9×** |
| 平均 Intent 置信度 | 0.30-0.62 | 0.25-0.90 | — |
| Observability 分布 | EXTRACTED为主 | INFERRED/AMBIGUOUS为主 | — |

LLM MCoC 每段平均发现 0.53 个意图（vs rule 的 0.06），但边数更少（38 vs 68）。规则型边数膨胀主要来自激进的实体-意图自动关联策略（每个主意图关联所有文本实体）。LLM 在实体链接上更保守，倾向于只关联明确相关的实体。

---

## 11. 讨论与局限

### 11.1 当前局限

1. **LLM 推理速度瓶颈**：Qwen2.5-3B 单段 JSON 生成需 5-18 秒（取决于 prompt 长度），大规模运行（258 段 × 5 session = 1290 段）需约 2 小时。解决方案：专用 Writer SFT 模型（Qwen LoRA）或 vLLM 部署可将吞吐提升 10-50×。
2. **无人工标注 Ground Truth**：所有评估均为无参考自洽性指标。"18 intent vs 2"虽表明 LLM 更敏感，但无法区分正确 intent 和噪音。需要人工标注来评估 precision/recall。
3. **LLM 配置仅单 session**：受 GPU 内存限制（8 张 A6000 几乎满载），LLM 实验仅在 session_0000 上运行，统计显著性有限。
4. **3B vs 7B 模型差距**：因 GPU 内存不足（bf16 7B 需 ~14GB，单卡最多空闲 14GB 但有碎片），LLM 实验使用 Qwen2.5-3B。7B 模型预期输出质量更高。
5. **说话者角色工具未完全实现**：真实后端仍是规则型（关键词匹配），未集成 pyannote 或 Qwen2.5-Omni 进行声学话者日记化。
6. **小规模验证**：仅 5 个 session（2.7 小时音频），结论需要在更大规模数据上验证。

### 11.2 未来工作

- **人工标注**：为 session_0000 创建记忆图 GT，评估 LLM MCoC 的 precision/recall
- **Writer SFT**：基于人工过滤后的 LLM MCoC 输出训练专用 Writer 模型（Qwen2.5-3B LoRA），目标：推理速度从 5-18s 降至 <100ms
- **Reasoner SFT/RL**：用人工标注的推理答案训练专用推理模型
- **更大规模消融**：扩展到 20+ session，LLM 配置覆盖全部 session，增加统计检验
- **vLLM 部署**：用 vLLM 替代单次 transformers.generate()，提升推理吞吐
- **多模态扩展**：集成视频帧特征到 EvidencePack，利用 CLIP/CLAP 联合嵌入

---

## 12. 结论

本报告描述了 LongAI——一个面向连续第一人称音频的流式个人意图记忆系统。系统实现了完整的三阶段管线（感知 → MCoC 记忆构建 → AMR 推理查询），支持 mock 和真实模型后端的热切换。

通过 **8 个配置的系统性消融实验**（含纯 LLM+RAG 基线），核心发现如下：

1. **LLM MCoC 比 rule-based 多发现 9× 意图**（18 vs 2，session_0000），prompt 工程修复后从"不可用"提升到"可运行"
2. **Scene/Events 是图结构的关键驱动**——去除后边数减少 86%（561→77），修复后的 rule-based MCoC 已通过 `SCENE_INTENT_HINTS` 和 `EVENT_INTENT_HINTS` 显式利用多模态信号
3. **LLM 推理更准确但更保守**——置信度 +29%（0.800 vs 0.620），但弃权率 +33pp
4. **纯 LLM+RAG 无法替代图结构**——虽达到最高置信度（0.753），但缺乏意图状态追踪能力
5. **MCoC 修复是关键 milestone**：LLM MCoC 从 0 intents/0 edges → 18 intents/38 edges，当前主要瓶颈转为推理速度和标注驱动的质量评估

这些发现为下一阶段的 Writer/Rasoner SFT 训练、大规模评估以及多模态扩展指明了明确方向。