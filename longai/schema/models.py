from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SpeakerRole(str, Enum):
    WEARER = "wearer"
    OTHER = "other"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class Observability(str, Enum):
    EXTRACTED = "EXTRACTED"
    INFERRED = "INFERRED"
    AMBIGUOUS = "AMBIGUOUS"


class OperationKind(str, Enum):
    CREATE_NODE = "CREATE_NODE"
    UPDATE_NODE = "UPDATE_NODE"
    MERGE_NODE = "MERGE_NODE"
    CREATE_EDGE = "CREATE_EDGE"
    UPDATE_STATE = "UPDATE_STATE"
    ATTACH_EVIDENCE = "ATTACH_EVIDENCE"
    ARCHIVE_NODE = "ARCHIVE_NODE"


class EvidenceRef(BaseModel):
    evidence_id: str
    modality: str
    source_tool: str
    start_time: float
    end_time: float
    payload: dict[str, Any] = Field(default_factory=dict)
    observability: Observability
    confidence: float = 0.5


class EvidencePack(BaseModel):
    segment_id: str
    session_id: str
    start_time: float
    end_time: float
    waveform_path: str
    speaker_role: SpeakerRole = SpeakerRole.UNKNOWN
    asr_text: str = ""
    scene_label: str = "other"
    event_tags: list[str] = Field(default_factory=list)
    affective_tags: list[str] = Field(default_factory=list)
    tool_confidences: dict[str, float] = Field(default_factory=dict)
    context_window_refs: list[dict[str, Any]] = Field(default_factory=list)


class MemoryNode(BaseModel):
    node_id: str
    node_type: str
    label: str
    attributes: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 0.5
    created_at: float
    last_updated_at: float


class MemoryEdge(BaseModel):
    edge_id: str
    source: str
    target: str
    relation: str
    confidence: float = 0.5
    attributes: dict[str, Any] = Field(default_factory=dict)


class GraphWriteOperation(BaseModel):
    kind: OperationKind
    session_id: str
    segment_id: str
    timestamp: float
    payload: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 0.5
    observability: Observability = Observability.AMBIGUOUS


class ReasoningResult(BaseModel):
    query: str
    query_type: str
    answer: str
    graph_trace: list[dict[str, Any]] = Field(default_factory=list)
    evidence_trace: list[dict[str, Any]] = Field(default_factory=list)
    confidence: float = 0.0
    abstained: bool = False
