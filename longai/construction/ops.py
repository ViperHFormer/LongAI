from __future__ import annotations

from longai.schema.models import GraphWriteOperation, OperationKind, Observability


def make_op(
    kind: OperationKind,
    session_id: str,
    segment_id: str,
    timestamp: float,
    payload: dict,
    confidence: float,
    observability: Observability,
) -> GraphWriteOperation:
    return GraphWriteOperation(
        kind=kind,
        session_id=session_id,
        segment_id=segment_id,
        timestamp=timestamp,
        payload=payload,
        confidence=confidence,
        observability=observability,
    )
