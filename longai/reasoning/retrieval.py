from __future__ import annotations

from typing import Any


def retrieve(graph_snapshot: dict[str, Any], query: str, topk: int = 5) -> tuple[list[dict], list[dict]]:
    q = query.lower()
    matched_nodes = []
    for node in graph_snapshot.get("nodes", []):
        label = str(node.get("label", "")).lower()
        attrs = str(node.get("attributes", "")).lower()
        if any(token in (label + " " + attrs) for token in q.split() if len(token) > 2):
            matched_nodes.append(node)
    matched_nodes = matched_nodes[:topk]

    if not matched_nodes:
        matched_nodes = graph_snapshot.get("nodes", [])[:topk]

    evidence_trace = []
    for seg_id, ev in graph_snapshot.get("evidence", {}).items():
        payload = str(ev).lower()
        if any(token in payload for token in q.split() if len(token) > 2):
            evidence_trace.append({"segment_id": seg_id, "payload": ev})
    evidence_trace = evidence_trace[:topk]
    return matched_nodes, evidence_trace
