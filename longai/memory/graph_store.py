from __future__ import annotations

import logging
import time
from pathlib import Path

import networkx as nx

from longai.schema.models import GraphWriteOperation, OperationKind
from longai.utils.io import write_json, write_jsonl

logger = logging.getLogger(__name__)


class MemoryGraphStore:
    def __init__(self, rejected_ops_path: Path | None = None):
        self.graph = nx.MultiDiGraph()
        self.evidence: dict[str, dict] = {}
        self.history: list[dict] = []
        self.rejected: list[dict] = []
        self.rejected_ops_path = rejected_ops_path

    def _create_or_update_node(
        self, payload: dict, confidence: float, timestamp: float
    ) -> None:
        node_id = payload["node_id"]
        if node_id not in self.graph:
            self.graph.add_node(
                node_id,
                node_type=payload.get("node_type", "Unknown"),
                label=payload.get("label", node_id),
                attributes=payload.get("attributes", {}),
                confidence=confidence,
                created_at=timestamp,
                last_updated_at=timestamp,
                archived=False,
            )
            return
        node = self.graph.nodes[node_id]
        node["last_updated_at"] = timestamp
        node["confidence"] = max(
            float(node.get("confidence", 0.5)), confidence
        )
        attrs = node.get("attributes", {})
        attrs.update(payload.get("attributes", {}))
        node["attributes"] = attrs

    def apply_operation(self, op: GraphWriteOperation) -> bool:
        """Apply one graph write operation. Returns False if rejected."""
        kind = op.kind
        p = op.payload
        now = time.time()

        if kind in (OperationKind.CREATE_NODE, OperationKind.UPDATE_NODE):
            self._create_or_update_node(p, op.confidence, op.timestamp)

        elif kind == OperationKind.CREATE_EDGE:
            src = p["source"]
            tgt = p["target"]
            relation = p.get("relation", "related_to")
            if src not in self.graph:
                self._reject(op, f"source node '{src}' does not exist")
                return False
            if tgt not in self.graph:
                self._reject(op, f"target node '{tgt}' does not exist")
                return False
            # Deduplicate: skip if edge with same (src, tgt, relation) already exists
            if self.graph.has_edge(src, tgt):
                for key in self.graph[src][tgt]:
                    edge_attrs = self.graph[src][tgt][key]
                    if edge_attrs.get("relation") == relation:
                        edge_attrs["confidence"] = max(
                            float(edge_attrs.get("confidence", 0.5)), op.confidence
                        )
                        break
                else:
                    self.graph.add_edge(
                        src, tgt,
                        relation=relation,
                        confidence=op.confidence,
                    )
            else:
                self.graph.add_edge(
                    src, tgt,
                    relation=relation,
                    confidence=op.confidence,
                )

        elif kind == OperationKind.UPDATE_STATE:
            node_id = p["node_id"]
            if node_id in self.graph:
                attrs = self.graph.nodes[node_id].get("attributes", {})
                attrs["state"] = p.get("state", "tentative")
                self.graph.nodes[node_id]["attributes"] = attrs
                self.graph.nodes[node_id]["last_updated_at"] = op.timestamp
            else:
                self._reject(op, f"node '{node_id}' not found for state update")

        elif kind == OperationKind.ARCHIVE_NODE:
            node_id = p["node_id"]
            if node_id in self.graph:
                self.graph.nodes[node_id]["archived"] = True
                self.graph.nodes[node_id]["last_updated_at"] = op.timestamp
            else:
                self._reject(op, f"node '{node_id}' not found for archive")

        elif kind == OperationKind.ATTACH_EVIDENCE:
            self.evidence[p["segment_id"]] = p

        elif kind == OperationKind.MERGE_NODE:
            # For now, treat as CREATE_NODE (upsert)
            self._create_or_update_node(p, op.confidence, op.timestamp)

        self.history.append(
            {
                "kind": kind.value,
                "session_id": op.session_id,
                "segment_id": op.segment_id,
                "timestamp": op.timestamp,
                "payload": p,
                "confidence": op.confidence,
                "observability": op.observability.value,
                "applied_at": now,
            }
        )
        return True

    def _reject(self, op: GraphWriteOperation, reason: str) -> None:
        entry = {
            "kind": op.kind.value,
            "session_id": op.session_id,
            "segment_id": op.segment_id,
            "reason": reason,
            "payload": op.payload,
        }
        self.rejected.append(entry)
        logger.warning(
            "Rejected %s operation [%s/%s]: %s",
            op.kind.value, op.session_id, op.segment_id, reason,
        )

    def snapshot(self) -> dict:
        nodes = []
        for node_id, attrs in self.graph.nodes(data=True):
            nodes.append({"node_id": node_id, **attrs})
        edges = []
        for src, tgt, key, attrs in self.graph.edges(data=True, keys=True):
            edges.append({"source": src, "target": tgt, "key": key, **attrs})
        return {"nodes": nodes, "edges": edges, "evidence": self.evidence}

    def save(self, graph_path: Path, log_path: Path) -> None:
        write_json(graph_path, self.snapshot())
        write_jsonl(log_path, self.history)
        if self.rejected and self.rejected_ops_path:
            write_jsonl(self.rejected_ops_path, self.rejected)

    def stats(self) -> dict:
        node_types = {}
        for _, attrs in self.graph.nodes(data=True):
            nt = attrs.get("node_type", "Unknown")
            node_types[nt] = node_types.get(nt, 0) + 1
        edge_relations = {}
        for _, _, attrs in self.graph.edges(data=True):
            rel = attrs.get("relation", "unknown")
            edge_relations[rel] = edge_relations.get(rel, 0) + 1
        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "node_types": node_types,
            "edge_relations": edge_relations,
            "rejected_ops": len(self.rejected),
        }
