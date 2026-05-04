from __future__ import annotations

import logging
import re

from longai.construction.candidates import spot_candidates
from longai.construction.observability import classify_observability
from longai.construction.ops import make_op
from longai.schema.models import EvidencePack, OperationKind, Observability

logger = logging.getLogger(__name__)

# Regex for slugging labels into safe node_id components
_SLUG_RE = re.compile(r"[^a-zA-Z0-9_]+")


def _slug(text: str) -> str:
    """Convert arbitrary text to a safe, deterministic identifier fragment."""
    return _SLUG_RE.sub("_", text.lower()).strip("_") or "unknown"


def _node_id(session_id: str, node_type: str, label: str) -> str:
    """Deterministic node_id: {node_type}:{slug}. Stable across runs."""
    return f"{node_type}:{_slug(label)}"


def _intent_node_id(session_id: str, label: str) -> str:
    return _node_id(session_id, "Intent", label)


def _entity_node_id(session_id: str, label: str) -> str:
    return _node_id(session_id, "Entity", label)


def _episode_node_id(session_id: str, segment_id: str) -> str:
    # Episode nodes are segment-scoped, so include segment_id for uniqueness
    return f"Episode:{_slug(segment_id)}"


def generate_write_operations(pack: EvidencePack, backend: str = "mock") -> list:
    """Generate GraphWriteOperations from an EvidencePack.

    Args:
        pack: The EvidencePack for one speech segment.
        backend: "mock" for rule-based, "local_hf" for LLM-based.

    Returns:
        List of GraphWriteOperation objects, with nodes ordered before edges.
    """
    cands = spot_candidates(pack, backend=backend)

    # If abstaining, only create Episode + ATTACH_EVIDENCE, no intents
    if cands.get("abstain", False):
        ts = pack.end_time
        episode_id = _episode_node_id(pack.session_id, pack.segment_id)
        ep_label = cands.get("episode", {}).get("label", f"Episode {pack.segment_id}")
        ep_summary = cands.get("episode", {}).get("summary", "")

        ops = []
        ops.append(make_op(
            OperationKind.CREATE_NODE, pack.session_id, pack.segment_id, ts,
            {
                "node_id": episode_id, "node_type": "Episode",
                "label": ep_label,
                "attributes": {
                    "speaker_role": pack.speaker_role.value,
                    "scene": pack.scene_label,
                    "event_tags": pack.event_tags,
                    "summary": ep_summary,
                    "abstain_reason": cands.get("abstain_reason", "no evidence"),
                },
            },
            0.6, Observability.AMBIGUOUS,
        ))
        ops.append(make_op(
            OperationKind.ATTACH_EVIDENCE, pack.session_id, pack.segment_id, ts,
            {
                "segment_id": pack.segment_id, "asr_text": pack.asr_text,
                "scene_label": pack.scene_label, "event_tags": pack.event_tags,
                "start": pack.start_time, "end": pack.end_time,
            },
            0.7, Observability.AMBIGUOUS,
        ))
        return ops

    # ── Determine fallback observability for the episode ──
    fallback_obs = classify_observability(
        pack.asr_text,
        pack.tool_confidences.get("asr", 0.5),
        speaker_role=pack.speaker_role.value,
        scene_label=pack.scene_label,
        event_tags=pack.event_tags,
        backend=backend,
    )

    ts = pack.end_time
    episode_label = cands.get("episode", {}).get("label", f"Episode {pack.segment_id}")
    episode_summary = cands.get("episode", {}).get("summary", "")
    episode_id = _episode_node_id(pack.session_id, pack.segment_id)

    # Phase 1: collect all operations, nodes first, then edges
    node_ops = []
    edge_ops = []

    # ── Episode node ──
    node_ops.append(make_op(
        OperationKind.CREATE_NODE, pack.session_id, pack.segment_id, ts,
        {
            "node_id": episode_id, "node_type": "Episode",
            "label": episode_label,
            "attributes": {
                "speaker_role": pack.speaker_role.value,
                "scene": pack.scene_label,
                "event_tags": pack.event_tags,
                "summary": episode_summary,
            },
        },
        0.6, fallback_obs,
    ))

    # ── Intent nodes + edges ──
    intent_candidates = cands.get("intents", [])
    seen_intent_ids = set()
    seen_entity_ids = set()

    for intent_data in intent_candidates:
        label = intent_data.get("label", "unknown intent")
        intent_id = _intent_node_id(pack.session_id, label)

        # Parse observability from intent data, fall back to computed
        obs_str = intent_data.get("observability", "AMBIGUOUS")
        try:
            obs = Observability(obs_str)
        except ValueError:
            obs = fallback_obs

        intent_conf = float(intent_data.get("confidence", 0.5))
        intent_desc = intent_data.get("description", "")
        intent_state = intent_data.get("state", "planned")
        intent_rationale = intent_data.get("evidence_rationale", "")

        # Create Intent node
        node_ops.append(make_op(
            OperationKind.CREATE_NODE, pack.session_id, pack.segment_id, ts,
            {
                "node_id": intent_id, "node_type": "Intent",
                "label": label,
                "attributes": {
                    "state": intent_state,
                    "owner": "USER" if pack.speaker_role.value == "wearer" else "OTHER",
                    "description": intent_desc,
                    "observability": obs.value,
                    "evidence_rationale": intent_rationale,
                },
            },
            intent_conf, obs,
        ))
        seen_intent_ids.add(intent_id)

        # Create intent → episode edge (realized_by)
        edge_ops.append(make_op(
            OperationKind.CREATE_EDGE, pack.session_id, pack.segment_id, ts,
            {"source": intent_id, "target": episode_id, "relation": "realized_by"},
            intent_conf, obs,
        ))

        # ── Intent-scoped entities + intent→entity edges ──
        for ent_data in intent_data.get("entities", []):
            ent_label = ent_data.get("label", "unknown")
            ent_id = _entity_node_id(pack.session_id, ent_label)
            ent_type = ent_data.get("type", "other")
            ent_role = ent_data.get("role", "other")
            ent_conf = float(ent_data.get("confidence", 0.5)) if "confidence" in ent_data else min(intent_conf, 0.6)

            # Create Entity node (only once per entity label)
            if ent_id not in seen_entity_ids:
                node_ops.append(make_op(
                    OperationKind.CREATE_NODE, pack.session_id, pack.segment_id, ts,
                    {
                        "node_id": ent_id, "node_type": "Entity",
                        "label": ent_label,
                        "attributes": {"type": ent_type},
                    },
                    ent_conf, obs,
                ))
                seen_entity_ids.add(ent_id)

            # Create intent → entity edge based on role
            relation_map = {
                "target": "target",
                "location": "location",
                "object": "involves",
                "participant": "participant",
                "time": "time_anchor",
            }
            relation = relation_map.get(ent_role, "involves")
            edge_ops.append(make_op(
                OperationKind.CREATE_EDGE, pack.session_id, pack.segment_id, ts,
                {"source": intent_id, "target": ent_id, "relation": relation},
                ent_conf, obs,
            ))

    # ── Standalone entities (not tied to a specific intent) ──
    for ent_data in cands.get("entities", []):
        ent_label = ent_data.get("label", "unknown")
        ent_id = _entity_node_id(pack.session_id, ent_label)
        ent_type = ent_data.get("type", "other")
        ent_conf = float(ent_data.get("confidence", 0.5))

        if ent_id not in seen_entity_ids:
            node_ops.append(make_op(
                OperationKind.CREATE_NODE, pack.session_id, pack.segment_id, ts,
                {
                    "node_id": ent_id, "node_type": "Entity",
                    "label": ent_label,
                    "attributes": {"type": ent_type},
                },
                ent_conf, fallback_obs,
            ))
            seen_entity_ids.add(ent_id)

    # ── ATTACH_EVIDENCE ──
    node_ops.append(make_op(
        OperationKind.ATTACH_EVIDENCE, pack.session_id, pack.segment_id, ts,
        {
            "segment_id": pack.segment_id, "asr_text": pack.asr_text,
            "scene_label": pack.scene_label, "event_tags": pack.event_tags,
            "start": pack.start_time, "end": pack.end_time,
        },
        0.7, fallback_obs,
    ))

    # Phase 2: nodes before edges (guarantees referential integrity in graph store)
    return node_ops + edge_ops
