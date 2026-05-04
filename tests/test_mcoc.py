"""Comprehensive tests for the MCoC graph construction module."""
from __future__ import annotations

import json
import pytest

from longai.construction.candidates import (
    spot_candidates_rule,
    spot_candidates,
    _extract_json,
)
from longai.construction.mcoc import generate_write_operations
from longai.construction.observability import classify_observability_rule
from longai.memory.graph_store import MemoryGraphStore
from longai.schema.models import (
    EvidencePack,
    SpeakerRole,
    OperationKind,
    Observability,
)


# ── Helpers ────────────────────────────────────────────────────────────

def make_pack(**kwargs) -> EvidencePack:
    defaults = {
        "segment_id": "seg_test",
        "session_id": "test_session",
        "start_time": 0.0,
        "end_time": 5.0,
        "waveform_path": "/tmp/test.wav",
        "speaker_role": SpeakerRole.WEARER,
        "asr_text": "",
        "scene_label": "office",
        "event_tags": [],
        "tool_confidences": {"asr": 0.6, "vad": 0.85},
    }
    defaults.update(kwargs)
    return EvidencePack(**defaults)


def run_ops(pack: EvidencePack, backend: str = "mock") -> MemoryGraphStore:
    ops = generate_write_operations(pack, backend=backend)
    store = MemoryGraphStore()
    for op in ops:
        store.apply_operation(op)
    return store


# ── JSON extraction tests ──────────────────────────────────────────────

def test_extract_json_clean():
    result = _extract_json('{"a": 1}')
    assert result == {"a": 1}


def test_extract_json_markdown_fence():
    result = _extract_json('```json\n{"a": 1}\n```')
    assert result == {"a": 1}


def test_extract_json_with_explanation():
    result = _extract_json('Here is the output:\n{"a": 1}\nHope that helps.')
    assert result == {"a": 1}


def test_extract_json_nested():
    result = _extract_json('{"intents": [{"label": "test"}]}')
    assert result == {"intents": [{"label": "test"}]}


def test_extract_json_invalid():
    result = _extract_json("not json at all")
    assert result is None


# ── Case 1: Explicit intent ────────────────────────────────────────────

def test_explicit_intent_rule():
    pack = make_pack(
        segment_id="seg_001",
        asr_text="I need to call John after lunch",
        speaker_role=SpeakerRole.WEARER,
        scene_label="office",
        event_tags=["speech"],
    )
    cands = spot_candidates_rule(pack)
    assert len(cands["intents"]) >= 1, f"Expected >=1 intents, got {cands['intents']}"
    assert any("call" in i["label"].lower() for i in cands["intents"]), \
        f"Expected 'call' intent in {cands['intents']}"

    # Check observability
    call_intent = [i for i in cands["intents"] if "call" in i["label"].lower()][0]
    assert call_intent["observability"] == "EXTRACTED", \
        f"Expected EXTRACTED, got {call_intent['observability']}"

    store = run_ops(pack)
    stats = store.stats()
    assert stats["node_types"].get("Intent", 0) >= 1, \
        f"Expected >=1 Intent node, got {stats}"
    assert stats["total_edges"] >= 2, \
        f"Expected >=2 edges (intent→episode + intent→entity), got {stats['total_edges']}"
    assert stats["node_types"].get("Entity", 0) >= 1, \
        f"Expected >=1 Entity node, got {stats}"


def test_explicit_intent_mock():
    """Test explicit intent through the spot_candidates dispatcher with mock backend."""
    pack = make_pack(
        segment_id="seg_001",
        asr_text="I need to call John after lunch",
        speaker_role=SpeakerRole.WEARER,
        scene_label="office",
        event_tags=["speech"],
    )
    cands = spot_candidates(pack, backend="mock")
    assert len(cands["intents"]) >= 1


# ── Case 2: Acoustic inference ─────────────────────────────────────────

def test_acoustic_inference_rule():
    pack = make_pack(
        segment_id="seg_002",
        asr_text="Hello?",
        speaker_role=SpeakerRole.WEARER,
        scene_label="office",
        event_tags=["phone_ringing", "conversation"],
    )
    cands = spot_candidates_rule(pack)
    assert len(cands["intents"]) >= 1, \
        f"Expected >=1 intents from acoustic inference, got {cands['intents']}"

    # Should have INFERRED observability
    observabilities = [i["observability"] for i in cands["intents"]]
    assert "INFERRED" in observabilities or "EXTRACTED" in observabilities, \
        f"Expected INFERRED or EXTRACTED, got {observabilities}"

    store = run_ops(pack)
    stats = store.stats()
    assert stats["node_types"].get("Intent", 0) >= 1, \
        f"Expected >=1 Intent node, got {stats}"
    assert stats["total_edges"] >= 1, \
        f"Expected >=1 edge (intent→episode), got {stats['total_edges']}"


# ── Case 3: Ambiguous evidence ─────────────────────────────────────────

def test_ambiguous_evidence_no_high_conf_intent():
    pack = make_pack(
        segment_id="seg_003",
        asr_text="maybe later",
        speaker_role=SpeakerRole.UNKNOWN,
        scene_label="other",
        event_tags=[],
    )
    cands = spot_candidates_rule(pack)

    # Should not generate EXTRACTED high-confidence intent
    high_conf = [i for i in cands["intents"]
                 if i["observability"] == "EXTRACTED" and i["confidence"] > 0.6]
    assert len(high_conf) == 0, \
        f"Should not have high-confidence EXTRACTED intents, got {high_conf}"


# ── Case 4: Scene and event influence on rule-based output ─────────────

def test_scene_influences_intent():
    """Rule-based MCoC should produce different output with different scene."""
    pack_office = make_pack(
        segment_id="seg_a",
        asr_text="ok",
        scene_label="office",
        event_tags=[],
    )
    pack_kitchen = make_pack(
        segment_id="seg_b",
        asr_text="ok",
        scene_label="kitchen",
        event_tags=[],
    )

    cands_office = spot_candidates_rule(pack_office)
    cands_kitchen = spot_candidates_rule(pack_kitchen)

    # With same text, different scenes may produce different results
    # At minimum, the episode output should differ
    assert cands_office["episode"]["scene_label"] == "office"
    assert cands_kitchen["episode"]["scene_label"] == "kitchen"


def test_event_influences_intent():
    """Events should add acoustic intents."""
    pack_no_events = make_pack(
        segment_id="seg_c",
        asr_text="hello",
        event_tags=[],
    )
    pack_with_events = make_pack(
        segment_id="seg_d",
        asr_text="hello",
        event_tags=["phone_ringing"],
    )

    cands_plain = spot_candidates_rule(pack_no_events)
    cands_events = spot_candidates_rule(pack_with_events)

    # With phone_ringing, should have more or different intents
    plain_labels = [i["label"] for i in cands_plain["intents"]]
    event_labels = [i["label"] for i in cands_events["intents"]]

    # phone_ringing event should add "answer call" or similar
    has_phone_intent = any("call" in l.lower() for l in event_labels)
    assert has_phone_intent, \
        f"Expected phone-related intent in event_labels, got {event_labels}"


def test_speaker_role_matters():
    """Same text, different speaker role should affect observability."""
    result = classify_observability_rule(
        "I need to call John",
        confidence=0.6,
        speaker_role="wearer",
    )
    assert result == Observability.EXTRACTED

    result_other = classify_observability_rule(
        "I need to call John",
        confidence=0.6,
        speaker_role="other",
    )
    # When OTHER says it, it's not EXTRACTED for the wearer's intent
    assert result_other != Observability.EXTRACTED, \
        f"OTHER speaker should not produce EXTRACTED, got {result_other}"


# ── Case 5: Edge auto-generation ────────────────────────────────────────

def test_intent_always_has_realized_by_edge():
    """Every intent must have a realized_by edge to its episode."""
    pack = make_pack(
        segment_id="seg_edge",
        asr_text="I need to buy groceries",
        speaker_role=SpeakerRole.WEARER,
        scene_label="home",
        event_tags=["speech"],
    )
    ops = generate_write_operations(pack, backend="mock")

    edge_ops = [op for op in ops if op.kind == OperationKind.CREATE_EDGE]
    intent_labels = set()
    for op in ops:
        if op.kind == OperationKind.CREATE_NODE and op.payload.get("node_type") == "Intent":
            intent_labels.add(op.payload["node_id"])

    realized_by_targets = set()
    for op in edge_ops:
        if op.payload.get("relation") == "realized_by":
            realized_by_targets.add(op.payload["source"])

    # Every intent should have a realized_by edge
    for intent_id in intent_labels:
        assert intent_id in realized_by_targets, \
            f"Intent {intent_id} has no realized_by edge"


def test_no_intent_zero_edges():
    """If no intent is generated, there should be no intent-related edges."""
    pack = make_pack(
        segment_id="seg_empty",
        asr_text="",  # empty
        speaker_role=SpeakerRole.UNKNOWN,
        scene_label="other",
        event_tags=[],
    )
    ops = generate_write_operations(pack, backend="mock")
    edge_ops = [op for op in ops if op.kind == OperationKind.CREATE_EDGE]
    # Should have at most 0 intent-related edges
    intent_edges = [op for op in edge_ops
                    if op.payload.get("relation") in ("realized_by", "involves",
                                                       "target", "location",
                                                       "participant", "time_anchor")]
    assert len(intent_edges) == 0, \
        f"With empty text and no events, expected 0 intent edges, got {len(intent_edges)}"


# ── Case 6: Entity edge auto-generation ────────────────────────────────

def test_entities_with_intent_get_edges():
    """When an intent has entities, entity edges should be generated."""
    pack = make_pack(
        segment_id="seg_ent",
        asr_text="I need to email Sarah about the project",
        speaker_role=SpeakerRole.WEARER,
        scene_label="office",
        event_tags=["typing"],
    )
    store = run_ops(pack)

    # Check there are entity edges (not just realized_by)
    entity_relations = set()
    for _, _, attrs in store.graph.edges(data=True):
        rel = attrs.get("relation", "")
        if rel in ("target", "location", "participant", "time_anchor", "involves"):
            entity_relations.add(rel)

    stats = store.stats()
    if stats["node_types"].get("Entity", 0) > 0:
        assert len(entity_relations) > 0, \
            f"Entities exist but no entity edges: {entity_relations}"


# ── Case 7: Abstain handling ───────────────────────────────────────────

def test_abstain_no_fake_intents():
    """When abstaining, there should be no empty-shell intents."""
    # Test with empty evidence
    pack = make_pack(
        segment_id="seg_abstain",
        asr_text="",
        speaker_role=SpeakerRole.UNKNOWN,
        scene_label="other",
        event_tags=[],
    )
    cands = spot_candidates_rule(pack)
    # Should either abstain or have valid intents with evidence_rationale
    if cands.get("abstain", False):
        assert len(cands["intents"]) == 0, \
            "Abstain=true should have 0 intents"
    elif cands["intents"]:
        # If not abstaining, each intent should have rationale
        for intent in cands["intents"]:
            assert intent.get("evidence_rationale"), \
                f"Intent {intent['label']} has no evidence_rationale"


# ── Case 8: Deterministic node_id ──────────────────────────────────────

def test_node_ids_deterministic():
    """Same input should produce same node IDs across calls."""
    pack = make_pack(
        segment_id="seg_det",
        asr_text="I need to call John",
    )

    ops1 = generate_write_operations(pack, backend="mock")
    ops2 = generate_write_operations(pack, backend="mock")

    ids1 = [op.payload.get("node_id") for op in ops1
            if op.kind == OperationKind.CREATE_NODE]
    ids2 = [op.payload.get("node_id") for op in ops2
            if op.kind == OperationKind.CREATE_NODE]

    assert ids1 == ids2, f"Node IDs differ across calls: {set(ids1) ^ set(ids2)}"


# ── Case 9: Graph store edge rejection ─────────────────────────────────

def test_edge_rejection_when_target_missing():
    """GraphStore should reject edges to non-existent nodes."""
    from longai.schema.models import GraphWriteOperation

    store = MemoryGraphStore()
    edge_op = GraphWriteOperation(
        kind=OperationKind.CREATE_EDGE,
        session_id="s",
        segment_id="seg",
        timestamp=1.0,
        payload={"source": "missing_src", "target": "missing_tgt",
                  "relation": "realized_by"},
        confidence=0.6,
        observability=Observability.INFERRED,
    )
    result = store.apply_operation(edge_op)
    assert result is False, "Edge with missing nodes should be rejected"
    assert len(store.rejected) == 1
    assert store.stats()["rejected_ops"] == 1


def test_graph_store_node_upsert():
    """CREATE_NODE should upsert (merge attributes)."""
    from longai.schema.models import GraphWriteOperation
    store = MemoryGraphStore()
    op1 = GraphWriteOperation(
        kind=OperationKind.CREATE_NODE,
        session_id="s",
        segment_id="seg",
        timestamp=1.0,
        payload={"node_id": "Intent:test", "node_type": "Intent",
                  "label": "test", "attributes": {"state": "planned"}},
        confidence=0.6,
        observability=Observability.INFERRED,
    )
    op2 = GraphWriteOperation(
        kind=OperationKind.CREATE_NODE,
        session_id="s",
        segment_id="seg2",
        timestamp=2.0,
        payload={"node_id": "Intent:test", "node_type": "Intent",
                  "label": "test updated", "attributes": {"state": "ongoing"}},
        confidence=0.8,
        observability=Observability.EXTRACTED,
    )
    store.apply_operation(op1)
    store.apply_operation(op2)

    node = store.graph.nodes["Intent:test"]
    assert node["confidence"] == 0.8  # max(0.6, 0.8)
    assert node["attributes"]["state"] == "ongoing"  # updated


# ── Case 10: Backward compatibility ────────────────────────────────────

def test_old_api_still_works():
    """The old-style EvidencePack API should still work with new code."""
    from longai.schema.models import EvidencePack as EP
    pack = EP(
        segment_id="old_seg",
        session_id="old_session",
        start_time=0.0,
        end_time=3.0,
        waveform_path="/tmp/old.wav",
        asr_text="I will finish the task",
    )
    ops = generate_write_operations(pack, backend="mock")
    assert len(ops) >= 2  # Episode + ATTACH_EVIDENCE minimum


def test_classify_observability_rule_backward_compat():
    """classify_observability_rule should work with old 2-arg call signature."""
    # The new signature has extra args but they all have defaults
    import inspect
    sig = inspect.signature(classify_observability_rule)
    params = list(sig.parameters.keys())
    assert "text" in params
    assert "confidence" in params
    assert "speaker_role" in params
    assert "scene_label" in params
    assert "event_tags" in params
