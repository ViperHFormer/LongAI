#!/usr/bin/env python
"""Regression script: run MCoC on synthetic EvidencePacks and verify output.

Usage:
  python scripts/debug_mcoc_one_pack.py [--backend mock|local_hf]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from longai.construction.mcoc import generate_write_operations
from longai.memory.graph_store import MemoryGraphStore
from longai.schema.models import (
    EvidencePack,
    SpeakerRole,
    OperationKind,
    Observability,
)

# ── Synthetic test cases ───────────────────────────────────────────────

def make_pack(
    segment_id: str,
    asr_text: str,
    speaker_role: SpeakerRole = SpeakerRole.WEARER,
    scene_label: str = "office",
    event_tags: list | None = None,
    tool_confidences: dict | None = None,
) -> EvidencePack:
    return EvidencePack(
        segment_id=segment_id,
        session_id="test_session",
        start_time=0.0,
        end_time=5.0,
        waveform_path="/tmp/test.wav",
        speaker_role=speaker_role,
        asr_text=asr_text,
        scene_label=scene_label,
        event_tags=event_tags or [],
        tool_confidences=tool_confidences or {"asr": 0.6, "vad": 0.85},
    )


TEST_CASES = [
    {
        "name": "Case 1: explicit intent",
        "pack": make_pack(
            "seg_001",
            "I need to call John after lunch",
            SpeakerRole.WEARER,
            "office",
            ["speech"],
        ),
        "expect": {
            "min_intents": 1,
            "min_edges": 2,  # intent→episode + intent→entity
            "min_entities": 1,
            "expect_observability": "EXTRACTED",
        },
    },
    {
        "name": "Case 2: acoustic inference",
        "pack": make_pack(
            "seg_002",
            "Hello?",
            SpeakerRole.WEARER,
            "office",
            ["phone_ringing", "conversation"],
        ),
        "expect": {
            "min_intents": 1,
            "min_edges": 1,
            "min_entities": 0,
            "expect_observability": None,  # INFERRED expected but rule might vary
        },
    },
    {
        "name": "Case 3: ambiguous evidence",
        "pack": make_pack(
            "seg_003",
            "maybe later",
            SpeakerRole.UNKNOWN,
            "other",
            [],
        ),
        "expect": {
            "min_intents": 0,
            "min_edges": 0,
            "min_entities": 0,
            "expect_observability": "AMBIGUOUS",
            "allow_abstain": True,
        },
    },
    {
        "name": "Case 4: kitchen scene with events",
        "pack": make_pack(
            "seg_004",
            "I'm going to cook dinner tonight",
            SpeakerRole.WEARER,
            "kitchen",
            ["kitchen_activity", "speech"],
        ),
        "expect": {
            "min_intents": 1,
            "min_edges": 2,
            "min_entities": 1,
            "expect_observability": "EXTRACTED",
        },
    },
    {
        "name": "Case 5: empty text, scene hints",
        "pack": make_pack(
            "seg_005",
            "",
            SpeakerRole.UNKNOWN,
            "meeting_room",
            ["conversation"],
        ),
        "expect": {
            "min_intents": 0,  # rule may or may not infer
            "min_edges": 0,
            "min_entities": 0,
            "allow_abstain": True,
        },
    },
    {
        "name": "Case 6: wearer planning with entities",
        "pack": make_pack(
            "seg_006",
            "I need to email Sarah about the project deadline next week",
            SpeakerRole.WEARER,
            "office",
            ["typing", "speech"],
        ),
        "expect": {
            "min_intents": 1,
            "min_edges": 3,  # intent→episode + intent→entity (Sarah) + intent→entity (project/deadline)
            "min_entities": 2,
            "expect_observability": "EXTRACTED",
        },
    },
    {
        "name": "Case 7: meeting room with discussion",
        "pack": make_pack(
            "seg_007",
            "Let's discuss the Q4 results",
            SpeakerRole.WEARER,
            "meeting_room",
            ["conversation", "speech"],
        ),
        "expect": {
            "min_intents": 1,
            "min_edges": 1,
            "min_entities": 0,
        },
    },
]


def run_test(backend: str) -> dict:
    """Run all test cases through MCoC and return results."""
    results = {}

    for tc in TEST_CASES:
        name = tc["name"]
        pack = tc["pack"]
        expected = tc["expect"]

        print(f"\n{'='*70}")
        print(f"  {name}  [backend={backend}]")
        print(f"  asr_text: \"{pack.asr_text}\"")
        print(f"  speaker_role: {pack.speaker_role.value}, scene: {pack.scene_label}")
        print(f"  event_tags: {pack.event_tags}")
        print(f"{'='*70}")

        ops = generate_write_operations(pack, backend=backend)

        # Apply to store
        store = MemoryGraphStore()
        for op in ops:
            store.apply_operation(op)

        snap = store.snapshot()
        stats = store.stats()

        # Print generated operations
        print(f"\n  Generated {len(ops)} operations:")
        for op in ops:
            if op.kind == OperationKind.CREATE_NODE:
                node_type = op.payload.get("node_type", "?")
                label = op.payload.get("label", "?")
                obs = op.observability.value
                print(f"    [CREATE_NODE] {node_type}: {label} (obs={obs}, conf={op.confidence:.2f})")
            elif op.kind == OperationKind.CREATE_EDGE:
                src = op.payload.get("source", "?")
                tgt = op.payload.get("target", "?")
                rel = op.payload.get("relation", "?")
                print(f"    [CREATE_EDGE] {src} → {tgt} ({rel})")
            elif op.kind == OperationKind.ATTACH_EVIDENCE:
                print(f"    [ATTACH_EVIDENCE] {op.payload.get('segment_id', '?')}")

        print(f"\n  Graph stats: nodes={stats['total_nodes']}, edges={stats['total_edges']}, "
              f"rejected={stats['rejected_ops']}, intents={stats['node_types'].get('Intent', 0)}, "
              f"entities={stats['node_types'].get('Entity', 0)}")

        # Verify
        checks = []
        n_intents = stats["node_types"].get("Intent", 0)
        n_entities = stats["node_types"].get("Entity", 0)
        n_edges = stats["total_edges"]

        # Check 1: Min intents
        if "min_intents" in expected and n_intents < expected["min_intents"]:
            checks.append(f"FAIL: expected >= {expected['min_intents']} intents, got {n_intents}")

        # Check 2: Min edges
        if "min_edges" in expected and n_edges < expected["min_edges"]:
            checks.append(f"FAIL: expected >= {expected['min_edges']} edges, got {n_edges}")

        # Check 3: Min entities
        if "min_entities" in expected and n_entities < expected["min_entities"]:
            checks.append(f"FAIL: expected >= {expected['min_entities']} entities, got {n_entities}")

        # Check 4: If intents > 0, ensure edges > 0 (critical regression test)
        if n_intents > 0 and n_edges == 0:
            if not expected.get("allow_abstain", False):
                checks.append(f"CRITICAL FAIL: {n_intents} intents but 0 edges — regression detected!")

        # Check 5: If entities > 0 and intents > 0, there should be entity edges
        if n_intents > 0 and n_entities > 0:
            entity_edges = sum(
                1 for _, _, attrs in store.graph.edges(data=True)
                if attrs.get("relation") in ("target", "location", "participant",
                                             "time_anchor", "involves")
            )
            if entity_edges == 0:
                checks.append(f"WARN: {n_intents} intents and {n_entities} entities but 0 entity edges")

        if checks:
            print(f"\n  VERIFICATION FAILURES:")
            for c in checks:
                print(f"    ❌ {c}")
        else:
            print(f"\n  ✅ All checks passed")

        results[name] = {
            "checks": checks,
            "stats": stats,
            "passed": len(checks) == 0,
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="Debug/regression test for MCoC")
    parser.add_argument("--backend", type=str, default="mock",
                        help="Backend: mock or local_hf")
    args = parser.parse_args()

    print(f"Running MCoC regression tests [backend={args.backend}]")
    print(f"Test cases: {len(TEST_CASES)}")

    results = run_test(args.backend)

    # Summary
    passed = sum(1 for v in results.values() if v["passed"])
    failed = sum(1 for v in results.values() if not v["passed"])
    print(f"\n{'='*70}")
    print(f"SUMMARY: {passed} passed, {failed} failed out of {len(TEST_CASES)}")
    print(f"{'='*70}")

    if failed > 0:
        print("\nFAILING CASES:")
        for name, v in results.items():
            if not v["passed"]:
                print(f"  - {name}: {v['checks']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
