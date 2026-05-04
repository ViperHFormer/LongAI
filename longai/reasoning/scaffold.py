from __future__ import annotations


def build_scaffold(query_type: str) -> list[dict]:
    base = [
        {"id": "g1", "goal": "identify_relevant_intents", "deps": []},
        {"id": "g2", "goal": "collect_evidence", "deps": ["g1"]},
    ]
    if query_type in {"temporal_multi_hop", "update_conflict"}:
        base.append({"id": "g3", "goal": "resolve_temporal_or_update", "deps": ["g2"]})
    if query_type in {"next_intent_prediction", "short_horizon_planning"}:
        base.append({"id": "g3", "goal": "predict_or_plan", "deps": ["g2"]})
    base.append({"id": "g4", "goal": "self_check", "deps": [base[-1]["id"]]})
    return base
