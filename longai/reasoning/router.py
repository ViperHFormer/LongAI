from __future__ import annotations


def route_query_rule(query: str) -> str:
    q = query.lower()
    if "status" in q or "pending" in q:
        return "status_query"
    if "when" in q or "before" in q or "after" in q:
        return "temporal_multi_hop"
    if "cancel" in q or "changed" in q or "update" in q:
        return "update_conflict"
    if "next" in q and "intent" in q:
        return "next_intent_prediction"
    if "plan" in q or "suggest" in q:
        return "short_horizon_planning"
    if "sound" in q or "acoustic" in q or "voice" in q:
        return "acoustic_disambiguation"
    return "status_query"


def route_query_llm(query: str) -> str:
    import json
    from longai.utils.llm import generate

    types = [
        "status_query", "temporal_multi_hop", "update_conflict",
        "next_intent_prediction", "short_horizon_planning", "acoustic_disambiguation",
    ]
    prompt = f"""Classify this query about personal intent memory into one type:
- status_query: asking about current state of intents
- temporal_multi_hop: asking about when something happens
- update_conflict: asking about changes, cancellations, or conflicts
- next_intent_prediction: asking what comes next
- short_horizon_planning: asking for plan suggestions
- acoustic_disambiguation: asking about sound or audio

Query: "{query}"

Return ONLY the type name from the list above."""
    try:
        result = generate(prompt, max_new_tokens=64).strip().lower()
        for t in types:
            if t in result:
                return t
    except Exception:
        pass
    return route_query_rule(query)


def route_query(query: str, backend: str = "mock") -> str:
    if backend == "local_hf":
        return route_query_llm(query)
    return route_query_rule(query)
