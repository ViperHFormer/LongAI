from __future__ import annotations

import logging

from longai.schema.models import Observability

logger = logging.getLogger(__name__)


def classify_observability_rule(
    text: str,
    confidence: float,
    speaker_role: str = "unknown",
    scene_label: str = "other",
    event_tags: list | None = None,
) -> Observability:
    """Rule-based observability classification using all available context fields."""
    low = text.lower().strip()
    events = event_tags or []

    if not low:
        # No text — check if events hint at activity
        if events and any(e in ("phone_ringing", "conversation", "typing") for e in events):
            return Observability.INFERRED
        return Observability.AMBIGUOUS

    # Explicit first-person intent markers → EXTRACTED
    explicit_markers = [
        "i will", "i need to", "i have to", "i must", "i'm going to", "i am going to",
        "i plan to", "i want to", "i should", "i'll", "i gotta",
        "need to", "have to", "must", "going to", "plan to",
    ]
    if any(k in low for k in explicit_markers) and speaker_role == "wearer":
        return Observability.EXTRACTED

    # Generic plan markers without wearer confirmation → INFERRED
    if any(k in low for k in ["will", "should", "need to", "have to", "going to"]):
        return Observability.INFERRED

    # High ASR confidence + explicit keywords → INFERRED
    if confidence > 0.7 and any(k in low for k in ["call", "meet", "buy", "send", "go to"]):
        return Observability.INFERRED

    # Acoustic evidence with matching text → INFERRED
    if events and confidence > 0.5:
        if "phone_ringing" in events and any(k in low for k in ["hello", "call", "phone"]):
            return Observability.INFERRED
        if "conversation" in events and len(low) > 15:
            return Observability.INFERRED

    # Scene context with matching keywords → INFERRED
    if scene_label == "meeting_room" and any(k in low for k in ["discuss", "meet", "present"]):
        return Observability.INFERRED
    if scene_label == "kitchen" and any(k in low for k in ["cook", "eat", "food"]):
        return Observability.INFERRED

    # Moderate evidence → AMBIGUOUS
    return Observability.AMBIGUOUS


def classify_observability_llm(
    text: str,
    confidence: float,
    speaker_role: str = "unknown",
    scene_label: str = "other",
    event_tags: list | None = None,
) -> Observability:
    import json
    from longai.utils.llm import generate

    events = event_tags or []
    prompt = f"""Classify the observability of this statement for personal intent memory.

- EXTRACTED: the wearer explicitly stated their own intent (e.g. "I need to call John")
- INFERRED: plausible conclusion from context, sound events, or scene — NOT explicitly stated
- AMBIGUOUS: unclear, unreliable, or insufficient evidence

Context:
  Speaker role: {speaker_role}
  Scene: {scene_label}
  Events: {', '.join(events) if events else 'none'}
  Transcript: "{text}"
  ASR confidence: {confidence}

Return ONLY one word: EXTRACTED, INFERRED, or AMBIGUOUS."""
    try:
        result = generate(prompt, max_new_tokens=32).strip().upper()
        for obs in Observability:
            if obs.value in result:
                return obs
        # Try matching just the first word
        first_word = result.split()[0] if result else ""
        for obs in Observability:
            if obs.value == first_word:
                return obs
    except Exception as e:
        logger.warning("LLM observability classification failed: %s, falling back to rule", e)

    return classify_observability_rule(text, confidence, speaker_role, scene_label, event_tags)


def classify_observability(
    text: str,
    confidence: float,
    speaker_role: str = "unknown",
    scene_label: str = "other",
    event_tags: list | None = None,
    backend: str = "mock",
) -> Observability:
    events = event_tags or []
    if backend == "local_hf":
        return classify_observability_llm(text, confidence, speaker_role, scene_label, event_tags)
    return classify_observability_rule(text, confidence, speaker_role, scene_label, events)
