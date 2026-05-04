from __future__ import annotations

import json
import logging
import re

from longai.schema.models import EvidencePack

logger = logging.getLogger(__name__)

INTENT_KEYWORDS = {
    "meeting": "attend meeting",
    "call": "make call",
    "schedule": "check schedule",
    "prepare": "prepare materials",
    "finish": "finish pending task",
    "buy": "buy item",
}

PLAN_MARKERS = [
    "need to", "have to", "should", "must", "going to",
    "plan to", "want to", "remember to", "will", "gonna",
    "pick up", "email", "send", "remind",
]

# Scene → plausible intents (used when text evidence is weak)
SCENE_INTENT_HINTS = {
    "meeting_room": [("attend meeting", 0.35), ("discuss topic", 0.30)],
    "office": [("work task", 0.30)],
    "kitchen": [("prepare food", 0.35)],
    "home": [("household task", 0.25)],
    "vehicle": [("travel", 0.35)],
    "cafe_shop": [("meet someone", 0.25), ("order food", 0.30)],
    "street_outdoor": [("travel", 0.30)],
}

# Event → plausible intents (acoustic inference)
EVENT_INTENT_HINTS = {
    "phone_ringing": [("answer call", 0.40), ("make call", 0.30)],
    "conversation": [("discuss topic", 0.25)],
    "typing": [("write message", 0.30), ("work task", 0.25)],
    "door": [("exit", 0.25), ("enter", 0.25)],
    "footsteps": [("walking", 0.20)],
    "kitchen_activity": [("prepare food", 0.35)],
    "traffic": [("travel", 0.30)],
    "appliance": [("household task", 0.20)],
    "crowd_chatter": [("social interaction", 0.25)],
}


# ── robust JSON extraction ─────────────────────────────────────────────

def _extract_json(text: str) -> dict | None:
    """Robust JSON extraction: strip markdown fences, find first valid JSON object."""
    # Strip markdown code fences
    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```\s*$", "", cleaned, flags=re.IGNORECASE)

    # Find first JSON object
    start = cleaned.find("{")
    end = cleaned.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(cleaned[start:end])
        except json.JSONDecodeError:
            pass

    # Try to find JSON anywhere in text
    for match in re.finditer(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text):
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            continue

    return None


def _safe_get(obj: dict, key: str, default: object = None) -> object:
    return obj.get(key, default)


# ── new unified result schema ──────────────────────────────────────────

def _empty_result(segment_id: str) -> dict:
    return {
        "episode": {
            "label": f"Episode {segment_id}",
            "summary": "",
        },
        "intents": [],
        "entities": [],
        "abstain": False,
        "abstain_reason": None,
        "time_expressions": [],
    }


# ── LLM-based spotting ─────────────────────────────────────────────────

_LLM_SYSTEM = """You are a personal intent memory extractor for a wearable AI assistant.
Your task: analyze a speech segment with its multimodal context and extract structured memory.

OBSERVABILITY GUIDE:
- EXTRACTED: the wearer or speaker explicitly states their intent (e.g. "I need to call John")
- INFERRED: intent is plausible from sound events, scene, or context, NOT explicitly stated
- AMBIGUOUS: evidence is insufficient or multiple interpretations are reasonable

INTENT THRESHOLD:
- Only create an intent when there is evidence of action, plan, need, task, state change,
  or future behavior. Do NOT create an intent for every segment.
- If the transcript is empty but scene/events strongly suggest an activity, you may
  create an INFERRED or AMBIGUOUS intent with low confidence (0.25-0.45).
- Do not confuse entity mentions with intents. "John" is an entity, not an intent.
- If you are unsure, set abstain=true and provide abstain_reason.

ENTITY TYPES: person, place, object, organization, time, other
ENTITY ROLES (within an intent): target, location, object, participant, time, other
RELATIONS (within an intent): realized_by (intent→episode), involves, target, location,
  participant, time_anchor

OUTPUT FORMAT: Return ONLY a valid JSON object. No markdown, no explanation text."""


def _build_llm_prompt(pack: EvidencePack) -> str:
    ctx_str = ""
    if pack.context_window_refs:
        ctx_str = ", ".join(
            f"{c.get('modality', '?')}:{c.get('payload', {}).get('label', '')}"
            for c in pack.context_window_refs[:3]
        )

    return f"""Analyze this speech segment:

  segment_id: {pack.segment_id}
  session_id: {pack.session_id}
  time: {pack.start_time:.1f}s - {pack.end_time:.1f}s (duration {pack.end_time - pack.start_time:.1f}s)
  speaker_role: {pack.speaker_role.value}
  asr_text: "{pack.asr_text}"
  scene_label: {pack.scene_label}
  event_tags: {', '.join(pack.event_tags) if pack.event_tags else 'none'}
  tool_confidences: {json.dumps(pack.tool_confidences)}
  context: {ctx_str if ctx_str else 'none'}

Return a JSON object with these fields:
{{
  "episode": {{
    "label": "short episode label",
    "summary": "one-sentence summary of what happened",
    "scene_label": "{pack.scene_label}",
    "event_tags": {json.dumps(pack.event_tags) if pack.event_tags else '[]'},
    "speaker_role": "{pack.speaker_role.value}"
  }},
  "intents": [
    {{
      "label": "canonical intent label (short noun phrase)",
      "description": "short description of the intent",
      "state": "planned",
      "observability": "EXTRACTED|INFERRED|AMBIGUOUS",
      "confidence": 0.0,
      "evidence_rationale": "brief justification",
      "entities": [
        {{"label": "entity name", "type": "person|place|object|organization|time|other", "role": "target|location|object|participant|time|other"}}
      ],
      "relations": [
        {{"source": "intent", "target": "episode", "relation": "realized_by"}}
      ]
    }}
  ],
  "entities": [
    {{"label": "entity name", "type": "person|place|object|organization|time|other", "confidence": 0.0}}
  ],
  "time_expressions": ["list of time-related phrases found"],
  "abstain": false,
  "abstain_reason": null
}}

If you truly cannot find any intent or entity, set abstain=true and provide abstain_reason.
Otherwise abstain must be false."""


def spot_candidates_llm(pack: EvidencePack) -> dict:
    from longai.utils.llm import generate

    prompt = _LLM_SYSTEM + "\n\n" + _build_llm_prompt(pack)

    try:
        text = generate(prompt, max_new_tokens=512)
        result = _extract_json(text)
        if result is None:
            logger.warning("LLM returned unparseable JSON for %s, falling back to rule", pack.segment_id)
            return spot_candidates_rule(pack)

        # Validate and fill defaults
        intents = result.get("intents", [])
        entities = result.get("entities", [])

        # Validate each intent
        for intent in intents:
            intent.setdefault("label", "unknown intent")
            intent.setdefault("description", "")
            intent.setdefault("state", "planned")
            intent.setdefault("confidence", 0.5)
            intent.setdefault("evidence_rationale", "")
            intent.setdefault("entities", [])
            intent.setdefault("relations", [])

            # Validate observability enum
            obs = intent.get("observability", "AMBIGUOUS")
            if obs not in ("EXTRACTED", "INFERRED", "AMBIGUOUS"):
                intent["observability"] = "AMBIGUOUS"

            # Validate state enum
            state = intent.get("state", "planned")
            if state not in ("tentative", "planned", "ongoing", "done", "canceled", "dropped"):
                intent["state"] = "planned"

            # Auto-add realized_by edge if missing
            has_realized_by = any(
                r.get("relation") == "realized_by" for r in intent["relations"]
            )
            if not has_realized_by:
                intent["relations"].append({
                    "source": "intent",
                    "target": "episode",
                    "relation": "realized_by",
                })

            # Validate entities within intent
            for ent in intent.get("entities", []):
                ent.setdefault("label", "unknown")
                if ent.get("type") not in ("person", "place", "object", "organization", "time", "other"):
                    ent["type"] = "other"
                if ent.get("role") not in ("target", "location", "object", "participant", "time", "other"):
                    ent["role"] = "other"

        # Validate entities
        for ent in entities:
            ent.setdefault("label", "unknown")
            ent.setdefault("type", "other")
            ent.setdefault("confidence", 0.5)
            if ent.get("type") not in ("person", "place", "object", "organization", "time", "other"):
                ent["type"] = "other"

        return {
            "episode": result.get("episode", {
                "label": f"Episode {pack.segment_id}",
                "summary": pack.asr_text[:80] if pack.asr_text else "",
            }),
            "intents": intents,
            "entities": entities,
            "abstain": result.get("abstain", False),
            "abstain_reason": result.get("abstain_reason", None),
            "time_expressions": result.get("time_expressions", []),
        }

    except Exception as e:
        logger.warning("LLM candidate spotting failed for %s: %s, falling back to rule", pack.segment_id, e)
        return spot_candidates_rule(pack)


# ── Rule-based spotting (improved with scene/event/speaker) ────────────

def spot_candidates_rule(pack: EvidencePack) -> dict:
    text = pack.asr_text.lower().strip()
    scene = pack.scene_label or "other"
    events = pack.event_tags or []
    speaker = pack.speaker_role.value
    confidences = pack.tool_confidences or {}

    intents = []
    entities = []
    time_expressions = []

    # ── 1. Text-based intent detection ──
    has_explicit_plan = False
    for marker in PLAN_MARKERS:
        if marker in text:
            has_explicit_plan = True
            break

    for kw, intent_label in INTENT_KEYWORDS.items():
        if kw in text:
            intents.append(intent_label)

    # Broader plan detection from plan markers
    if has_explicit_plan and not intents:
        # Try to extract what the plan is about
        if "call" in text:
            intents.append("make call")
        elif "buy" in text or "pick up" in text:
            intents.append("buy item")
        elif "meet" in text or "meeting" in text:
            intents.append("attend meeting")
        elif "go" in text or "leave" in text:
            intents.append("go somewhere")
        elif "eat" in text or "cook" in text or "food" in text:
            intents.append("prepare food")
        elif "finish" in text or "done" in text:
            intents.append("finish pending task")
        elif "email" in text or "send" in text:
            intents.append("send message")
        elif "schedule" in text or "remind" in text:
            intents.append("check schedule")
        else:
            intents.append("general ongoing intent")

    # ── 2. Acoustic inference from event tags ──
    acoustic_intents = []
    for event in events:
        hints = EVENT_INTENT_HINTS.get(event, [])
        for hint_label, hint_conf in hints:
            acoustic_intents.append((hint_label, hint_conf, f"event:{event}"))

    # ── 3. Scene-based inference ──
    scene_intents = []
    hints = SCENE_INTENT_HINTS.get(scene, [])
    for hint_label, hint_conf in hints:
        scene_intents.append((hint_label, hint_conf, f"scene:{scene}"))

    # ── 4. Merge text + acoustic + scene intents ──
    # Use dict to deduplicate by label, keeping highest priority
    intent_map = {}  # label -> (label, evidence_source, confidence)
    for i in intents:
        intent_map[i] = (i, "text", 0.62)

    for label, conf, source in acoustic_intents:
        if label not in intent_map:
            intent_map[label] = (label, source, conf)
        elif intent_map[label][2] < conf:
            intent_map[label] = (label, source, conf)

    for label, conf, source in scene_intents:
        if label not in intent_map:
            intent_map[label] = (label, source, conf)

    # ── 5. Fallback to general if some text but no intent ──
    if not intent_map and len(text) > 6:
        intent_map["general ongoing intent"] = ("general ongoing intent", "text", 0.35)

    # ── 6. Don't generate intent for very short/empty text without event hints ──
    if not intent_map:
        return {
            "episode": {
                "label": f"Episode {pack.segment_id}",
                "summary": text[:80] if text else "",
                "scene_label": scene,
                "event_tags": events,
                "speaker_role": speaker,
            },
            "intents": [],
            "entities": [],
            "abstain": len(text) == 0,
            "abstain_reason": "no speech or evidence" if len(text) == 0 else None,
            "time_expressions": [],
        }

    # ── 7. Entity extraction and association ──
    # Entity patterns: (label, type, likely_role)
    person_names = [w for w in text.split() if w and w[0].isupper() and len(w) > 1]
    entity_keywords = {
        "person": ["teammate", "friend", "boss", "mom", "dad", "john", "mike",
                    "sarah", "david", "alex", "lisa", "emma", "james"],
        "place": ["office", "home", "meeting room", "kitchen", "store"],
        "object": ["lunch", "dinner", "email", "phone", "call", "project",
                    "report", "document", "file", "paper", "message", "coffee"],
        "time": ["tomorrow", "next week", "afternoon", "morning", "evening",
                  "today", "tonight", "deadline"],
        "organization": [],
    }

    extracted_entities = []  # (label, type, role)
    for etype, words in entity_keywords.items():
        for word in words:
            if word in text:
                # Determine role based on type
                role_map = {
                    "person": "target",
                    "place": "location",
                    "object": "object",
                    "time": "time",
                }
                extracted_entities.append((
                    word, etype, role_map.get(etype, "other"), 0.55
                ))

    # Extract capitalized person names
    for name in person_names:
        if name.lower() not in {e[0] for e in extracted_entities}:
            extracted_entities.append((name.lower(), "person", "target", 0.55))

    # Time expressions
    time_phrases = ["tomorrow", "next week", "later", "afternoon", "morning",
                    "evening", "today", "tonight", "next"]
    for phrase in time_phrases:
        if phrase in text:
            time_expressions.append(phrase)
            extracted_entities.append((phrase, "time", "time", 0.55))

    # ── 8. Build structured intents with observability ──
    # Primary intent (text-based) gets associated entities
    primary_intents = [i for i, (_, src, _) in intent_map.items() if src == "text"]
    secondary_intents = [i for i, (_, src, _) in intent_map.items() if src != "text"]

    structured_intents = []
    assigned_entities = set()

    for idx, label in enumerate(primary_intents + secondary_intents):
        canonical, evidence_source, confidence = intent_map[label]

        # Determine observability based on evidence source and speaker role
        if evidence_source == "text" and speaker == "wearer" and has_explicit_plan:
            observability = "EXTRACTED"
        elif evidence_source == "text":
            observability = "INFERRED"
        elif evidence_source.startswith("event:"):
            observability = "INFERRED"
            confidence = min(confidence, 0.45)
        elif evidence_source.startswith("scene:"):
            observability = "AMBIGUOUS"
            confidence = min(confidence, 0.35)
        else:
            observability = "AMBIGUOUS"
            confidence = min(confidence, 0.30)

        # Boost confidence if speaker is wearer
        if speaker == "wearer" and evidence_source == "text":
            confidence = min(confidence * 1.1, 0.85)

        # Associate entities: primary intent gets all entities found in text
        intent_entities = []
        if idx == 0:  # Primary intent gets the associated entities
            for ent_label, ent_type, ent_role, ent_conf in extracted_entities:
                intent_entities.append({
                    "label": ent_label,
                    "type": ent_type,
                    "role": ent_role,
                    "confidence": ent_conf,
                })
                assigned_entities.add(ent_label)
        elif idx < len(primary_intents):
            # Other text-based intents share remaining entities related to their keywords
            intent_keywords = canonical.split()
            for ent_label, ent_type, ent_role, ent_conf in extracted_entities:
                if ent_label not in assigned_entities and ent_label in intent_keywords:
                    intent_entities.append({
                        "label": ent_label,
                        "type": ent_type,
                        "role": ent_role,
                        "confidence": ent_conf,
                    })
                    assigned_entities.add(ent_label)

        structured_intents.append({
            "label": canonical,
            "description": f"Intent: {canonical}",
            "state": "planned",
            "observability": observability,
            "confidence": round(confidence, 3),
            "evidence_rationale": f"Source: {evidence_source}, speaker={speaker}, scene={scene}",
            "entities": intent_entities,
            "relations": [{"source": "intent", "target": "episode", "relation": "realized_by"}],
        })

    # Remaining unassigned entities as standalone
    for ent_label, ent_type, ent_role, ent_conf in extracted_entities:
        if ent_label not in assigned_entities:
            entities.append({
                "label": ent_label,
                "type": ent_type,
                "confidence": ent_conf,
            })

    return {
        "episode": {
            "label": f"Episode {pack.segment_id}",
            "summary": text[:80] if text else "",
            "scene_label": scene,
            "event_tags": events,
            "speaker_role": speaker,
        },
        "intents": structured_intents,
        "entities": entities,
        "abstain": False,
        "abstain_reason": None,
        "time_expressions": time_expressions,
    }


# ── Dispatcher ─────────────────────────────────────────────────────────

def spot_candidates(pack: EvidencePack, backend: str = "mock") -> dict:
    if backend == "local_hf":
        return spot_candidates_llm(pack)
    return spot_candidates_rule(pack)
