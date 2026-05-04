from __future__ import annotations

from pathlib import Path

from longai.utils.io import write_json


def write_annotation_templates(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    tool_level = {
        "session_id": "session_0000",
        "vad_regions": [{"start": 0.0, "end": 1.2}],
        "speaker_role": [{"start": 0.0, "end": 1.2, "role": "wearer"}],
        "asr": [{"start": 0.0, "end": 1.2, "text": "sample transcript"}],
        "scene": [{"start": 0.0, "end": 5.0, "label": "office"}],
        "event_tags": [{"start": 0.0, "end": 5.0, "tags": ["typing"]}],
    }
    graph_level = {
        "session_id": "session_0000",
        "nodes": [],
        "edges": [],
        "evidence": [],
        "timeline_updates": [],
    }
    reasoning_level = {
        "session_id": "session_0000",
        "qa_items": [
            {
                "query_id": "q1",
                "query": "What is pending next?",
                "answer": "Unknown",
                "type": "next_intent",
            }
        ],
        "next_intent_labels": [],
        "planning_refs": [],
    }
    write_json(root / "tool_level_template.json", tool_level)
    write_json(root / "graph_level_template.json", graph_level)
    write_json(root / "reasoning_level_template.json", reasoning_level)
