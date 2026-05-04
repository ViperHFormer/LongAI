from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from longai.utils.io import read_json, read_jsonl, write_jsonl


def build_writer_prompt(evidence_pack: dict[str, Any], local_graph_context: dict[str, Any]) -> str:
    lines = [
        "You are a LongAI MCoC construction writer.",
        "Given one EvidencePack and local graph context, output graph write operations as JSON.",
        "Only use operation kinds: CREATE_NODE, UPDATE_NODE, MERGE_NODE, CREATE_EDGE, UPDATE_STATE, ATTACH_EVIDENCE, ARCHIVE_NODE.",
        "",
        "EvidencePack:",
        str(evidence_pack),
        "",
        "LocalGraphContext:",
        str(local_graph_context),
        "",
        "Return JSON with key 'operations' as a list of operation objects.",
    ]
    return "\n".join(lines)


def build_writer_completion(operations: list[dict[str, Any]]) -> str:
    return '{"operations": ' + str(operations).replace("'", '"') + "}"


def _normalize_op(op: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": op.get("kind"),
        "payload": op.get("payload", {}),
        "confidence": op.get("confidence", 0.5),
        "observability": op.get("observability", "AMBIGUOUS"),
    }


def _ensure_text_fields(row: dict[str, Any]) -> dict[str, Any]:
    prompt = build_writer_prompt(row["input"]["evidence_pack"], row["input"].get("local_graph_context", {}))
    completion = build_writer_completion(row["output"]["operations"])
    row["prompt"] = prompt
    row["completion"] = completion
    row["text"] = f"### Instruction\n{prompt}\n\n### Response\n{completion}"
    return row


def convert_graph_gt_to_ops(gt_graph_path: Path, output_jsonl: Path) -> int:
    gt = read_json(gt_graph_path)
    rows = []
    session_id = gt.get("session_id", "unknown")
    for node in gt.get("nodes", []):
        row = {
            "session_id": session_id,
            "segment_id": "gt_segment_unknown",
            "input": {
                "evidence_pack": {
                    "segment_id": "gt_segment_unknown",
                    "session_id": session_id,
                    "asr_text": "TODO_from_gt",
                    "scene_label": "other",
                    "event_tags": [],
                    "speaker_role": "unknown",
                },
                "local_graph_context": {},
            },
            "output": {
                "operations": [
                    {
                        "kind": "CREATE_NODE",
                        "payload": {
                            "node_id": node.get("node_id", "TODO"),
                            "node_type": node.get("node_type", "Intent"),
                            "label": node.get("label", "TODO"),
                            "attributes": node.get("attributes", {}),
                        },
                        "confidence": node.get("confidence", 0.7),
                        "observability": node.get("observability", "EXTRACTED"),
                    }
                ]
            },
        }
        rows.append(_ensure_text_fields(row))

    for edge in gt.get("edges", []):
        row = {
            "session_id": session_id,
            "segment_id": "gt_segment_unknown",
            "input": {
                "evidence_pack": {
                    "segment_id": "gt_segment_unknown",
                    "session_id": session_id,
                    "asr_text": "TODO_from_gt",
                    "scene_label": "other",
                    "event_tags": [],
                    "speaker_role": "unknown",
                },
                "local_graph_context": {},
            },
            "output": {
                "operations": [
                    {
                        "kind": "CREATE_EDGE",
                        "payload": {
                            "source": edge.get("source"),
                            "target": edge.get("target"),
                            "relation": edge.get("relation", "related_to"),
                        },
                        "confidence": edge.get("confidence", 0.7),
                        "observability": edge.get("observability", "EXTRACTED"),
                    }
                ]
            },
        }
        rows.append(_ensure_text_fields(row))

    write_jsonl(output_jsonl, rows)
    return len(rows)


def convert_update_logs_to_writer_data(
    update_logs_dir: Path,
    evidence_pack_dir: Path,
    output_jsonl: Path,
    max_samples: int | None = None,
) -> int:
    rows: list[dict[str, Any]] = []
    log_files = sorted(update_logs_dir.glob("*.jsonl"))

    for log_file in log_files:
        session_id = log_file.stem
        ev_map = {
            x["segment_id"]: x
            for x in read_jsonl(evidence_pack_dir / f"{session_id}.jsonl")
            if "segment_id" in x
        }

        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for op in read_jsonl(log_file):
            grouped[op.get("segment_id", "unknown")].append(op)

        for segment_id, seg_ops in grouped.items():
            evidence = ev_map.get(
                segment_id,
                {
                    "segment_id": segment_id,
                    "session_id": session_id,
                    "asr_text": "",
                    "scene_label": "other",
                    "event_tags": [],
                    "speaker_role": "unknown",
                },
            )
            out_ops = [_normalize_op(op) for op in seg_ops]
            row = {
                "session_id": session_id,
                "segment_id": segment_id,
                "input": {
                    "evidence_pack": evidence,
                    "local_graph_context": {
                        "recent_ops": [
                            {
                                "kind": op.get("kind"),
                                "payload": op.get("payload", {}),
                            }
                            for op in seg_ops[:2]
                        ]
                    },
                },
                "output": {"operations": out_ops},
            }
            rows.append(_ensure_text_fields(row))
            if max_samples is not None and len(rows) >= max_samples:
                break
        if max_samples is not None and len(rows) >= max_samples:
            break

    write_jsonl(output_jsonl, rows)
    return len(rows)


def split_writer_data(
    input_jsonl: Path,
    train_jsonl: Path,
    eval_jsonl: Path,
    eval_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[int, int]:
    rows = read_jsonl(input_jsonl)
    if not rows:
        write_jsonl(train_jsonl, [])
        write_jsonl(eval_jsonl, [])
        return 0, 0

    rng = random.Random(seed)
    rng.shuffle(rows)

    eval_size = max(1, int(len(rows) * eval_ratio)) if len(rows) > 3 else 1
    eval_rows = rows[:eval_size]
    train_rows = rows[eval_size:]
    if not train_rows:
        train_rows = eval_rows[:]

    write_jsonl(train_jsonl, train_rows)
    write_jsonl(eval_jsonl, eval_rows)
    return len(train_rows), len(eval_rows)
