from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from longai.utils.io import read_jsonl, write_json


def _extract_json_block(text: str) -> dict[str, Any] | None:
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    raw = text[start : end + 1]
    try:
        return json.loads(raw)
    except Exception:
        return None


def _op_signature(op: dict[str, Any]) -> tuple:
    payload = op.get("payload", {})
    return (
        op.get("kind"),
        payload.get("node_id"),
        payload.get("node_type"),
        payload.get("label"),
        payload.get("source"),
        payload.get("target"),
        payload.get("relation"),
    )


def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f1}


def evaluate_writer_predictions(eval_rows: list[dict[str, Any]], pred_rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = 0
    exact = 0
    tp = fp = fn = 0

    for gt, pred in zip(eval_rows, pred_rows):
        total += 1
        gt_ops = gt.get("output", {}).get("operations", [])
        pred_ops = pred.get("operations", [])

        if gt_ops == pred_ops:
            exact += 1

        gt_set = {_op_signature(op) for op in gt_ops}
        pred_set = {_op_signature(op) for op in pred_ops}
        tp += len(gt_set & pred_set)
        fp += len(pred_set - gt_set)
        fn += len(gt_set - pred_set)

    out = {
        "count": total,
        "exact_match": exact / total if total else 0.0,
        "operation_prf": _prf(tp, fp, fn),
    }
    return out


def evaluate_writer_model(
    eval_jsonl: Path,
    output_json: Path,
    base_model: str,
    checkpoint_dir: Path,
    max_eval_samples: int | None = None,
    max_new_tokens: int = 384,
) -> dict[str, Any]:
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        summary = {
            "status": "error",
            "message": "Missing eval deps. Install transformers + peft + torch.",
            "error": str(exc),
        }
        write_json(output_json, summary)
        return summary

    rows = read_jsonl(eval_jsonl)
    if max_eval_samples is not None:
        rows = rows[: max(0, max_eval_samples)]

    if not rows:
        summary = {"status": "error", "message": f"No eval rows found in {eval_jsonl}"}
        write_json(output_json, summary)
        return summary

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(base_model, trust_remote_code=True, device_map="auto")
    model = PeftModel.from_pretrained(base, str(checkpoint_dir))
    model.eval()

    pred_rows: list[dict[str, Any]] = []
    for row in rows:
        prompt = row.get("prompt")
        if not prompt:
            inp = row.get("input", {})
            prompt = (
                "You are a LongAI MCoC construction writer. Return JSON with key operations.\n"
                f"EvidencePack:\n{inp.get('evidence_pack', {})}\n"
                f"LocalGraphContext:\n{inp.get('local_graph_context', {})}\n"
            )
        with torch.no_grad():
            tokens = tokenizer(prompt, return_tensors="pt").to(model.device)
            out = model.generate(**tokens, max_new_tokens=max_new_tokens, do_sample=False)
            text = tokenizer.decode(out[0], skip_special_tokens=True)
        parsed = _extract_json_block(text) or {"operations": []}
        pred_rows.append({"operations": parsed.get("operations", [])})

    metrics = evaluate_writer_predictions(rows, pred_rows)
    summary = {
        "status": "ok",
        "base_model": base_model,
        "checkpoint": str(checkpoint_dir),
        "metrics": metrics,
    }
    write_json(output_json, summary)
    return summary
