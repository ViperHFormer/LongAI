from __future__ import annotations

from collections import Counter
from typing import Any

from jiwer import cer, wer


def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": p, "recall": r, "f1": f1}


def eval_asr(pred: list[dict], gt: list[dict]) -> dict[str, float]:
    pred_text = " ".join([x.get("text", "") for x in pred])
    gt_text = " ".join([x.get("text", "") for x in gt])
    return {"wer": wer(gt_text, pred_text), "cer": cer(gt_text, pred_text)}


def eval_speaker_role(pred: list[dict], gt: list[dict]) -> dict[str, Any]:
    gt_map = {x["segment_id"]: x["role"] for x in gt}
    labels = sorted(set(gt_map.values()) | {x["role"] for x in pred})
    conf = {label: Counter() for label in labels}
    tp = fp = fn = 0
    for row in pred:
        sid = row["segment_id"]
        if sid not in gt_map:
            continue
        pred_label = row["role"]
        true_label = gt_map[sid]
        conf[true_label][pred_label] += 1
        if pred_label == true_label:
            tp += 1
        else:
            fp += 1
            fn += 1
    return {"macro": _prf(tp, fp, fn), "confusion": {k: dict(v) for k, v in conf.items()}}


def eval_simple_label_accuracy(pred: list[dict], gt: list[dict], key: str = "label") -> dict[str, float]:
    gt_map = {x["segment_id"]: x[key] for x in gt}
    total = 0
    hit = 0
    for row in pred:
        sid = row["segment_id"]
        if sid in gt_map:
            total += 1
            if row.get(key) == gt_map[sid]:
                hit += 1
    acc = hit / total if total else 0.0
    return {"accuracy": acc}


def eval_vad(pred: list[dict], gt: list[dict]) -> dict[str, float]:
    # Segment overlap approximation for pilot phase.
    def overlap(a0: float, a1: float, b0: float, b1: float) -> float:
        return max(0.0, min(a1, b1) - max(a0, b0))

    tp = 0
    fp = max(0, len(pred) - len(gt))
    fn = max(0, len(gt) - len(pred))
    iou_sum = 0.0
    for p in pred:
        best = 0.0
        p0, p1 = p["start"], p["end"]
        for g in gt:
            inter = overlap(p0, p1, g["start"], g["end"])
            union = (p1 - p0) + (g["end"] - g["start"]) - inter
            iou = inter / union if union > 0 else 0.0
            best = max(best, iou)
        if best > 0.3:
            tp += 1
        iou_sum += best
    prf = _prf(tp, fp, fn)
    prf["segment_iou"] = iou_sum / len(pred) if pred else 0.0
    return prf
