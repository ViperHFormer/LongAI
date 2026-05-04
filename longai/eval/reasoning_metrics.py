from __future__ import annotations


def _normalize(text: str) -> str:
    return " ".join(text.lower().strip().split())


def evaluate_reasoning(results: list[dict], gt_items: list[dict]) -> dict:
    gt_map = {x["query"]: x for x in gt_items}
    total = 0
    exact = 0
    abstain_correct = 0
    for r in results:
        q = r["query"]
        if q not in gt_map:
            continue
        total += 1
        gold = _normalize(gt_map[q].get("answer", ""))
        pred = _normalize(r.get("answer", ""))
        if pred == gold:
            exact += 1
        if r.get("abstained") and gt_map[q].get("allow_abstain", False):
            abstain_correct += 1

    return {
        "exact_match": exact / total if total else 0.0,
        "abstain_correct": abstain_correct / total if total else 0.0,
        "count": total,
    }
