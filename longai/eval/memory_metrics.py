from __future__ import annotations


def evaluate_memory_graph(pred_graph: dict, gt_graph: dict) -> dict:
    pred_nodes = {(x.get("node_type"), str(x.get("label", "")).lower()) for x in pred_graph.get("nodes", [])}
    gt_nodes = {(x.get("node_type"), str(x.get("label", "")).lower()) for x in gt_graph.get("nodes", [])}

    pred_edges = {
        (x.get("source"), x.get("target"), x.get("relation"))
        for x in pred_graph.get("edges", [])
    }
    gt_edges = {
        (x.get("source"), x.get("target"), x.get("relation"))
        for x in gt_graph.get("edges", [])
    }

    node_tp = len(pred_nodes & gt_nodes)
    node_fp = len(pred_nodes - gt_nodes)
    node_fn = len(gt_nodes - pred_nodes)

    edge_tp = len(pred_edges & gt_edges)
    edge_fp = len(pred_edges - gt_edges)
    edge_fn = len(gt_edges - pred_edges)

    def prf(tp: int, fp: int, fn: int) -> dict:
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        return {"precision": p, "recall": r, "f1": f1}

    return {
        "node": prf(node_tp, node_fp, node_fn),
        "edge": prf(edge_tp, edge_fp, edge_fn),
        "compactness_ratio": (len(pred_nodes) / len(gt_nodes)) if gt_nodes else 0.0,
    }
