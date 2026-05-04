from __future__ import annotations

from pathlib import Path

from longai.construction.mcoc import generate_write_operations
from longai.memory.graph_store import MemoryGraphStore
from longai.schema.models import EvidencePack
from longai.utils.io import read_jsonl


def build_memory_from_evidence(evidence_pack_path: Path, graph_out: Path, log_out: Path, backend: str = "mock") -> dict:
    rows = read_jsonl(evidence_pack_path)
    packs = [EvidencePack(**x) for x in sorted(rows, key=lambda y: y["start_time"])]
    store = MemoryGraphStore()
    for pack in packs:
        ops = generate_write_operations(pack, backend=backend)
        for op in ops:
            store.apply_operation(op)
    store.save(graph_out, log_out)
    return store.snapshot()
