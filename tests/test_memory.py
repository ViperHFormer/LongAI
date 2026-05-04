from longai.memory.graph_store import MemoryGraphStore
from longai.schema.models import GraphWriteOperation, Observability, OperationKind


def test_graph_store_create_node():
    store = MemoryGraphStore()
    op = GraphWriteOperation(
        kind=OperationKind.CREATE_NODE,
        session_id="s",
        segment_id="seg",
        timestamp=1.0,
        payload={"node_id": "n1", "node_type": "Intent", "label": "meeting", "attributes": {}},
        confidence=0.5,
        observability=Observability.EXTRACTED,
    )
    store.apply_operation(op)
    assert "n1" in store.graph
