"""Unit tests for GraphMemory operations."""

from memory.graph_store import GraphMemory


def test_graph_store_linking(graph_memory: GraphMemory):
    t_id = graph_memory.upsert_node(node_id="task_101", kind="task", label="Research LLM APIs")
    a_id = graph_memory.upsert_node(node_id="researcher", kind="agent", label="Researcher")

    edge_id = graph_memory.link(src_id=t_id, dst_id=a_id, relation="spawned_by")
    assert edge_id is not None

    nbrs = graph_memory.neighbors(node_id="task_101", max_depth=1)
    assert len(nbrs) > 0
    assert nbrs[0]["relation"] == "spawned_by"
    assert nbrs[0]["node"]["id"] == "researcher"
