"""Memory manipulation tools for vector and graph memory operations."""

from typing import Any, Dict, Optional
from core.interfaces import ToolResult
from memory.graph_store import GraphMemory
from memory.vector_store import VectorMemory

# Singleton instances initialized at runtime
_vector_instance: Optional[VectorMemory] = None
_graph_instance: Optional[GraphMemory] = None


def init_memory_tools(vector: VectorMemory, graph: GraphMemory):
    global _vector_instance, _graph_instance
    _vector_instance = vector
    _graph_instance = graph


def memory_recall(namespace: str, query: str, n_results: int = 5) -> ToolResult:
    if not _vector_instance:
        return ToolResult(success=False, output="", error="Vector memory not initialized.")

    results = _vector_instance.recall(namespace=namespace, query=query, n_results=n_results)
    if not results:
        return ToolResult(success=True, output=f"No matching memories found in namespace '{namespace}'.")

    formatted = []
    for r in results:
        meta = r["metadata"]
        score = r.get("score", 0.0)
        formatted.append(
            f"ID: {r['id']}\nText: {r['document']}\nKind: {meta.get('kind', 'finding')} | Score: {score:.2f} | Pinned: {meta.get('pinned', False)}\n"
        )
    return ToolResult(success=True, output="\n---\n".join(formatted))


def memory_store(
    namespace: str,
    text: str,
    importance: float = 0.5,
    pinned: bool = False,
    kind: str = "finding",
    tags: str = "",
) -> ToolResult:
    if not _vector_instance:
        return ToolResult(success=False, output="", error="Vector memory not initialized.")

    doc_id = _vector_instance.store(
        namespace=namespace,
        text=text,
        importance=importance,
        pinned=pinned,
        kind=kind,
        tags=tags,
    )
    return ToolResult(success=True, output=f"Stored memory record [{doc_id}] in namespace '{namespace}'.")


def memory_prune(namespace: str, doc_ids: list[str]) -> ToolResult:
    if not _vector_instance:
        return ToolResult(success=False, output="", error="Vector memory not initialized.")

    count = _vector_instance.prune(namespace=namespace, doc_ids=doc_ids)
    return ToolResult(success=True, output=f"Pruned {count} unpinned memory records from '{namespace}'.")


def memory_pin(namespace: str, doc_id: str, pinned: bool = True) -> ToolResult:
    if not _vector_instance:
        return ToolResult(success=False, output="", error="Vector memory not initialized.")

    ok = _vector_instance.pin(namespace=namespace, doc_id=doc_id, pinned=pinned)
    status = "pinned" if pinned else "unpinned"
    if ok:
        return ToolResult(success=True, output=f"Successfully {status} record [{doc_id}].")
    return ToolResult(success=False, output="", error=f"Could not find or pin record [{doc_id}].")


def graph_link(src_id: str, dst_id: str, relation: str) -> ToolResult:
    if not _graph_instance:
        return ToolResult(success=False, output="", error="Graph memory not initialized.")

    edge_id = _graph_instance.link(src_id=src_id, dst_id=dst_id, relation=relation)
    return ToolResult(
        success=True, output=f"Linked graph nodes: ({src_id}) -[{relation}]-> ({dst_id}) [edge_id: {edge_id}]"
    )


def graph_query(node_id: str, max_depth: int = 2) -> ToolResult:
    if not _graph_instance:
        return ToolResult(success=False, output="", error="Graph memory not initialized.")

    neighbors = _graph_instance.neighbors(node_id=node_id, max_depth=max_depth)
    if not neighbors:
        return ToolResult(success=True, output=f"No connected graph nodes found for '{node_id}'.")

    formatted = []
    for n in neighbors:
        formatted.append(
            f"Node: {n['node']['label']} ({n['node']['kind']}) | Edge: {n['src']} -> {n['relation']} -> {n['dst']}"
        )
    return ToolResult(success=True, output="\n".join(formatted))
