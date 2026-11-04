"""Memory manipulation tools for vector and graph memory operations.

Tools are bound to ONE user's stores: ``MemoryTools(vector, graph)`` holds that
user's VectorMemory / GraphMemory / SkillStore and exposes the tool functions as
bound methods, which are what the per-user ToolRegistry registers. There is no
module-level state, so two users' Peters can never write into each other's
memory (audit item 2: the old module globals were re-pointed at whichever user
bootstrapped last).

Writes into a knowledge collection (``k_*``) go through the KnowledgeStore so the
record is bound to its node, the registry count stays right and vector links are
maintained; ``graph_link`` validates the relation (audit item 20).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from core.interfaces import ToolResult
from memory.graph_store import VALID_RELATIONS, GraphMemory
from memory.skills import SkillStore
from memory.vector_store import VectorMemory

# Relations an agent may draw by hand. Structure (has_subject), plumbing (helper_of,
# spawned_by) and storage bookkeeping (has_collection) are owned by the memory code.
AGENT_LINKABLE = VALID_RELATIONS - {"has_subject", "helper_of", "spawned_by", "has_collection"}


class MemoryTools:
    """The memory tools of one user (see module docstring)."""

    def __init__(self, vector: VectorMemory, graph: GraphMemory, skill_store: Optional[SkillStore] = None,
                 knowledge: Any = None) -> None:
        self.vector = vector
        self.graph = graph
        self.skill_store = skill_store if skill_store is not None else SkillStore(graph=graph)
        self._knowledge = knowledge

    # The user's KnowledgeStore (CognitiveMemory attaches its own; a plain one otherwise)
    @property
    def knowledge(self) -> Any:
        if self._knowledge is None:
            from memory.knowledge import KnowledgeStore
            self._knowledge = KnowledgeStore(self.vector, self.graph)
        return self._knowledge

    @knowledge.setter
    def knowledge(self, value: Any) -> None:
        self._knowledge = value

    def _knowledge_node(self, namespace: str) -> Optional[Dict[str, Any]]:
        coll = self.graph.get_collection(namespace)
        if not coll or not coll.get("node_id"):
            return None
        return self.graph.get_node(coll["node_id"])

    # ------------------------------------------------------------------ vector tools

    def memory_recall(self, namespace: str, query: str, n_results: int = 5) -> ToolResult:
        results = self.vector.recall(namespace=namespace, query=query, n_results=n_results)
        if not results:
            return ToolResult(success=True, output=f"No matching memories found in namespace '{namespace}'.")
        formatted = []
        for r in results:
            meta = r["metadata"]
            score = r.get("score", 0.0)
            formatted.append(
                f"ID: {r['id']}\nText: {r['document']}\nKind: {meta.get('kind', 'finding')} | Score: {score:.2f} | "
                f"Pinned: {meta.get('pinned', False)}\n"
            )
        return ToolResult(success=True, output="\n---\n".join(formatted))

    def memory_store(self, namespace: str, text: str, importance: float = 0.5, pinned: bool = False,
                     kind: str = "finding", tags: str = "") -> ToolResult:
        if namespace == VectorMemory.INDEX_NAMESPACE:
            return ToolResult(success=False, output="", error="The entity index is maintained by Peter's memory.")
        if VectorMemory.is_knowledge_namespace(namespace):
            node = self._knowledge_node(namespace)
            if node is None:
                return ToolResult(success=False, output="",
                                  error=f"'{namespace}' is not a knowledge collection of any topic.")
            doc_id = self.knowledge.add_record(node, text, {"kind": kind, "tags": tags, "trust": "agent_note"},
                                               importance=importance)
            if pinned:
                self.vector.pin(namespace, doc_id, True)
            self.knowledge.refresh_stats(node)
            return ToolResult(success=True, output=f"Stored memory record [{doc_id}] under topic '{node['label']}'.")
        doc_id = self.vector.store(namespace=namespace, text=text, importance=importance, pinned=pinned,
                                   kind=kind, tags=tags)
        return ToolResult(success=True, output=f"Stored memory record [{doc_id}] in namespace '{namespace}'.")

    def memory_prune(self, namespace: str, doc_ids: List[str]) -> ToolResult:
        # VectorMemory notifies the KnowledgeStore, which drops copies / vector links and recounts
        count = self.vector.prune(namespace=namespace, doc_ids=doc_ids)
        return ToolResult(success=True, output=f"Pruned {count} unpinned memory records from '{namespace}'.")

    def memory_pin(self, namespace: str, doc_id: str, pinned: bool = True) -> ToolResult:
        ok = self.vector.pin(namespace=namespace, doc_id=doc_id, pinned=pinned)
        status = "pinned" if pinned else "unpinned"
        if ok:
            return ToolResult(success=True, output=f"Successfully {status} record [{doc_id}].")
        return ToolResult(success=False, output="", error=f"Could not find or pin record [{doc_id}].")

    # ------------------------------------------------------------------ graph tools

    def graph_link(self, src_id: str, dst_id: str, relation: str) -> ToolResult:
        if relation not in AGENT_LINKABLE:
            return ToolResult(success=False, output="", error=f"Relation '{relation}' cannot be linked by a tool.")
        if src_id == dst_id:
            return ToolResult(success=False, output="", error="A node cannot be linked to itself.")
        for nid in (src_id, dst_id):
            if self.graph.get_node(nid) is None:
                return ToolResult(success=False, output="", error=f"Unknown graph node '{nid}'.")
        edge_id, created, why = self.knowledge.link_checked(src_id, dst_id, relation)
        if edge_id is None:
            return ToolResult(success=False, output="", error=f"Link refused: {why}.")
        verb = "Linked" if created else "Already linked"
        return ToolResult(success=True,
                          output=f"{verb} graph nodes: ({src_id}) -[{relation}]-> ({dst_id}) [edge_id: {edge_id}]")

    def graph_query(self, node_id: str, max_depth: int = 2) -> ToolResult:
        neighbors = self.graph.neighbors(node_id=node_id, max_depth=max_depth)
        if not neighbors:
            return ToolResult(success=True, output=f"No connected graph nodes found for '{node_id}'.")
        formatted = []
        for n in neighbors:
            formatted.append(
                f"Node: {n['node']['label']} ({n['node']['kind']}) | Edge: {n['src']} -> {n['relation']} -> {n['dst']}"
            )
        return ToolResult(success=True, output="\n".join(formatted))

    # ------------------------------------------------------------------ skill tools

    def skill_list(self, status: str = "active") -> ToolResult:
        """List skills by status (active/candidate/deprecated)."""
        skills = self.skill_store.list_skills(status=status)
        if not skills:
            return ToolResult(success=True, output=f"No {status} skills found.")
        return ToolResult(success=True, output="\n".join(f"[{s.get('skill_id', '')}] {s.get('trigger', '')}"
                                                         for s in skills))

    def skill_get(self, skill_id: str) -> ToolResult:
        """Get details of a specific skill."""
        skill = self.skill_store.get_skill(skill_id)
        if not skill:
            return ToolResult(success=False, output="", error=f"Skill not found: {skill_id}")
        return ToolResult(success=True, output=json.dumps(skill, indent=2))

    def skill_deprecate(self, skill_id: str, reason: str = "", _caller_agent: str = "") -> ToolResult:
        """Deprecate a skill. Only callable by the librarian agent."""
        if _caller_agent != "librarian":
            return ToolResult(success=False, output="", error="Permission denied. Only librarian can deprecate skills.")
        ok = self.skill_store._set_status(skill_id, None, "deprecated")
        if ok:
            return ToolResult(success=True, output=f"Skill {skill_id} deprecated successfully. Reason: {reason}")
        return ToolResult(success=False, output="", error=f"Failed to deprecate skill {skill_id}.")


def init_memory_tools(vector: VectorMemory, graph: GraphMemory,
                      skill_store: Optional[SkillStore] = None, knowledge: Any = None) -> MemoryTools:
    """Build the memory tools bound to one user's stores (no global state is kept)."""
    return MemoryTools(vector, graph, skill_store=skill_store, knowledge=knowledge)


# Tool schemas, shared by the CLI and the web bootstrap
TOOL_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "memory_recall": {
        "name": "memory_recall", "description": "Recalls memories from vector store by query.",
        "parameters": {"type": "object", "properties": {
            "namespace": {"type": "string"}, "query": {"type": "string"},
            "n_results": {"type": "integer", "default": 5}}, "required": ["namespace", "query"]}},
    "memory_store": {
        "name": "memory_store", "description": "Stores a finding or text in vector memory.",
        "parameters": {"type": "object", "properties": {
            "namespace": {"type": "string"}, "text": {"type": "string"},
            "importance": {"type": "number", "default": 0.5}, "pinned": {"type": "boolean", "default": False},
            "kind": {"type": "string", "default": "finding"}, "tags": {"type": "string", "default": ""}},
            "required": ["namespace", "text"]}},
    "memory_prune": {
        "name": "memory_prune", "description": "Prunes unpinned memories by document IDs.",
        "parameters": {"type": "object", "properties": {
            "namespace": {"type": "string"}, "doc_ids": {"type": "array", "items": {"type": "string"}}},
            "required": ["namespace", "doc_ids"]}},
    "memory_pin": {
        "name": "memory_pin", "description": "Pins or unpins a vector memory record.",
        "parameters": {"type": "object", "properties": {
            "namespace": {"type": "string"}, "doc_id": {"type": "string"},
            "pinned": {"type": "boolean", "default": True}}, "required": ["namespace", "doc_id"]}},
    "graph_link": {
        "name": "graph_link", "description": "Creates a relationship edge between graph nodes.",
        "parameters": {"type": "object", "properties": {
            "src_id": {"type": "string"}, "dst_id": {"type": "string"}, "relation": {"type": "string"}},
            "required": ["src_id", "dst_id", "relation"]}},
    "graph_query": {
        "name": "graph_query", "description": "Queries connected neighbors in graph memory.",
        "parameters": {"type": "object", "properties": {
            "node_id": {"type": "string"}, "max_depth": {"type": "integer", "default": 2}},
            "required": ["node_id"]}},
}


def register_memory_tools(registry: Any, tools: MemoryTools, names: Optional[List[str]] = None) -> None:
    """Register this user's bound memory tools in that user's ToolRegistry."""
    for name in names or list(TOOL_SCHEMAS):
        registry.register(name, getattr(tools, name), TOOL_SCHEMAS[name])
