"""Integration test verifying complete system bootstrapping and component assembly."""

import pytest
from core.guardrails import GuardrailManager
from memory.vector_store import VectorMemory
from memory.graph_store import GraphMemory
from llm.router import LLMRouter
from tools.registry import ToolRegistry
from tools.memory_tools import init_memory_tools, memory_store, memory_recall
from agents.peter import PeterAgent
from agents.researcher import ResearcherAgent
from agents.filesystem import FileSystemAgent
from agents.librarian import LibrarianAgent
from agents.devops import DevOpsAgent
from agents.qa import QAAgent
from agents.voice import VoiceAgent


def test_system_bootstrap_and_integration(temp_dir):
    # 1. Guardrails
    gm = GuardrailManager("config/guardrails.json")
    assert len(gm.compiled_rules) > 0

    # 2. Vector & Graph Memory
    chroma_dir = temp_dir / "chroma"
    graph_db = temp_dir / "graph.db"
    vm = VectorMemory(persist_directory=str(chroma_dir))
    gm_store = GraphMemory(database_path=str(graph_db))
    init_memory_tools(vm, gm_store)

    assert len(vm.collections) == 7

    # 3. Memory store roundtrip
    store_res = memory_store(namespace="peter", text="Sir Anton system bootstrap integration test.", importance=0.9)
    assert store_res.success
    recall_res = memory_recall(namespace="peter", query="bootstrap integration test")
    assert recall_res.success
    assert "Sir Anton" in recall_res.output

    # 4. Graph node & edge creation
    t_id = gm_store.upsert_node(node_id="task_test", kind="task", label="Integration Check")
    a_id = gm_store.upsert_node(node_id="researcher", kind="agent", label="Researcher")
    edge_id = gm_store.link(src_id=t_id, dst_id=a_id, relation="spawned_by")
    assert edge_id is not None
    nbrs = gm_store.neighbors("task_test")
    assert len(nbrs) == 1
    assert nbrs[0]["relation"] == "spawned_by"

    # 5. Router initialization
    router = LLMRouter("config/providers.yaml")
    assert len(router.providers) == 4

    # 6. Tool Registry
    registry = ToolRegistry(guardrails=gm)
    assert registry is not None

    # 7. Agent Fleet Instantiation
    researcher = ResearcherAgent(router, vm, gm_store, gm, registry)
    filesystem = FileSystemAgent(router, vm, gm_store, gm, registry)
    librarian = LibrarianAgent(router, vm, gm_store, gm, registry)
    devops = DevOpsAgent(router, vm, gm_store, gm, registry)
    qa = QAAgent(router, vm, gm_store, gm, registry)
    voice = VoiceAgent(router, vm, gm_store, gm, registry)

    specialists = {
        "researcher": researcher,
        "filesystem": filesystem,
        "librarian": librarian,
        "devops": devops,
        "qa": qa,
        "voice": voice,
    }

    peter = PeterAgent(router, vm, gm_store, gm, registry, specialists=specialists)
    assert peter.name == "Peter"
    assert len(peter.specialists) == 6
