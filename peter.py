#!/usr/bin/env python3
"""
Peter AI — Locally-Executing Multi-Agent Assistant Harness
Entry Point: python peter.py
"""

import sys
from pathlib import Path
from dotenv import load_dotenv

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

load_dotenv()

from agents.devops import DevOpsAgent
from agents.filesystem import FileSystemAgent
from agents.librarian import LibrarianAgent
from agents.peter import PeterAgent
from agents.qa import QAAgent
from agents.researcher import ResearcherAgent
from agents.voice import VoiceAgent
from core.guardrails import GuardrailManager
from core.interfaces import ToolResult
from core.logger import logger, setup_logger
from engine.loop import REPLEngine
from llm.router import LLMRouter
from memory.eviction import EvictionScheduler
from memory.graph_store import GraphMemory
from memory.vector_store import VectorMemory
from tools import fs_tools, git_tools, memory_tools, web_tools
from tools.memory_tools import init_memory_tools
from tools.registry import ToolRegistry
from tts.service import TTSService


def bootstrap_system():
    """Initializes system subsystems, memory stores, agents, and tool registries."""
    logger.info("Bootstrapping Peter AI System...")

    # 1. Guardrails
    guardrails = GuardrailManager("config/guardrails.json")

    # 2. Memory Stores
    vector_store = VectorMemory(persist_directory="data/chroma", embedding_model="all-MiniLM-L6-v2")
    graph_store = GraphMemory(database_path="data/peter_graph.db")
    init_memory_tools(vector_store, graph_store)

    # 3. LLM Router
    router = LLMRouter("config/providers.yaml")

    # 4. Voice Subsystem
    tts_service = TTSService()

    # 5. Tool Registry & Function Binding
    registry = ToolRegistry(guardrails=guardrails)

    # Register Filesystem tools
    registry.register(
        "list_directory",
        fs_tools.list_directory,
        {
            "name": "list_directory",
            "description": "Lists contents of a directory.",
            "parameters": {
                "type": "object",
                "properties": {"directory_path": {"type": "string", "default": "."}},
            },
        },
    )
    registry.register(
        "read_file",
        fs_tools.read_file,
        {
            "name": "read_file",
            "description": "Reads text content from a file.",
            "parameters": {
                "type": "object",
                "properties": {"file_path": {"type": "string"}},
                "required": ["file_path"],
            },
        },
    )
    registry.register(
        "write_file",
        fs_tools.write_file,
        {
            "name": "write_file",
            "description": "Writes text content to a file inside the workspace boundary.",
            "parameters": {
                "type": "object",
                "properties": {"file_path": {"type": "string"}, "content": {"type": "string"}},
                "required": ["file_path", "content"],
            },
        },
    )
    registry.register(
        "create_directory",
        fs_tools.create_directory,
        {
            "name": "create_directory",
            "description": "Creates a directory path.",
            "parameters": {
                "type": "object",
                "properties": {"directory_path": {"type": "string"}},
                "required": ["directory_path"],
            },
        },
    )
    registry.register(
        "search_files",
        fs_tools.search_files,
        {
            "name": "search_files",
            "description": "Searches for files matching a glob pattern.",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory_path": {"type": "string", "default": "."},
                    "pattern": {"type": "string", "default": "*"},
                },
            },
        },
    )

    # Register Web tools
    registry.register(
        "web_search",
        web_tools.web_search,
        {
            "name": "web_search",
            "description": "Searches the web using DuckDuckGo.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        },
    )
    registry.register(
        "fetch_url",
        web_tools.fetch_url,
        {
            "name": "fetch_url",
            "description": "Extracts text content from a web page URL.",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
        },
    )
    registry.register(
        "scholarly_search",
        web_tools.scholarly_search,
        {
            "name": "scholarly_search",
            "description": "Searches academic papers via arXiv, Semantic Scholar, and PubMed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        },
    )

    # Register Git tools
    registry.register(
        "git_status",
        git_tools.git_status,
        {"name": "git_status", "description": "Gets git repository status.", "parameters": {"type": "object"}},
    )
    registry.register(
        "git_commit",
        git_tools.git_commit,
        {
            "name": "git_commit",
            "description": "Commits git changes (gated by QA tests).",
            "parameters": {
                "type": "object",
                "properties": {"message": {"type": "string"}},
                "required": ["message"],
            },
        },
    )
    registry.register(
        "git_branch",
        git_tools.git_branch,
        {
            "name": "git_branch",
            "description": "Lists or creates git branches.",
            "parameters": {
                "type": "object",
                "properties": {"branch_name": {"type": "string"}},
            },
        },
    )
    registry.register(
        "git_revert",
        git_tools.git_revert,
        {
            "name": "git_revert",
            "description": "Reverts a commit by hash.",
            "parameters": {
                "type": "object",
                "properties": {"commit_hash": {"type": "string"}},
                "required": ["commit_hash"],
            },
        },
    )
    registry.register(
        "git_stash",
        git_tools.git_stash,
        {
            "name": "git_stash",
            "description": "Stashes or pops uncommitted git changes.",
            "parameters": {
                "type": "object",
                "properties": {"action": {"type": "string", "default": "save"}},
            },
        },
    )
    registry.register(
        "git_log",
        git_tools.git_log,
        {
            "name": "git_log",
            "description": "Displays recent git commit log.",
            "parameters": {
                "type": "object",
                "properties": {"max_count": {"type": "integer", "default": 5}},
            },
        },
    )
    registry.register(
        "run_pytest",
        git_tools.run_pytest,
        {
            "name": "run_pytest",
            "description": "Runs pytest test suite.",
            "parameters": {
                "type": "object",
                "properties": {"target_path": {"type": "string", "default": "tests"}},
            },
        },
    )
    registry.register(
        "run_hypothesis",
        git_tools.run_hypothesis,
        {
            "name": "run_hypothesis",
            "description": "Runs hypothesis property-based test suite.",
            "parameters": {
                "type": "object",
                "properties": {"target_path": {"type": "string", "default": "tests/property"}},
            },
        },
    )
    registry.register(
        "run_tests",
        git_tools.run_tests,
        {"name": "run_tests", "description": "Runs test suite for QA validation.", "parameters": {"type": "object"}},
    )

    # Register Memory tools
    registry.register(
        "memory_recall",
        memory_tools.memory_recall,
        {
            "name": "memory_recall",
            "description": "Recalls memories from vector store by query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "namespace": {"type": "string"},
                    "query": {"type": "string"},
                    "n_results": {"type": "integer", "default": 5},
                },
                "required": ["namespace", "query"],
            },
        },
    )
    registry.register(
        "memory_store",
        memory_tools.memory_store,
        {
            "name": "memory_store",
            "description": "Stores a finding or text in vector memory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "namespace": {"type": "string"},
                    "text": {"type": "string"},
                    "importance": {"type": "number", "default": 0.5},
                    "pinned": {"type": "boolean", "default": False},
                    "kind": {"type": "string", "default": "finding"},
                    "tags": {"type": "string", "default": ""},
                },
                "required": ["namespace", "text"],
            },
        },
    )
    registry.register(
        "memory_prune",
        memory_tools.memory_prune,
        {
            "name": "memory_prune",
            "description": "Prunes unpinned memories by document IDs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "namespace": {"type": "string"},
                    "doc_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["namespace", "doc_ids"],
            },
        },
    )
    registry.register(
        "memory_pin",
        memory_tools.memory_pin,
        {
            "name": "memory_pin",
            "description": "Pins or unpins a vector memory record.",
            "parameters": {
                "type": "object",
                "properties": {
                    "namespace": {"type": "string"},
                    "doc_id": {"type": "string"},
                    "pinned": {"type": "boolean", "default": True},
                },
                "required": ["namespace", "doc_id"],
            },
        },
    )
    registry.register(
        "graph_link",
        memory_tools.graph_link,
        {
            "name": "graph_link",
            "description": "Creates a relationship edge between graph nodes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "src_id": {"type": "string"},
                    "dst_id": {"type": "string"},
                    "relation": {"type": "string"},
                },
                "required": ["src_id", "dst_id", "relation"],
            },
        },
    )
    registry.register(
        "graph_query",
        memory_tools.graph_query,
        {
            "name": "graph_query",
            "description": "Queries connected neighbors in graph memory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "node_id": {"type": "string"},
                    "max_depth": {"type": "integer", "default": 2},
                },
                "required": ["node_id"],
            },
        },
    )

    # Register Voice tool
    def tts_speak(text: str) -> ToolResult:
        tts_service.speak(text)
        return ToolResult(success=True, output="Enqueued speech output.")

    registry.register(
        "tts_speak",
        tts_speak,
        {
            "name": "tts_speak",
            "description": "Speaks text output using TTS voice engine.",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        },
    )

    # 6. Instantiate Specialist Fleet
    specialists = {
        "researcher": ResearcherAgent(router, vector_store, graph_store, guardrails, registry),
        "filesystem": FileSystemAgent(router, vector_store, graph_store, guardrails, registry),
        "librarian": LibrarianAgent(router, vector_store, graph_store, guardrails, registry),
        "devops": DevOpsAgent(router, vector_store, graph_store, guardrails, registry),
        "qa": QAAgent(router, vector_store, graph_store, guardrails, registry),
        "voice": VoiceAgent(router, vector_store, graph_store, guardrails, registry),
    }

    # 7. Instantiate Peter Orchestrator
    peter = PeterAgent(router, vector_store, graph_store, guardrails, registry, specialists=specialists)

    # 8. Eviction Scheduler
    eviction_scheduler = EvictionScheduler(vector_store, graph_store)
    eviction_scheduler.start()

    logger.info("Peter AI bootstrap complete.")
    return peter, tts_service, guardrails, eviction_scheduler


def main():
    try:
        peter, tts_service, guardrails, eviction_scheduler = bootstrap_system()
        repl = REPLEngine(peter, tts_service, guardrails)
        repl.start()
    except Exception as e:
        logger.critical(f"Fatal system crash during bootstrap or REPL execution: {e}", exc_info=True)
        sys.exit(1)
    finally:
        logger.info("Cleaning up background workers...")


if __name__ == "__main__":
    main()
