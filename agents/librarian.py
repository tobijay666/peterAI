"""Librarian Specialist Agent."""

from agents.base import BaseAgent
from core.guardrails import GuardrailManager
from llm.router import LLMRouter
from memory.graph_store import GraphMemory
from memory.vector_store import VectorMemory
from tools.registry import ToolRegistry


class LibrarianAgent(BaseAgent):

    def __init__(
        self,
        router: LLMRouter,
        vector: VectorMemory,
        graph: GraphMemory,
        guardrails: GuardrailManager,
        tools: ToolRegistry,
    ):
        system_prompt = (
            "You are the Librarian agent in Peter's harness.\n"
            "Mission: Manage long-term vector memory and knowledge graph relationships.\n"
            "Constraints:\n"
            "- Recall, store, link, and prune memories across agent namespaces.\n"
            "- Never delete records marked pinned=True."
        )

        super().__init__(
            agent_id="librarian",
            name="Librarian",
            system_prompt=system_prompt,
            policy=["groq", "nvidia", "openrouter", "huggingface"],
            tool_names=[
                "memory_recall",
                "memory_store",
                "memory_prune",
                "memory_pin",
                "graph_link",
                "graph_query",
            ],
            max_steps=15,
            router=router,
            vector=vector,
            graph=graph,
            guardrails=guardrails,
            tools=tools,
        )
