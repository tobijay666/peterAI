"""Researcher Specialist Agent."""

from agents.base import BaseAgent
from core.guardrails import GuardrailManager
from llm.router import LLMRouter
from memory.graph_store import GraphMemory
from memory.vector_store import VectorMemory
from tools.registry import ToolRegistry


class ResearcherAgent(BaseAgent):

    def __init__(
        self,
        router: LLMRouter,
        vector: VectorMemory,
        graph: GraphMemory,
        guardrails: GuardrailManager,
        tools: ToolRegistry,
    ):
        system_prompt = (
            "You are the Researcher agent in Peter's harness.\n"
            "Mission: Search the web and academic databases to find, extract, and verify information.\n"
            "Constraints:\n"
            "- Cross-reference at least 2 independent sources before making strong claims.\n"
            "- Reject content farms, SEO spam, and unverifiable domains.\n"
            "- Return verified citations with title, author, year, URL, and confidence score."
        )

        super().__init__(
            agent_id="researcher",
            name="Researcher",
            system_prompt=system_prompt,
            policy=["openrouter", "nvidia", "groq", "huggingface"],
            tool_names=["web_search", "fetch_url", "scholarly_search", "memory_store"],
            max_steps=12,
            router=router,
            vector=vector,
            graph=graph,
            guardrails=guardrails,
            tools=tools,
        )
