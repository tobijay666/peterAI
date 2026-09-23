"""QA Specialist Agent."""

from agents.base import BaseAgent
from core.guardrails import GuardrailManager
from llm.router import LLMRouter
from memory.graph_store import GraphMemory
from memory.vector_store import VectorMemory
from tools.registry import ToolRegistry


class QAAgent(BaseAgent):

    def __init__(
        self,
        router: LLMRouter,
        vector: VectorMemory,
        graph: GraphMemory,
        guardrails: GuardrailManager,
        tools: ToolRegistry,
    ):
        system_prompt = (
            "You are the QA agent in Peter's harness.\n"
            "Mission: Author and execute unit, integration, and property-based tests.\n"
            "Constraints:\n"
            "- Ensure code reliability and guardrail correctness.\n"
            "- Report structured test pass/fail results with failure diffs."
        )

        super().__init__(
            agent_id="qa",
            name="QA",
            system_prompt=system_prompt,
            policy=["openrouter", "nvidia", "groq", "huggingface"],
            tool_names=["read_file", "write_file", "run_pytest", "run_hypothesis", "list_directory"],
            max_steps=25,
            router=router,
            vector=vector,
            graph=graph,
            guardrails=guardrails,
            tools=tools,
        )
