"""DevOps Specialist Agent."""

from agents.base import BaseAgent
from core.guardrails import GuardrailManager
from llm.router import LLMRouter
from memory.graph_store import GraphMemory
from memory.vector_store import VectorMemory
from tools.registry import ToolRegistry


class DevOpsAgent(BaseAgent):

    def __init__(
        self,
        router: LLMRouter,
        vector: VectorMemory,
        graph: GraphMemory,
        guardrails: GuardrailManager,
        tools: ToolRegistry,
    ):
        system_prompt = (
            "You are the DevOps agent in Peter's harness.\n"
            "Mission: Handle git branches, commits, and project checkpoints.\n"
            "Constraints:\n"
            "- Refuse to commit unless the QA agent's test suite passes.\n"
            "- Use Conventional Commits format (feat, fix, docs, test, chore)."
        )

        super().__init__(
            agent_id="devops",
            name="DevOps",
            system_prompt=system_prompt,
            policy=["nvidia", "groq", "openrouter", "huggingface"],
            tool_names=["git_status", "git_commit", "git_branch", "git_log", "run_tests", "read_file", "write_file"],
            max_steps=20,
            router=router,
            vector=vector,
            graph=graph,
            guardrails=guardrails,
            tools=tools,
        )
