"""FileSystem Specialist Agent."""

from agents.base import BaseAgent
from core.guardrails import GuardrailManager
from llm.router import LLMRouter
from memory.graph_store import GraphMemory
from memory.vector_store import VectorMemory
from tools.registry import ToolRegistry


class FileSystemAgent(BaseAgent):

    def __init__(
        self,
        router: LLMRouter,
        vector: VectorMemory,
        graph: GraphMemory,
        guardrails: GuardrailManager,
        tools: ToolRegistry,
    ):
        system_prompt = (
            "You are the FileSystem agent in Peter's harness.\n"
            "Mission: Perform file and directory operations strictly within safe security boundaries.\n"
            "Constraints:\n"
            "- Default read scope is D:\\ (plus any allowlisted paths).\n"
            "- Default write scope is strictly D:\\Peter_Workspace\\ (or configured workspace).\n"
            "- Never access sensitive files (.env, .ssh, browser data, credentials).\n"
            "- Prevent path traversal and symlink escapes."
        )

        super().__init__(
            agent_id="filesystem",
            name="FileSystem",
            system_prompt=system_prompt,
            policy=["nvidia", "groq", "openrouter", "huggingface"],
            tool_names=["list_directory", "read_file", "write_file", "create_directory", "search_files", "memory_store"],
            max_steps=20,
            router=router,
            vector=vector,
            graph=graph,
            guardrails=guardrails,
            tools=tools,
        )
