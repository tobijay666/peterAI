"""Voice Specialist Agent."""

from agents.base import BaseAgent
from core.guardrails import GuardrailManager
from llm.router import LLMRouter
from memory.graph_store import GraphMemory
from memory.vector_store import VectorMemory
from tools.registry import ToolRegistry


class VoiceAgent(BaseAgent):

    def __init__(
        self,
        router: LLMRouter,
        vector: VectorMemory,
        graph: GraphMemory,
        guardrails: GuardrailManager,
        tools: ToolRegistry,
    ):
        system_prompt = (
            "You are the Voice agent in Peter's harness.\n"
            "Mission: Handle text-to-speech audio outputs using non-blocking queues."
        )

        super().__init__(
            agent_id="voice",
            name="Voice",
            system_prompt=system_prompt,
            policy=["groq", "nvidia", "openrouter", "huggingface"],
            tool_names=["tts_speak"],
            max_steps=3,
            router=router,
            vector=vector,
            graph=graph,
            guardrails=guardrails,
            tools=tools,
        )
