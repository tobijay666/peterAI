"""Abstract Base Class for all agents in the fleet."""

import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from core.guardrails import GuardrailManager
from core.interfaces import AgentState, Message
from core.logger import logger
from llm.router import LLMRouter
from memory.graph_store import GraphMemory
from memory.vector_store import VectorMemory
from tools.registry import ToolRegistry


class BaseAgent(ABC):
    """Base class providing LLM tool loop execution, state management, and memory integration."""

    def __init__(
        self,
        agent_id: str,
        name: str,
        system_prompt: str,
        policy: List[str],
        tool_names: List[str],
        max_steps: int,
        router: LLMRouter,
        vector: VectorMemory,
        graph: GraphMemory,
        guardrails: GuardrailManager,
        tools: ToolRegistry,
    ):
        self.agent_id = agent_id
        self.name = name
        self.system_prompt = system_prompt
        self.policy = policy
        self.tool_names = tool_names
        self.max_steps = max_steps
        self.router = router
        self.vector = vector
        self.graph = graph
        self.guardrails = guardrails
        self.tools = tools
        self.state = AgentState.IDLE

    def run(self, user_input: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Executes the agent tool reasoning loop up to max_steps."""
        self.state = AgentState.RUNNING
        logger.info(f"Agent [{self.name}] starting turn with input: '{user_input[:60]}...'")

        # Check input guardrail
        gr_res = self.guardrails.check_input(user_input)
        if not gr_res.allowed:
            self.state = AgentState.ERROR
            err_msg = f"Agent [{self.name}] input blocked by guardrail [{gr_res.rule_id}]: {gr_res.message}"
            logger.warning(err_msg)
            return err_msg

        messages: List[Message] = [
            Message(role="system", content=self.system_prompt),
            Message(role="user", content=user_input),
        ]

        if context:
            ctx_str = json.dumps(context)
            messages.append(Message(role="system", content=f"Context information: {ctx_str}"))

        tool_schemas = self.tools.get_schemas(self.tool_names) if self.tool_names else None
        step_count = 0

        while step_count < self.max_steps:
            step_count += 1
            logger.debug(f"Agent [{self.name}] step {step_count}/{self.max_steps}")

            try:
                completion = self.router.complete(
                    messages=messages,
                    policy=self.policy,
                    tools=tool_schemas,
                )
            except Exception as e:
                self.state = AgentState.ERROR
                err_msg = f"Agent [{self.name}] routing error: {e}"
                logger.error(err_msg)
                return f"Error encountered by {self.name}: {e}"

            content = completion.get("content", "") or ""
            tool_calls = completion.get("tool_calls")

            # Append assistant turn to messages
            messages.append(
                Message(
                    role="assistant",
                    content=content,
                    tool_calls=tool_calls,
                )
            )

            # If no tool calls requested, agent has completed its reasoning
            if not tool_calls:
                self.state = AgentState.IDLE
                # Check output guardrail
                gr_out = self.guardrails.check_output(content)
                if not gr_out.allowed:
                    logger.warning(f"Agent [{self.name}] output blocked: {gr_out.message}")
                    return f"[Output Blocked by Guardrail: {gr_out.message}]"

                # Persist turn finding to agent's vector memory namespace
                self.vector.store(
                    namespace=self.agent_id,
                    text=f"User Query: {user_input}\nResponse: {content}",
                    kind="task",
                )
                return content

            # Handle tool calls
            for tool_call in tool_calls:
                fn_info = tool_call.get("function", {})
                tool_name = fn_info.get("name")
                raw_args = fn_info.get("arguments", "{}")
                tool_call_id = tool_call.get("id")

                try:
                    tool_args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except Exception:
                    tool_args = {}

                # Execute tool
                tool_res = self.tools.dispatch(name=tool_name, args=tool_args, caller_agent=self.agent_id)
                tool_output_str = tool_res.output if tool_res.success else f"Error: {tool_res.error}"

                # Append tool output message
                messages.append(
                    Message(
                        role="tool",
                        content=tool_output_str,
                        tool_call_id=tool_call_id,
                    )
                )

        self.state = AgentState.IDLE
        msg = f"Agent [{self.name}] reached maximum step limit ({self.max_steps})."
        logger.warning(msg)
        return f"{msg} Partial response: {messages[-1].content if messages else ''}"
