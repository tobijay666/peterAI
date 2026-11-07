"""Abstract Base Class for all agents in the fleet."""

import json
import uuid
from abc import ABC
from typing import Any, Dict, List, Optional
from core.guardrails import GuardrailManager
from core.interfaces import AgentState, Message
from core.events import emit
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
        self._user_id: str = "local"
        self._context_builder: object = None  # injected by bootstrap_for_user

    def run(self, user_input: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Executes the agent tool reasoning loop up to max_steps."""
        self.state = AgentState.RUNNING
        _turn_id = str(uuid.uuid4())
        emit("turn_start", user_id=self._user_id, turn_id=_turn_id, agent=self.agent_id)
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

        # Phase 3: inject recalled context (memory + skill hints) before planning
        if self._context_builder is not None:
            try:
                from memory.recall import ContextBuilder  # type: ignore
                rc = self._context_builder.build(user_input, namespace=self.agent_id, turn_id=_turn_id)

                # Cognitive memory (Stages 2–3): a knowledge gap triggers research
                # before the LLM loop. Sync mode researches now and re-reads memory;
                # async mode tells the model research is under way.
                gap_note = self._handle_knowledge_gap(user_input, rc, _turn_id)
                if gap_note == "researched":
                    rc = self._context_builder.build(user_input, namespace=self.agent_id,
                                                     turn_id=_turn_id, count_access=False)
                    gap_note = None

                if not rc.is_empty():
                    messages.append(
                        Message(
                            role="system",
                            content=f"[CONTEXT FROM MEMORY]\n{rc.memory_text}",
                        )
                    )
                    if "[KNOWLEDGE" in rc.memory_text:
                        messages.append(Message(role="system", content=(
                            "When you use [KNOWLEDGE] items, cite them inline as [n] and end with a short "
                            "'Sources:' list of the URLs you cited. Do not cite items you did not use.")))
                    emit("context_injected", user_id=self._user_id, turn_id=_turn_id,
                         agent=self.agent_id, memory_ids=rc.memory_ids,
                         skill_hint=rc.skill_hint)
                if gap_note:
                    messages.append(Message(role="system", content=gap_note))
            except Exception as _cb_err:
                logger.debug(f"ContextBuilder skipped: {_cb_err}")

        tool_schemas = self.tools.get_schemas(self.tool_names) if self.tool_names else None
        step_count = 0

        while step_count < self.max_steps:
            step_count += 1
            logger.debug(f"Agent [{self.name}] step {step_count}/{self.max_steps}")

            emit("llm_call", user_id=self._user_id, turn_id=_turn_id, agent=self.agent_id, step=step_count)
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
                emit("turn_end", user_id=self._user_id, turn_id=_turn_id, agent=self.agent_id)
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
                emit("tool_call", user_id=self._user_id, turn_id=_turn_id, agent=self.agent_id, tool=tool_name)
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

    def _handle_knowledge_gap(self, user_input: str, rc: Any, turn_id: str) -> Optional[str]:
        """Peter only: research missing knowledge. Returns "researched", a note for the model, or None."""
        cognition = getattr(self._context_builder, "cognition", None)
        verdict = getattr(rc, "verdict", None)
        if self.agent_id != "peter" or cognition is None or verdict is None or verdict.sufficient:
            return None
        try:
            outcome = cognition.handle_gap(user_input, rc.graph_result, verdict, turn_id=turn_id)
        except Exception as e:
            logger.warning(f"Knowledge-gap handling failed: {e}")
            return None
        if outcome.mode == "sync" and outcome.records > 0:
            return "researched"
        if outcome.mode == "async":
            topics = ", ".join(outcome.topics)
            return (f"[RESEARCH IN PROGRESS] Research on '{topics}' is active in the background. "
                    "Answer the user's question directly, comprehensively, and accurately using what you know. "
                    "Do NOT tell the user that you are looking into it or will follow up with an answer shortly.")
        return None

