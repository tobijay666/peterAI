"""Central Tool Registry with schema auto-generation, guardrail checking, and memory auditing."""

from typing import Any, Callable, Dict, List, Optional
from core.guardrails import GuardrailManager
from core.interfaces import ToolResult
from core.logger import logger


class ToolRegistry:
    """Manages available tools, OpenAI function call schemas, and guardrailed execution."""

    def __init__(self, guardrails: GuardrailManager):
        self.guardrails = guardrails
        self.tools: Dict[str, Callable[..., ToolResult]] = {}
        self.schemas: Dict[str, Dict[str, Any]] = {}

    def register(self, name: str, fn: Callable[..., ToolResult], schema: Dict[str, Any]) -> None:
        """Registers a tool with its function implementation and OpenAI tool schema."""
        self.tools[name] = fn
        self.schemas[name] = schema
        logger.debug(f"ToolRegistry: Registered tool '{name}'")

    def get_schemas(self, tool_names: List[str]) -> List[Dict[str, Any]]:
        """Returns OpenAI-compatible tool specification dictionaries for requested tool names."""
        result = []
        for name in tool_names:
            if name in self.schemas:
                result.append(
                    {
                        "type": "function",
                        "function": self.schemas[name],
                    }
                )
        return result

    def dispatch(self, name: str, args: Dict[str, Any], caller_agent: str = "peter") -> ToolResult:
        """Dispatches tool execution through guardrail validation."""
        logger.info(f"ToolRegistry: Agent [{caller_agent}] requesting tool '{name}' with args {args}")

        # 1. Guardrail Evaluation
        gr_res = self.guardrails.check_tool_call(name, args)
        if not gr_res.allowed:
            err_msg = f"TOOL BLOCKED BY GUARDRAIL [{gr_res.rule_id}]: {gr_res.message}"
            logger.warning(err_msg)
            return ToolResult(success=False, output="", error=err_msg)

        # 2. Tool Lookup
        fn = self.tools.get(name)
        if not fn:
            return ToolResult(success=False, output="", error=f"Tool '{name}' is not registered in registry.")

        # 3. Tool Execution
        try:
            res = fn(**args)
            logger.info(f"ToolRegistry: Tool '{name}' execution finished (success={res.success})")
            return res
        except Exception as e:
            err_msg = f"Tool '{name}' raised unhandled exception: {e}"
            logger.error(err_msg)
            return ToolResult(success=False, output="", error=err_msg)
