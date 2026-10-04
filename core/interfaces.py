"""Core data structures and interfaces for Peter AI."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class AgentState(Enum):
    IDLE = "idle"
    RUNNING = "running"
    WAITING = "waiting"
    ERROR = "error"


class TaskStatus(Enum):
    """Lifecycle states for an async TaskPlan."""
    PLANNING = "planning"
    QUEUED = "queued"
    RUNNING = "running"
    COMPILING = "compiling"
    DONE = "done"
    FAILED = "failed"


@dataclass
class Message:
    role: str  # "system", "user", "assistant", "tool"
    content: str
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {"role": self.role, "content": self.content}
        if self.name:
            d["name"] = self.name
        if self.tool_calls:
            d["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        return d


@dataclass
class ToolResult:
    success: bool
    output: str
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GuardrailResult:
    allowed: bool
    rule_id: Optional[str] = None
    message: str = "Passed"
    action: str = "allow"  # "allow", "block", "warn"


@dataclass
class TaskStep:
    """A single unit of work in an async TaskPlan, delegated to one specialist."""
    step_id: int
    agent: str                          # specialist name: researcher, filesystem, etc.
    query: str                          # self-contained instruction for the agent
    depends_on: List[int] = field(default_factory=list)  # step_ids this step waits for
    status: str = "pending"             # pending | running | done | failed
    result: Optional[str] = None        # text output from the agent
    error: Optional[str] = None
    started_at: Optional[float] = None  # epoch
    completed_at: Optional[float] = None


@dataclass
class TaskPlan:
    """An async execution plan created by Peter's planner and executed by the TaskQueue."""
    task_id: str
    original_query: str
    created_at: float
    status: str = TaskStatus.PLANNING.value
    steps: List[TaskStep] = field(default_factory=list)
    final_output: Optional[str] = None
    error: Optional[str] = None

    def is_complete(self) -> bool:
        """Returns True when all steps have reached a terminal state."""
        if not self.steps:
            return False
        return all(s.status in ("done", "failed") for s in self.steps)

    def has_failures(self) -> bool:
        return any(s.status == "failed" for s in self.steps)

    def pending_steps(self) -> List[TaskStep]:
        """Returns steps whose dependencies are all done and that are still pending."""
        done_ids = {s.step_id for s in self.steps if s.status == "done"}
        return [
            s for s in self.steps
            if s.status == "pending" and all(d in done_ids for d in s.depends_on)
        ]
