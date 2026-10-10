"""Object factories for tests (A3)."""
from __future__ import annotations
import uuid
from core.interfaces import TaskPlan, TaskStep


def make_task_plan(
    steps: int = 1,
    agent: str = "researcher",
    original_query: str = "test query",
    created_at: float = 0.0,
) -> TaskPlan:
    plan = TaskPlan(
        task_id=str(uuid.uuid4()),
        original_query=original_query,
        created_at=created_at,
    )
    plan.steps = [
        TaskStep(
            step_id=i + 1,
            agent=agent,
            query=f"step {i + 1} of {original_query}",
            depends_on=[i] if i > 0 else [],
        )
        for i in range(steps)
    ]
    return plan


def make_outcome(
    task_id: str = None,
    step_id: str = None,
    level: str = "task",
    agent: str = "peter",
    status: str = "success",
    latency_ms: int = 100,
    skill_id: str = None,
    provider: str = None,
    model: str = None,
    tokens_in: int = 0,
    tokens_out: int = 0,
    memory_ids: list = None,
    created_at: float = 0.0,
) -> dict:
    return {
        "id": str(uuid.uuid4()),
        "task_id": task_id or str(uuid.uuid4()),
        "step_id": step_id,
        "level": level,
        "agent": agent,
        "tool": None,
        "skill_id": skill_id,
        "provider": provider,
        "model": model,
        "status": status,
        "latency_ms": latency_ms,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "memory_ids": memory_ids or [],
        "created_at": created_at,
    }
