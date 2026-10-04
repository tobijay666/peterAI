"""File-backed task state store for async TaskPlans.

Stores task plans and step results as JSON files inside Peter's workspace
directory (PETER_WORKSPACE/tasks/<task_id>/). This approach:
  - Survives crashes (state is on disk, not in-memory only)
  - Allows audit / manual inspection by Sir Anton
  - Integrates cleanly with the FileSystem agent's write scope
"""

import json
import os
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from core.interfaces import TaskPlan, TaskStep, TaskStatus
from core.logger import logger


def _get_tasks_root() -> Path:
    """Returns the root directory for all task folders, creating it if needed."""
    workspace = os.getenv("PETER_WORKSPACE", "workspace")
    root = Path(workspace) / "tasks"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _task_dir(task_id: str) -> Path:
    root = _get_tasks_root()
    d = root / task_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _plan_path(task_id: str) -> Path:
    return _task_dir(task_id) / "plan.json"


def _step_path(task_id: str, step_id: int) -> Path:
    return _task_dir(task_id) / f"step_{step_id}.json"


# --------------------------------------------------------------------------
# Serialisation helpers
# --------------------------------------------------------------------------

def _step_to_dict(step: TaskStep) -> dict:
    return {
        "step_id": step.step_id,
        "agent": step.agent,
        "query": step.query,
        "depends_on": step.depends_on,
        "status": step.status,
        "result": step.result,
        "error": step.error,
        "started_at": step.started_at,
        "completed_at": step.completed_at,
    }


def _dict_to_step(d: dict) -> TaskStep:
    return TaskStep(
        step_id=d["step_id"],
        agent=d["agent"],
        query=d["query"],
        depends_on=d.get("depends_on", []),
        status=d.get("status", "pending"),
        result=d.get("result"),
        error=d.get("error"),
        started_at=d.get("started_at"),
        completed_at=d.get("completed_at"),
    )


def _plan_to_dict(plan: TaskPlan) -> dict:
    return {
        "task_id": plan.task_id,
        "original_query": plan.original_query,
        "created_at": plan.created_at,
        "status": plan.status,
        "steps": [_step_to_dict(s) for s in plan.steps],
        "final_output": plan.final_output,
        "error": plan.error,
    }


def _dict_to_plan(d: dict) -> TaskPlan:
    return TaskPlan(
        task_id=d["task_id"],
        original_query=d["original_query"],
        created_at=d["created_at"],
        status=d.get("status", TaskStatus.PLANNING.value),
        steps=[_dict_to_step(s) for s in d.get("steps", [])],
        final_output=d.get("final_output"),
        error=d.get("error"),
    )


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def create_task(query: str) -> TaskPlan:
    """Creates a new TaskPlan with a unique ID and persists it to disk."""
    task_id = str(uuid.uuid4())[:8]
    plan = TaskPlan(
        task_id=task_id,
        original_query=query,
        created_at=time.time(),
        status=TaskStatus.PLANNING.value,
    )
    save_plan(plan)
    logger.info(f"[TaskStore] Created new task '{task_id}'.")
    return plan


def save_plan(plan: TaskPlan) -> None:
    """Persists the full plan to disk."""
    path = _plan_path(plan.task_id)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(_plan_to_dict(plan), f, indent=2)
    except Exception as e:
        logger.error(f"[TaskStore] Failed to save plan {plan.task_id}: {e}")


def load_plan(task_id: str) -> Optional[TaskPlan]:
    """Loads a TaskPlan from disk. Returns None if not found."""
    path = _plan_path(task_id)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return _dict_to_plan(json.load(f))
    except Exception as e:
        logger.error(f"[TaskStore] Failed to load plan {task_id}: {e}")
        return None


def update_step(task_id: str, step: TaskStep) -> None:
    """Persists a single step update (result, status) and refreshes plan.json."""
    plan = load_plan(task_id)
    if not plan:
        logger.warning(f"[TaskStore] update_step: task '{task_id}' not found.")
        return
    for i, s in enumerate(plan.steps):
        if s.step_id == step.step_id:
            plan.steps[i] = step
            break
    save_plan(plan)

    # Also write an individual step result file for easy inspection
    step_path = _step_path(task_id, step.step_id)
    try:
        with open(step_path, "w", encoding="utf-8") as f:
            json.dump(_step_to_dict(step), f, indent=2)
    except Exception as e:
        logger.error(f"[TaskStore] Failed to write step file {step_path}: {e}")


def list_tasks() -> List[TaskPlan]:
    """Returns all task plans from the workspace, newest first."""
    root = _get_tasks_root()
    plans = []
    for task_dir in sorted(root.iterdir(), key=lambda d: d.stat().st_mtime, reverse=True):
        plan_file = task_dir / "plan.json"
        if plan_file.exists():
            p = load_plan(task_dir.name)
            if p:
                plans.append(p)
    return plans


def cleanup_task(task_id: str) -> None:
    """Removes a completed task directory (optional housekeeping)."""
    import shutil
    d = _task_dir(task_id)
    try:
        shutil.rmtree(d)
        logger.info(f"[TaskStore] Cleaned up task '{task_id}'.")
    except Exception as e:
        logger.warning(f"[TaskStore] Could not clean up task '{task_id}': {e}")
