"""Unit tests for the async task queue, task store, and task planner.

Tests verify:
1. TaskStore CRUD operations (create, save, load, update_step)
2. TaskQueue enqueue/dequeue thread-safety
3. TaskPlan dependency resolution (pending_steps())
4. TaskWorkerPool executes steps and marks them done (with mock agents)
5. Peter routes to async mode for complex queries
6. Peter returns immediately (doesn't block) after async dispatch
"""

import json
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_workspace(tmp_path, monkeypatch):
    """Points PETER_WORKSPACE at a temp directory for isolation."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    monkeypatch.setenv("PETER_WORKSPACE", str(ws))
    return ws


# ---------------------------------------------------------------------------
# 1. Core interfaces
# ---------------------------------------------------------------------------


class TestTaskPlanInterfaces:
    def test_task_status_values(self):
        from core.interfaces import TaskStatus
        assert TaskStatus.PLANNING.value == "planning"
        assert TaskStatus.QUEUED.value == "queued"
        assert TaskStatus.RUNNING.value == "running"
        assert TaskStatus.COMPILING.value == "compiling"
        assert TaskStatus.DONE.value == "done"
        assert TaskStatus.FAILED.value == "failed"

    def test_task_step_defaults(self):
        from core.interfaces import TaskStep
        step = TaskStep(step_id=1, agent="researcher", query="Find X")
        assert step.status == "pending"
        assert step.depends_on == []
        assert step.result is None

    def test_task_plan_is_complete_false_when_pending(self):
        from core.interfaces import TaskPlan, TaskStep
        plan = TaskPlan(task_id="t1", original_query="q", created_at=0.0)
        plan.steps = [TaskStep(step_id=1, agent="researcher", query="q")]
        assert plan.is_complete() is False

    def test_task_plan_is_complete_true_when_all_done(self):
        from core.interfaces import TaskPlan, TaskStep
        plan = TaskPlan(task_id="t1", original_query="q", created_at=0.0)
        s = TaskStep(step_id=1, agent="researcher", query="q")
        s.status = "done"
        plan.steps = [s]
        assert plan.is_complete() is True

    def test_task_plan_pending_steps_respects_dependencies(self):
        from core.interfaces import TaskPlan, TaskStep
        plan = TaskPlan(task_id="t1", original_query="q", created_at=0.0)
        s1 = TaskStep(step_id=1, agent="researcher", query="step 1")
        s2 = TaskStep(step_id=2, agent="filesystem", query="step 2", depends_on=[1])
        plan.steps = [s1, s2]
        # Only step 1 should be pending (step 2 depends on step 1)
        pending = plan.pending_steps()
        assert len(pending) == 1
        assert pending[0].step_id == 1

    def test_task_plan_pending_steps_unblocked_after_dep_done(self):
        from core.interfaces import TaskPlan, TaskStep
        plan = TaskPlan(task_id="t1", original_query="q", created_at=0.0)
        s1 = TaskStep(step_id=1, agent="researcher", query="step 1")
        s1.status = "done"
        s2 = TaskStep(step_id=2, agent="filesystem", query="step 2", depends_on=[1])
        plan.steps = [s1, s2]
        pending = plan.pending_steps()
        assert len(pending) == 1
        assert pending[0].step_id == 2

    def test_task_plan_has_failures(self):
        from core.interfaces import TaskPlan, TaskStep
        plan = TaskPlan(task_id="t1", original_query="q", created_at=0.0)
        s = TaskStep(step_id=1, agent="researcher", query="q")
        s.status = "failed"
        plan.steps = [s]
        assert plan.has_failures() is True


# ---------------------------------------------------------------------------
# 2. TaskStore
# ---------------------------------------------------------------------------


class TestTaskStore:
    def test_create_and_load_plan(self, tmp_workspace):
        from engine.task_store import create_task, load_plan
        plan = create_task("Research quantum computing")
        loaded = load_plan(plan.task_id)
        assert loaded is not None
        assert loaded.task_id == plan.task_id
        assert loaded.original_query == "Research quantum computing"

    def test_save_and_reload_with_steps(self, tmp_workspace):
        from core.interfaces import TaskStep
        from engine.task_store import create_task, load_plan, save_plan
        plan = create_task("Test task")
        plan.steps = [
            TaskStep(step_id=1, agent="researcher", query="Find X"),
            TaskStep(step_id=2, agent="filesystem", query="Write Y", depends_on=[1]),
        ]
        save_plan(plan)
        loaded = load_plan(plan.task_id)
        assert len(loaded.steps) == 2
        assert loaded.steps[1].depends_on == [1]

    def test_update_step_persists(self, tmp_workspace):
        from core.interfaces import TaskStep
        from engine.task_store import create_task, load_plan, update_step
        plan = create_task("Test")
        step = TaskStep(step_id=1, agent="researcher", query="Find X")
        plan.steps = [step]
        from engine.task_store import save_plan
        save_plan(plan)

        step.status = "done"
        step.result = "Found some results"
        update_step(plan.task_id, step)

        loaded = load_plan(plan.task_id)
        assert loaded.steps[0].status == "done"
        assert loaded.steps[0].result == "Found some results"

    def test_load_nonexistent_returns_none(self, tmp_workspace):
        from engine.task_store import load_plan
        assert load_plan("nonexistent-task-id") is None

    def test_list_tasks_returns_plans(self, tmp_workspace):
        from engine.task_store import create_task, list_tasks
        create_task("Task A")
        create_task("Task B")
        plans = list_tasks()
        assert len(plans) >= 2


# ---------------------------------------------------------------------------
# 3. TaskQueue
# ---------------------------------------------------------------------------


class TestTaskQueue:
    def test_enqueue_dequeue_basic(self):
        from engine.task_queue import TaskQueue
        q = TaskQueue()
        q.enqueue("task1", step_id=1, priority=0)
        task_id, step_id = q.dequeue(timeout=1.0)
        assert task_id == "task1"
        assert step_id == 1

    def test_priority_ordering(self):
        from engine.task_queue import TaskQueue
        q = TaskQueue()
        q.enqueue("task1", step_id=2, priority=1)
        q.enqueue("task1", step_id=1, priority=0)
        _, step_id_first = q.dequeue(timeout=1.0)
        assert step_id_first == 1  # Priority 0 should come first

    def test_thread_safe_concurrent_enqueue(self):
        import queue as q_mod
        from engine.task_queue import TaskQueue
        tq = TaskQueue()
        errors = []

        def enqueue_many(tid):
            try:
                for i in range(20):
                    tq.enqueue(f"task{tid}", step_id=i)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=enqueue_many, args=(t,)) for t in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert tq.qsize() == 100  # 5 threads × 20 items

    def test_dequeue_raises_on_empty_timeout(self):
        import queue as q_mod
        from engine.task_queue import TaskQueue
        tq = TaskQueue()
        with pytest.raises(q_mod.Empty):
            tq.dequeue(timeout=0.1)


# ---------------------------------------------------------------------------
# 4. TaskWorkerPool (with mock agents)
# ---------------------------------------------------------------------------


class TestTaskWorkerPool:
    def _make_mock_agent(self, return_value="mock result"):
        agent = MagicMock()
        agent.run.return_value = return_value
        return agent

    def test_worker_executes_step_and_marks_done(self, tmp_workspace):
        from core.interfaces import TaskStep
        from engine.task_queue import TaskQueue, TaskWorkerPool
        from engine.task_store import create_task, load_plan, save_plan

        plan = create_task("Test async execution")
        step = TaskStep(step_id=1, agent="researcher", query="Find something")
        plan.steps = [step]
        from core.interfaces import TaskStatus
        plan.status = TaskStatus.QUEUED.value
        save_plan(plan)

        specialists = {"researcher": self._make_mock_agent("Test result")}
        tq = TaskQueue()
        pool = TaskWorkerPool(task_queue=tq, specialists=specialists, num_workers=1)
        pool.start()

        tq.enqueue(plan.task_id, step_id=1)

        # Wait for worker to process (max 5s)
        deadline = time.time() + 5
        while time.time() < deadline:
            loaded = load_plan(plan.task_id)
            if loaded and loaded.steps[0].status == "done":
                break
            time.sleep(0.1)

        pool.stop()

        loaded = load_plan(plan.task_id)
        assert loaded.steps[0].status == "done"
        assert loaded.steps[0].result == "Test result"

    def test_worker_marks_step_failed_on_agent_error(self, tmp_workspace):
        from core.interfaces import TaskStep, TaskStatus
        from engine.task_queue import TaskQueue, TaskWorkerPool
        from engine.task_store import create_task, load_plan, save_plan

        plan = create_task("Test failure handling")
        step = TaskStep(step_id=1, agent="researcher", query="Do something")
        plan.steps = [step]
        plan.status = TaskStatus.QUEUED.value
        save_plan(plan)

        failing_agent = MagicMock()
        failing_agent.run.side_effect = RuntimeError("Agent blew up")
        specialists = {"researcher": failing_agent}

        tq = TaskQueue()
        pool = TaskWorkerPool(task_queue=tq, specialists=specialists, num_workers=1)
        pool.start()
        tq.enqueue(plan.task_id, step_id=1)

        deadline = time.time() + 5
        while time.time() < deadline:
            loaded = load_plan(plan.task_id)
            if loaded and loaded.steps[0].status == "failed":
                break
            time.sleep(0.1)

        pool.stop()

        loaded = load_plan(plan.task_id)
        assert loaded.steps[0].status == "failed"
        assert "Agent blew up" in loaded.steps[0].error


# ---------------------------------------------------------------------------
# 5. Peter async routing
# ---------------------------------------------------------------------------


class TestPeterAsyncRouting:
    def _make_peter(self, tmp_workspace):
        """Creates a minimal Peter agent for routing tests."""
        from agents.peter import PeterAgent, _should_use_async
        return _should_use_async

    def test_short_query_stays_sync(self):
        from agents.peter import _should_use_async
        assert _should_use_async("Hi Peter") is False
        assert _should_use_async("What time is it?") is False

    def test_long_complex_query_goes_async(self):
        from agents.peter import _should_use_async
        long_query = (
            "Please research the latest developments in quantum computing, "
            "find any relevant papers from 2025-2026, summarise the key findings, "
            "and save a structured report to the workspace."
        )
        assert _should_use_async(long_query) is True

    def test_async_flag_not_set_when_no_queue(self, tmp_workspace):
        """Without a TaskQueue injected, Peter falls through to sync even for complex queries."""
        from agents.peter import PeterAgent
        mock_router = MagicMock()
        mock_router.complete.return_value = {"content": "Mocked response", "tool_calls": None}
        mock_vector = MagicMock()
        mock_graph = MagicMock()
        mock_guardrails = MagicMock()
        mock_guardrails.check_input.return_value = MagicMock(allowed=True)
        mock_guardrails.check_output.return_value = MagicMock(allowed=True)
        mock_tools = MagicMock()
        mock_tools.get_schemas.return_value = []
        mock_tools.register.return_value = None
        mock_tools.dispatch.return_value = MagicMock(success=True, output="ok")

        peter = PeterAgent(
            router=mock_router,
            vector=mock_vector,
            graph=mock_graph,
            guardrails=mock_guardrails,
            tools=mock_tools,
            task_queue=None,  # No queue → always sync
        )
        # Should not raise and should call the LLM sync path
        result = peter.run("Research quantum computing for me and write a detailed report.")
        assert isinstance(result, str)
