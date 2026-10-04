"""Integration test: full async plan-execute-compile cycle.

Tests the complete pipeline end-to-end using mock LLM calls and mock
specialist agents, verifying that:
  1. Peter creates a plan and returns an ack immediately (not after finishing)
  2. Workers execute all steps in the background
  3. Compiler fires and produces a final_output
  4. The 'task.compiled' event is published
  5. The plan file on disk reaches DONE status
"""

import json
import threading
import time
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_workspace(tmp_path, monkeypatch):
    ws = tmp_path / "workspace"
    ws.mkdir()
    monkeypatch.setenv("PETER_WORKSPACE", str(ws))
    return ws


def _make_mock_router(plan_json: str = None, compile_text: str = "Compiled answer for Sir."):
    """Creates a mock LLMRouter that returns a JSON plan on the first call,
    then returns a compile response on subsequent calls."""
    call_count = [0]

    def mock_complete(messages, policy=None, tools=None, temperature=0.7, max_tokens=2048):
        call_count[0] += 1
        if call_count[0] == 1:
            # Planner call
            if plan_json:
                content = plan_json
            else:
                content = json.dumps({
                    "steps": [
                        {"step_id": 1, "agent": "researcher", "query": "Find recent papers on quantum", "depends_on": []},
                        {"step_id": 2, "agent": "filesystem", "query": "Save the findings", "depends_on": [1]},
                    ],
                    "summary": "2-step plan: research then save."
                })
            return {"content": content, "tool_calls": None, "usage": {}}
        else:
            # Compiler call
            return {"content": compile_text, "tool_calls": None, "usage": {}}

    mock = MagicMock()
    mock.complete.side_effect = mock_complete
    return mock


def _make_mock_agent(result: str = "agent result"):
    agent = MagicMock()
    agent.run.return_value = result
    return agent


# ---------------------------------------------------------------------------
# Test: Peter plan-and-dispatch returns immediately
# ---------------------------------------------------------------------------


class TestPeterReturnsImmediately:
    def test_async_ack_returned_before_agents_finish(self, tmp_workspace):
        """Peter must return an acknowledgement string immediately after queueing,
        not after all specialist agents have run."""
        from core.interfaces import TaskStatus
        from engine.task_queue import TaskQueue, TaskWorkerPool
        from engine.task_store import list_tasks

        mock_router = _make_mock_router()

        # Slow agent — simulates a long-running specialist
        slow_agent = MagicMock()
        slow_agent_called = threading.Event()

        def slow_run(user_input, **kwargs):
            slow_agent_called.set()
            time.sleep(3)  # Simulates 3s work
            return "slow result"

        slow_agent.run.side_effect = slow_run

        specialists = {
            "researcher": slow_agent,
            "filesystem": _make_mock_agent("saved"),
        }

        task_queue = TaskQueue()
        worker_pool = TaskWorkerPool(task_queue=task_queue, specialists=specialists, num_workers=2)
        worker_pool.start()

        from agents.peter import PeterAgent

        mock_vector = MagicMock()
        mock_graph = MagicMock()
        mock_graph.upsert_node.return_value = "node_id"
        mock_guardrails = MagicMock()
        mock_guardrails.check_input.return_value = MagicMock(allowed=True)
        mock_guardrails.check_output.return_value = MagicMock(allowed=True)
        mock_tools = MagicMock()
        mock_tools.get_schemas.return_value = []
        mock_tools.register.return_value = None

        peter = PeterAgent(
            router=mock_router,
            vector=mock_vector,
            graph=mock_graph,
            guardrails=mock_guardrails,
            tools=mock_tools,
            specialists=specialists,
            task_queue=task_queue,
        )

        start = time.time()
        ack = peter.run(
            "Please research quantum computing breakthroughs and write a full report to the workspace."
        )
        elapsed = time.time() - start

        worker_pool.stop()

        # Peter must return in well under 3 seconds (the slow agent's duration)
        assert elapsed < 2.5, f"Peter took {elapsed:.1f}s — should have returned immediately"
        assert "plan" in ack.lower() or "step" in ack.lower() or "task" in ack.lower()


# ---------------------------------------------------------------------------
# Test: Full pipeline — plan → execute → compile
# ---------------------------------------------------------------------------


class TestFullAsyncPipeline:
    def test_pipeline_completes_and_writes_final_output(self, tmp_workspace):
        """Full end-to-end: plan → worker executes all steps → compiler writes final output."""
        from core.events import event_bus
        from core.interfaces import TaskStatus
        from engine import task_planner, task_compiler, task_store
        from engine.task_queue import TaskQueue, TaskWorkerPool

        compiled_event = threading.Event()
        compiled_output = [None]

        def on_compiled(topic, payload):
            compiled_output[0] = payload.get("output")
            compiled_event.set()

        event_bus.subscribe("task.compiled", on_compiled)

        # Also wire the all_steps_completed → compile path manually
        def on_steps_done(topic, payload):
            task_id = payload["task_id"]
            mock_router2 = MagicMock()
            mock_router2.complete.return_value = {"content": "Final synthesis for Sir.", "tool_calls": None, "usage": {}}
            task_compiler.compile_plan(task_id=task_id, router=mock_router2)

        event_bus.subscribe("task.all_steps_completed", on_steps_done)

        # Create a plan with 2 sequential steps
        mock_router = _make_mock_router()
        plan, summary = task_planner.create_plan(
            query="Research X and save the results",
            router=mock_router,
        )
        assert len(plan.steps) > 0

        specialists = {
            "researcher": _make_mock_agent("research findings"),
            "filesystem": _make_mock_agent("file saved"),
        }
        task_queue = TaskQueue()
        pool = TaskWorkerPool(task_queue=task_queue, specialists=specialists, num_workers=2)
        pool.start()

        # Enqueue all ready steps
        for step in plan.pending_steps():
            task_queue.enqueue(plan.task_id, step.step_id)

        # Wait for the compiled event (max 15s)
        assert compiled_event.wait(timeout=15), "Pipeline did not complete within 15 seconds"
        pool.stop()

        # Verify on-disk state
        final_plan = task_store.load_plan(plan.task_id)
        assert final_plan is not None
        assert final_plan.status == TaskStatus.DONE.value
        assert final_plan.final_output is not None
        assert "Sir" in final_plan.final_output or len(final_plan.final_output) > 0

    def test_plan_with_dependencies_executes_in_order(self, tmp_workspace):
        """Steps with depends_on should execute after their dependencies complete."""
        from engine import task_store
        from engine.task_queue import TaskQueue, TaskWorkerPool
        from core.interfaces import TaskStep, TaskStatus

        execution_order = []
        lock = threading.Lock()

        def make_ordered_agent(name):
            agent = MagicMock()
            def run_fn(user_input, **kwargs):
                with lock:
                    execution_order.append(name)
                time.sleep(0.1)
                return f"result from {name}"
            agent.run.side_effect = run_fn
            return agent

        # Create plan manually
        from engine.task_store import create_task, save_plan
        plan = create_task("ordered test")
        s1 = TaskStep(step_id=1, agent="researcher", query="step 1", depends_on=[])
        s2 = TaskStep(step_id=2, agent="filesystem", query="step 2", depends_on=[1])
        plan.steps = [s1, s2]
        plan.status = TaskStatus.QUEUED.value
        save_plan(plan)

        specialists = {
            "researcher": make_ordered_agent("researcher"),
            "filesystem": make_ordered_agent("filesystem"),
        }
        tq = TaskQueue()
        pool = TaskWorkerPool(task_queue=tq, specialists=specialists, num_workers=2)
        pool.start()

        # Only step 1 is ready (step 2 depends on step 1)
        tq.enqueue(plan.task_id, step_id=1, priority=0)

        # Wait for both to finish
        deadline = time.time() + 10
        while time.time() < deadline:
            loaded = task_store.load_plan(plan.task_id)
            all_done = loaded and all(s.status in ("done", "failed") for s in loaded.steps)
            if all_done:
                break
            time.sleep(0.2)

        pool.stop()

        assert len(execution_order) == 2, f"Expected 2 steps executed, got {execution_order}"
        assert execution_order[0] == "researcher", "Step 1 (researcher) should run first"
        assert execution_order[1] == "filesystem", "Step 2 (filesystem) should run second"


# ---------------------------------------------------------------------------
# Test: Provider timeout fix
# ---------------------------------------------------------------------------


class TestProviderTimeoutFix:
    def test_failed_model_is_blacklisted_immediately(self):
        """After a model fails, subsequent complete() calls must skip it."""
        from llm.providers.base import BaseProvider, ProviderError

        class TestProvider(BaseProvider):
            pass

        provider = TestProvider(
            name="TestProvider",
            base_url="http://fake",
            api_key_env="FAKE_KEY",
            models=["model-a", "model-b"],
            priority=1,
        )
        # Inject a fake API key so is_available() returns True
        provider.api_key = "fake-key-123"

        call_log = []

        def fake_complete_side_effect(**kwargs):
            model = kwargs.get("model")
            call_log.append(model)
            if model == "model-a":
                raise ConnectionError("Connection timeout")
            # model-b succeeds
            mock_resp = MagicMock()
            mock_resp.choices[0].message.content = "Success"
            mock_resp.choices[0].message.tool_calls = None
            mock_resp.choices[0].message.role = "assistant"
            mock_resp.usage.prompt_tokens = 10
            mock_resp.usage.completion_tokens = 5
            mock_resp.usage.total_tokens = 15
            return mock_resp

        mock_client = MagicMock()
        # Disable the with_raw_response path so the plain create() path is used
        del mock_client.chat.completions.with_raw_response
        mock_client.chat.completions.create.side_effect = fake_complete_side_effect
        provider.client = mock_client


        from core.interfaces import Message

        msgs = [Message(role="user", content="hello")]

        # First call — model-a fails and is blacklisted, model-b succeeds
        result = provider.complete(messages=msgs)
        assert result["content"] == "Success"
        assert "model-a" in provider._failed_models

        call_log.clear()

        # Second call — model-a should be skipped immediately
        result = provider.complete(messages=msgs)
        assert "model-a" not in call_log, "Blacklisted model-a was called again!"
        assert result["content"] == "Success"

    def test_completion_timeout_env_var_respected(self):
        """COMPLETION_TIMEOUT_SEC env var must be read at construction time."""
        import os
        from unittest.mock import patch

        with patch.dict(os.environ, {"COMPLETION_TIMEOUT_SEC": "42", "FAKE_KEY": "val"}):
            from llm.providers.base import BaseProvider

            class P(BaseProvider):
                pass

            p = P(name="T", base_url="", api_key_env="FAKE_KEY", models=[], priority=1)
            assert p._completion_timeout == 42.0
