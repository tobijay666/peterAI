"""Async Task Queue and Worker Pool for Peter AI.

Architecture:
  - TaskQueue: thread-safe priority queue of (step, task_plan) pairs
  - TaskWorkerPool: N daemon worker threads that pop steps from the queue,
    run them on the appropriate specialist agent, write results to the
    TaskStore, and fire events when all steps of a plan complete.

Peter enqueues all steps from a new TaskPlan immediately and returns to
the user. Workers execute them concurrently (respecting depends_on ordering
by re-queuing dependant steps once their prerequisites finish).
"""

import queue
import threading
from typing import TYPE_CHECKING, Dict, Optional

from core.events import event_bus
from core.interfaces import TaskStatus
from core.logger import logger
from core.clock import Clock, SystemClock
from engine import task_store

if TYPE_CHECKING:
    from agents.base import BaseAgent

# Number of concurrent worker threads (agents share an LLM router with
# its own internal model-level serialisation — more than 3 threads rarely
# helps and can exhaust rate limits faster).
DEFAULT_WORKERS = 2

# Sentinel value to signal worker shutdown
_STOP = object()


class TaskQueue:
    """Thread-safe priority queue of (priority, step_id, task_id) tuples.

    Lower priority number = executed first.  Tasks with no dependencies
    get priority 0; dependent tasks are re-queued at priority 1 once their
    prerequisites finish.
    """

    def __init__(self) -> None:
        self._q: queue.PriorityQueue = queue.PriorityQueue()
        self._lock = threading.Lock()
        # Maps task_id → set of step_ids currently running or queued
        self._active: Dict[str, set] = {}

    def enqueue(self, task_id: str, step_id: int, priority: int = 0) -> None:
        with self._lock:
            self._active.setdefault(task_id, set()).add(step_id)
        self._q.put((priority, SystemClock().now(), task_id, step_id))
        logger.debug(f"[TaskQueue] Enqueued step {step_id} for task '{task_id}' (priority={priority}).")

    def dequeue(self, timeout: float = 1.0):
        """Returns (task_id, step_id) or raises queue.Empty."""
        priority, ts, task_id, step_id = self._q.get(timeout=timeout)
        return task_id, step_id

    def mark_done(self, task_id: str, step_id: int) -> None:
        with self._lock:
            active = self._active.get(task_id, set())
            active.discard(step_id)

    def task_done(self) -> None:
        self._q.task_done()

    def is_task_active(self, task_id: str) -> bool:
        with self._lock:
            return bool(self._active.get(task_id))

    def qsize(self) -> int:
        return self._q.qsize()


class TaskWorkerPool:
    """Pool of daemon threads that execute queued task steps.

    Each worker:
    1. Pops a (task_id, step_id) from the queue.
    2. Loads the plan from the TaskStore.
    3. Checks if the step's dependencies are all done (if not, re-queues
       with a lower priority to avoid spinning).
    4. Runs the appropriate specialist agent.
    5. Writes the result back to the TaskStore.
    6. Checks if the plan is now complete; if so, fires 'task.all_steps_completed'.
    7. Re-queues any newly unblocked dependent steps.
    """

    def __init__(
        self,
        task_queue: TaskQueue,
        specialists: Dict[str, "BaseAgent"],
        num_workers: int = DEFAULT_WORKERS,
        clock: Optional[Clock] = None,
    ) -> None:
        self._clock = clock or SystemClock()
        self._queue = task_queue
        self._specialists = specialists
        self._num_workers = num_workers
        self._threads: list = []
        self._running = False

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        for i in range(self._num_workers):
            t = threading.Thread(target=self._worker, name=f"TaskWorker-{i}", daemon=True)
            t.start()
            self._threads.append(t)
        logger.info(f"[TaskWorkerPool] Started {self._num_workers} worker thread(s).")

    def stop(self) -> None:
        self._running = False
        logger.info("[TaskWorkerPool] Stopping worker threads...")

    def _worker(self) -> None:
        while self._running:
            try:
                task_id, step_id = self._queue.dequeue(timeout=1.0)
            except queue.Empty:
                continue

            try:
                self._execute_step(task_id, step_id)
            except Exception as e:
                logger.error(f"[TaskWorker] Unhandled error on task={task_id} step={step_id}: {e}", exc_info=True)
            finally:
                self._queue.task_done()
                self._queue.mark_done(task_id, step_id)

    def _execute_step(self, task_id: str, step_id: int) -> None:
        plan = task_store.load_plan(task_id)
        if not plan:
            logger.warning(f"[TaskWorker] Plan '{task_id}' not found — skipping step {step_id}.")
            return

        step = next((s for s in plan.steps if s.step_id == step_id), None)
        if not step:
            logger.warning(f"[TaskWorker] Step {step_id} not found in plan '{task_id}'.")
            return

        # Skip if already processed
        if step.status in ("done", "failed"):
            return

        # Check dependencies — if not all done, re-queue with delay
        done_ids = {s.step_id for s in plan.steps if s.status == "done"}
        unmet = [d for d in step.depends_on if d not in done_ids]
        if unmet:
            logger.debug(f"[TaskWorker] Step {step_id} waiting for deps {unmet}. Re-queuing.")
            self._queue.enqueue(task_id, step_id, priority=1)
            return

        # Mark step as running
        step.status = "running"
        step.started_at = self._clock.now()
        task_store.update_step(task_id, step)

        # Also mark the overall plan as running
        plan = task_store.load_plan(task_id)
        if plan and plan.status == TaskStatus.QUEUED.value:
            plan.status = TaskStatus.RUNNING.value
            task_store.save_plan(plan)

        logger.info(f"[TaskWorker] Executing step {step_id} (agent={step.agent}) for task '{task_id}'.")

        agent = self._specialists.get(step.agent)
        if not agent:
            step.status = "failed"
            step.error = f"Specialist '{step.agent}' not available."
            step.completed_at = self._clock.now()
            task_store.update_step(task_id, step)
            logger.error(f"[TaskWorker] {step.error}")
        else:
            try:
                result_text = agent.run(user_input=step.query)
                step.status = "done"
                step.result = result_text
                step.completed_at = self._clock.now()
                task_store.update_step(task_id, step)
                logger.info(f"[TaskWorker] Step {step_id} completed for task '{task_id}'.")
            except Exception as e:
                step.status = "failed"
                step.error = str(e)
                step.completed_at = self._clock.now()
                task_store.update_step(task_id, step)
                logger.error(f"[TaskWorker] Step {step_id} failed for task '{task_id}': {e}")

        # Reload plan to get latest step states and enqueue newly-unblocked steps
        plan = task_store.load_plan(task_id)
        if not plan:
            return

        # Enqueue any steps that just became unblocked
        newly_unblocked = plan.pending_steps()
        for pending_step in newly_unblocked:
            self._queue.enqueue(task_id, pending_step.step_id, priority=0)

        # Fire completion event if all steps are terminal
        if plan.is_complete():
            plan.status = TaskStatus.COMPILING.value
            task_store.save_plan(plan)
            logger.info(f"[TaskWorker] All steps complete for task '{task_id}'. Firing completion event.")
            event_bus.publish("task.all_steps_completed", {"task_id": task_id})
