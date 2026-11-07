"""Interactive REPL engine loop for user interaction with Peter.

v1.2 additions:
  - Background task monitor thread: watches for 'task.all_steps_completed'
    events and triggers the TaskCompiler, then announces the result.
  - 'tasks' command: shows status of all active and recent tasks.
  - TTS announcement when a background task compiles.
"""

import sys
import threading
from typing import Optional

from agents.peter import PeterAgent
from core.console import Colors, format_agent, format_error, format_prompt, format_warn
from core.events import event_bus
from core.guardrails import GuardrailManager
from core.interfaces import TaskStatus
from core.logger import logger
from engine import task_compiler, task_store
from engine.task_queue import TaskQueue, TaskWorkerPool
from llm.router import LLMRouter
from tts.service import TTSService


class REPLEngine:
    """Manages the interactive terminal REPL session for Sir Anton."""

    def __init__(
        self,
        peter_agent: PeterAgent,
        tts_service: TTSService,
        guardrails: GuardrailManager,
        worker_pool: Optional[TaskWorkerPool] = None,
        router: Optional[LLMRouter] = None,
        context_builder: Optional[object] = None
    ):
        self.peter = peter_agent
        self.tts = tts_service
        self.guardrails = guardrails
        self.worker_pool = worker_pool
        self.router = router
        self.context_builder = context_builder
        self.running = False
        # Queue of completed-task announcements to print on the next REPL prompt
        self._pending_announcements: list = []
        self._announce_lock = threading.Lock()

        # Subscribe to task completion events from background workers
        event_bus.subscribe("task.all_steps_completed", self._on_steps_complete)
        event_bus.subscribe("task.compiled", self._on_compiled)

    # ------------------------------------------------------------------
    # Event handlers (called from worker threads)
    # ------------------------------------------------------------------

    def _on_steps_complete(self, topic: str, payload: dict) -> None:
        """Triggered when all steps of a background task finish.

        Runs the compiler (still in the worker thread) to produce the
        final answer, then queues it for announcement.
        """
        task_id = payload.get("task_id", "unknown")
        logger.info(f"[REPL] Received task.all_steps_completed for task '{task_id}'. Starting compilation.")
        try:
            policy = self.peter.policy if self.peter else None
            router = self.router or self.peter.router
            task_compiler.compile_plan(task_id=task_id, router=router, policy=policy)
        except Exception as e:
            logger.error(f"[REPL] Compilation failed for task '{task_id}': {e}")

    def _on_compiled(self, topic: str, payload: dict) -> None:
        """Triggered when the compiler has produced the final answer.

        Queues the announcement string so the main REPL thread can print
        it safely between prompts (avoids console interleaving).
        """
        task_id = payload.get("task_id", "unknown")
        output = payload.get("output", "")
        announcement = f"\n[Background Task '{task_id}' Complete]\n{output}"
        with self._announce_lock:
            self._pending_announcements.append((task_id, output, announcement))
        logger.info(f"[REPL] Queued announcement for task '{task_id}'.")

    def _flush_announcements(self) -> None:
        """Prints any queued task completion announcements and speaks them."""
        with self._announce_lock:
            announcements = list(self._pending_announcements)
            self._pending_announcements.clear()

        for task_id, output, announcement in announcements:
            print(f"\n{Colors.CYAN}{'=' * 60}{Colors.RESET}")
            print(format_agent("Peter", f"Sir, background task '{task_id}' has completed:"))
            print(format_agent("Peter", output))
            print(f"{Colors.CYAN}{'=' * 60}{Colors.RESET}\n")
            self.tts.speak(f"Sir, background task {task_id} has completed.")
            # Speak the actual result so Peter reads results aloud, not just the notice
            if output and output.strip():
                self.tts.speak(output)

    # ------------------------------------------------------------------
    # REPL lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        self.running = True
        print(f"\n{Colors.CYAN}{Colors.BOLD}===================================================={Colors.RESET}")
        print(f"{Colors.CYAN}{Colors.BOLD}     Peter AI Multi-Agent Harness (v1.2.0)          {Colors.RESET}")
        print(f"{Colors.CYAN}{Colors.BOLD}===================================================={Colors.RESET}\n")

        # Startup Announcement
        welcome_msg = "Peter online. Ready to assist, Sir."
        print(format_agent("Peter", welcome_msg))
        self.tts.speak(welcome_msg)

        while self.running:
            # Print any background task announcements before showing the prompt
            self._flush_announcements()

            try:
                user_input = input(format_prompt("\nSir Anton > ")).strip()
                if not user_input:
                    continue

                if user_input.lower() in ["exit", "quit", "q"]:
                    print(format_agent("Peter", "Shutting down system. Goodbye, Sir."))
                    self.tts.speak_blocking("Peter going offline. Goodbye, Sir.")
                    self.running = False
                    break

                if user_input.lower() == "help":
                    self._show_help()
                    continue

                if user_input.lower() == "status":
                    self._show_status()
                    continue

                if user_input.lower() == "tasks":
                    self._show_tasks()
                    continue

                if user_input.lower() == "reload-guardrails":
                    self.guardrails.reload()
                    print(format_agent("Peter", "Guardrail rules reloaded successfully, Sir."))
                    continue
                    
                if user_input.lower().startswith("recall "):
                    query = user_input[7:].strip()
                    if self.context_builder:
                        print(format_agent("Peter", self.context_builder.explain(query)))
                    else:
                        print(format_agent("Peter", "ContextBuilder is not available."))
                    continue

                # Run Peter agent (sync or async depending on query complexity)
                response = self.peter.run(user_input=user_input)

                # Output final response
                print("\n" + format_agent("Peter", response))
                self.tts.speak(response)

            except KeyboardInterrupt:
                print("\n" + format_warn("Interrupt signal received. Type 'exit' to quit."))
            except Exception as e:
                logger.error(f"REPL Loop Exception: {e}")
                print(format_error(f"An error occurred: {e}"))

    # ------------------------------------------------------------------
    # Help & status display
    # ------------------------------------------------------------------

    def _show_help(self) -> None:
        print("\n--- Peter AI Help Commands ---")
        print(" exit / quit  : Shut down the system")
        print(" help         : Display command instructions")
        print(" status       : Display agent state and provider token usage")
        print(" tasks        : Show background task queue status")
        print(" reload-guardrails : Reload security rules from config/guardrails.json")

    def _show_status(self) -> None:
        print("\n--- System Status ---")
        print(f"Peter State : {self.peter.state.value}")
        print(f"Specialists : {list(self.peter.specialists.keys())}")
        print(f"Token Stats : {self.peter.router.get_token_usage()}")

        if self.peter._task_queue:
            q = self.peter._task_queue
            print(f"Task Queue  : {q.qsize()} step(s) pending")

    def _show_tasks(self) -> None:
        """Displays the status of all known tasks from the workspace."""
        plans = task_store.list_tasks()
        if not plans:
            print(format_agent("Peter", "No tasks found in the workspace, Sir."))
            return

        print("\n--- Background Tasks ---")
        for plan in plans[:10]:  # Show most recent 10
            status_color = {
                TaskStatus.DONE.value: Colors.GREEN,
                TaskStatus.FAILED.value: Colors.RED,
                TaskStatus.RUNNING.value: Colors.YELLOW,
                TaskStatus.COMPILING.value: Colors.YELLOW,
                TaskStatus.QUEUED.value: Colors.CYAN,
                TaskStatus.PLANNING.value: Colors.CYAN,
            }.get(plan.status, Colors.RESET)

            done_steps = sum(1 for s in plan.steps if s.status == "done")
            total_steps = len(plan.steps)
            failed_steps = sum(1 for s in plan.steps if s.status == "failed")

            status_str = f"{status_color}{plan.status.upper()}{Colors.RESET}"
            print(
                f"  [{plan.task_id}] {status_str}  "
                f"Steps: {done_steps}/{total_steps}"
                + (f" ({failed_steps} failed)" if failed_steps else "")
            )
            print(f"           Query: {plan.original_query[:80]}{'...' if len(plan.original_query) > 80 else ''}")
            if plan.status == TaskStatus.DONE.value and plan.final_output:
                preview = plan.final_output[:120].replace('\n', ' ')
                print(f"           Result preview: {preview}...")
        print()


