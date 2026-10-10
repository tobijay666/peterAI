"""Task Planner — Peter's fast structured planning LLM call.

This module is responsible for a single, focused job: take a user query
and return a decomposed TaskPlan (structured JSON) as fast as possible,
so Peter can return to the user immediately with an ETA/plan summary
while workers execute in the background.

Design choices:
- Uses a dedicated, stripped-down system prompt focused ONLY on decomposition
  (not on execution), which keeps the context window tiny.
- Requests JSON-mode output to avoid ambiguous free-text parsing.
- Falls back to a single-step plan if the LLM fails or returns malformed JSON.
"""

import json
import re
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from core.interfaces import TaskPlan, TaskStep, TaskStatus
from core.logger import logger
from engine import task_store
from llm.router import LLMRouter
from core.interfaces import Message

if TYPE_CHECKING:
    pass

# Specialist agents that Peter can delegate to
VALID_AGENTS = {"researcher", "filesystem", "librarian", "devops", "qa", "voice"}

PLANNER_SYSTEM_PROMPT = """You are Peter's task planning module. Your ONLY job is to decompose a user request into a list of parallel or sequential steps that can be executed by specialist agents.

Specialist agents available:
- researcher: web search, scholarly search, URL fetching
- filesystem: reading/writing files on disk
- librarian: retrieving or storing long-term memory
- devops: git operations, running tests
- qa: writing and running test suites
- voice: text-to-speech output

Output ONLY valid JSON in this exact format — no explanation, no markdown fences:
{
  "steps": [
    {
      "step_id": 1,
      "agent": "<agent_name>",
      "query": "<self-contained instruction for that agent>",
      "depends_on": []
    },
    {
      "step_id": 2,
      "agent": "<agent_name>",
      "query": "<instruction that depends on step 1 output>",
      "depends_on": [1]
    }
  ],
  "summary": "<one sentence explaining the plan to Sir Anton>"
}

Rules:
1. Steps with no dependencies run in parallel. Use depends_on wisely.
2. Each step must be a SELF-CONTAINED instruction — the agent has no memory of other steps.
3. Use at most 6 steps. Prefer fewer, broader steps.
4. Only use agent names from the list above.
5. Do not include a compilation step — that is handled automatically.
"""


def _extract_json_from_response(text: str) -> Optional[dict]:
    """Extracts JSON from LLM response, stripping any markdown fences."""
    # Try raw parse first
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    # Strip markdown code fences and retry
    cleaned = re.sub(r"```(?:json)?", "", text).strip().strip("`")
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


def create_plan(
    query: str,
    router: LLMRouter,
    policy: Optional[List[str]] = None,
    context_builder: Optional[Any] = None,
    skill_hint: Optional[str] = None,
) -> tuple:
    """Runs the planning LLM call and returns a populated TaskPlan.

    If the LLM fails or returns malformed output, falls back to a
    single-step plan delegating to the researcher (safe default).
    """
    plan = task_store.create_task(query)

    recalled_ctx = None
    system_prompt = PLANNER_SYSTEM_PROMPT

    if context_builder is not None:
        recalled_ctx = context_builder.build(query)
        if recalled_ctx.memory_text:
            system_prompt = recalled_ctx.memory_text + "\n\n" + system_prompt
        if recalled_ctx.skill_hint:
            skill_hint = recalled_ctx.skill_hint

    if skill_hint:
        system_prompt += f"\n\nSKILL HINT: A similar task has been done before. Consider using this pattern:\n{skill_hint}"

    messages = [
        Message(role="system", content=system_prompt),
        Message(role="user", content=f"Decompose this request into steps:\n\n{query}"),
    ]

    plan_policy = policy or ["nvidia", "groq", "openrouter", "huggingface"]

    try:
        completion = router.complete(
            messages=messages,
            policy=plan_policy,
            tools=None,
            temperature=0.2,   # Low temp for deterministic structured output
            max_tokens=1024,
        )
        raw = completion.get("content", "")
        data = _extract_json_from_response(raw)

        if data and "steps" in data:
            steps = []
            for s in data["steps"]:
                agent = s.get("agent", "researcher").lower()
                if agent not in VALID_AGENTS:
                    agent = "researcher"  # safe fallback
                steps.append(TaskStep(
                    step_id=int(s.get("step_id", len(steps) + 1)),
                    agent=agent,
                    query=str(s.get("query", query)),
                    depends_on=[int(d) for d in s.get("depends_on", [])],
                ))
            plan.steps = steps
            plan.status = TaskStatus.QUEUED.value
            plan_summary = data.get("summary", f"Plan with {len(steps)} step(s) created.")
            task_store.save_plan(plan)
            logger.info(f"[TaskPlanner] Plan '{plan.task_id}' created with {len(steps)} step(s).")
            return plan, plan_summary, recalled_ctx
        else:
            raise ValueError(f"Malformed planner output: {raw[:200]}")

    except Exception as e:
        logger.error(f"[TaskPlanner] Planning LLM call failed: {e}. Using single-step fallback.")
        plan.steps = [
            TaskStep(
                step_id=1,
                agent="researcher",
                query=query,
                depends_on=[],
            )
        ]
        plan.status = TaskStatus.QUEUED.value
        task_store.save_plan(plan)
        summary = "I'll work on this for you, Sir. I've created a single-step plan."
        return plan, summary, recalled_ctx
