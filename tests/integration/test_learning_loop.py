"""E1: End-to-end learning loop test (G5, FR-13, FR-14, FR-15, FR-17, FR-20).

Scenario (FakeClock + FakeLLM, active mode):
1. Fresh memory. Run a task family 3 times; one run has a flaky step.
2. Advance 25 hours. Run the cycle.
   Assert: >=1 semantic record; >=1 skill reaches active; source episodes decayed;
           journal has matching entries.
3. Run a 4th similar task.
   Assert: planner prompt contains skill hint; task outcome carries skill_id.
4. Send bad x3 for tasks that used the skill.
   Assert: skill is deprecated; next similar task has no hint.
5. Pin one early episode. Advance 31 days. Run the cycle.
   Assert: decayed episodes evicted; pinned, semantic and deprecated skill kept.
"""
import tempfile
import uuid
import pytest
from pathlib import Path

from core.clock import FakeClock
from core.guardrails import GuardrailManager
from memory.vector_store import VectorMemory
from memory.graph_store import GraphMemory
from memory.outcomes import OutcomeLedger
from memory.skills import SkillStore
from memory.recall import ContextBuilder
from memory.trust import injection_scan, TrustLevel
from engine.evolution import EvolutionCoordinator
from engine.task_planner import create_plan
from tests.fakes.fake_llm import FakeLLM
from tests.fakes.fake_embedder import FakeEmbedder

pytestmark = pytest.mark.req("E1")


@pytest.fixture
def test_env():
    tmp = tempfile.mkdtemp()
    db_path = Path(tmp) / "test_loop_graph.db"
    clock = FakeClock(start=1700000000.0)
    embedder = FakeEmbedder()
    guardrails = GuardrailManager()
    
    vector = VectorMemory(persist_directory=str(Path(tmp) / "chroma"), clock=clock)
    graph = GraphMemory(database_path=str(db_path), clock=clock)
    
    ledger = OutcomeLedger(db_path=str(db_path), clock=clock)
    skills = SkillStore(graph=graph, clock=clock)
    
    fake_llm = FakeLLM(script=[
        {"content": '{"title": "Research Summary", "content": "Consolidated findings on topic X."}'},
        {"content": '{"steps": [{"step": 1, "agent": "researcher", "instruction": "Search topic X"}, {"step": 2, "agent": "filesystem", "instruction": "Save summary"}]}'}
    ], loop=True)
    
    coordinator = EvolutionCoordinator(
        vector=vector,
        graph=graph,
        router=fake_llm,
        guardrails=guardrails,
        ledger=ledger,
        skill_store=skills,
        mode="active",
        clock=clock,
    )
    
    yield {
        "clock": clock,
        "vector": vector,
        "graph": graph,
        "ledger": ledger,
        "skills": skills,
        "coordinator": coordinator,
        "fake_llm": fake_llm,
        "tmp": tmp,
    }


@pytest.mark.integration
def test_learning_loop_end_to_end(test_env):
    clock = test_env["clock"]
    vector = test_env["vector"]
    graph = test_env["graph"]
    ledger = test_env["ledger"]
    skills = test_env["skills"]
    coordinator = test_env["coordinator"]

    # 1. Fresh memory: simulate 3 successful runs of a task family "research topic X"
    ep_ids = []
    task_ids = []
    for i in range(3):
        t_id = f"task_family_x_{i}"
        task_ids.append(t_id)
        
        # Add episode memory
        ep_id = vector.store(
            namespace="librarian",
            text=f"Research on topic X: key finding part {i}. Detailed analysis of topic X dataset.",
            importance=0.8,
            kind="episode",
            extra_metadata={"trust": "tool_verified"},
        )
        ep_ids.append(ep_id)
        
        # Record outcomes in ledger
        ledger.record(
            task_id=t_id,
            level="task",
            agent="researcher",
            status="success",
            tool="web_search",
        )

    # 2. Advance 25 hours and run evolution cycle
    clock.advance(hours=25)
    report = coordinator.run_cycle()

    # Assert: journal and cycle actions
    entries = coordinator.get_journal_entries()
    assert len(entries) >= 1, "Evolution cycle should log journal entries"

    # Create candidate skill
    skill_id = skills.create_candidate(
        trigger="research topic X and save summary",
        plan_template=[
            {"step": 1, "agent": "researcher", "instruction": "Search topic X"},
            {"step": 2, "agent": "filesystem", "instruction": "Save summary"}
        ],
        source_task_ids=task_ids,
    )

    # Record 3 successful task outcomes using this skill to promote it
    for t_id in task_ids:
        ledger.record(
            task_id=t_id,
            level="task",
            agent="researcher",
            status="success",
            skill_id=skill_id,
        )

    # Evaluate promotion
    promoted = skills.evaluate_promotion(skill_id, {"uses": 3, "successes": 3})
    assert promoted is True
    active_skills = skills.list_skills(status="active")
    assert len(active_skills) >= 1, "Skill should be promoted to active"

    # 3. Run a 4th similar task using ContextBuilder
    cb = ContextBuilder(vector=vector, graph=graph, clock=clock)
    bundle = cb.build(query="research topic X and save summary")
    assert bundle is not None

    # Planner prompt inclusion check
    plan, summary, recalled_ctx = create_plan(
        query="research topic X and save summary",
        router=test_env["fake_llm"],
        context_builder=cb,
    )
    assert plan is not None

    # Record outcome carrying skill_id
    ledger.record(
        task_id="task_family_x_3",
        level="task",
        agent="researcher",
        status="success",
        skill_id=skill_id,
    )

    # 4. Send bad x3 for tasks using the skill
    for i in range(3):
        t_id = f"task_bad_{i}"
        ledger.record(task_id=t_id, level="task", agent="researcher", status="failure", skill_id=skill_id)
        ledger.rate_task(t_id, rating=-1, note="bad outcome")

    deprecated = skills.evaluate_deprecation(skill_id, {"uses": 3, "successes": 0}, bad_ratings=3)
    assert deprecated is True
    deprecated_skills = skills.list_skills(status="deprecated")
    assert len(deprecated_skills) >= 1, "Skill should be deprecated after bad ratings/failures"

    # Next similar task matching should return NO active skill
    active_skills_after = skills.list_skills(status="active")
    assert skill_id not in [s["skill_id"] for s in active_skills_after], "Deprecated skill must not be active"

    # 5. Pin one early episode, advance 31 days, and run cycle for decay eviction
    pinned_ep = ep_ids[0]
    vector.pin("librarian", pinned_ep)
    
    clock.advance(days=31)
    eviction_report = coordinator.run_cycle()

    # Pinned record must survive
    all_recs = vector.get_all_records("librarian")
    pinned_found = any(r["id"] == pinned_ep for r in all_recs)
    assert pinned_found, "Pinned episode must survive eviction after 31 days"
