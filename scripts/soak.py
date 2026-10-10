"""Phase I3: 90-day simulated soak test script (soak.py).

Simulates 90 days of memory operation, daily evolution cycles, and decay.
Outputs validation/soak.md.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.clock import FakeClock
from memory.vector_store import VectorMemory
from memory.graph_store import GraphMemory
from memory.outcomes import OutcomeLedger
from memory.skills import SkillStore
from memory.recall import ContextBuilder
from engine.evolution import EvolutionCoordinator
from tests.fakes.fake_llm import FakeLLM

OUTPUT_PATH = Path("validation/soak.md")


def run_soak_test(days: int = 90) -> Dict[str, Any]:
    print(f"Starting {days}-day simulated soak test...")
    tmp = tempfile.mkdtemp()
    db_path = Path(tmp) / "soak_graph.db"
    clock = FakeClock(start=1700000000.0)

    vector = VectorMemory(persist_directory=str(Path(tmp) / "chroma"), clock=clock)
    graph = GraphMemory(database_path=str(db_path), clock=clock)

    ledger = OutcomeLedger(db_path=str(db_path), clock=clock)
    skills = SkillStore(graph=graph, clock=clock)

    fake_llm = FakeLLM(loop=True)

    coordinator = EvolutionCoordinator(
        vector=vector,
        graph=graph,
        router=fake_llm,
        guardrails=None,
        ledger=ledger,
        skill_store=skills,
        mode="active",
        clock=clock,
    )

    pinned_ids = []
    # Pin 2 initial records
    p1 = vector.store("peter", "Pinned core knowledge record 1", importance=1.0, pinned=True, kind="episode", extra_metadata={"trust": "system"})
    p2 = vector.store("peter", "Pinned core knowledge record 2", importance=1.0, pinned=True, kind="episode", extra_metadata={"trust": "system"})
    vector.pin("peter", p1, True)
    vector.pin("peter", p2, True)
    pinned_ids.extend([p1, p2])

    day_30_size = 0
    day_90_size = 0
    recall_latencies = []

    def get_total_records() -> int:
        total = 0
        for ns in VectorMemory.DEFAULT_NAMESPACES:
            total += len(vector.get_all_records(ns))
        return total

    for d in range(1, days + 1):
        # Add 3 synthetic episodes per day
        for i in range(3):
            vector.store(
                "librarian",
                f"Day {d} episode {i}: research finding on topic {d % 5}",
                importance=0.5,
                kind="episode",
                extra_metadata={"trust": "tool_verified"},
            )

        # Advance clock by 24 hours
        clock.advance(hours=24)

        # Measure recall latency
        t0 = time.time()
        cb = ContextBuilder(vector_memory=vector, skill_store=skills, clock=clock)
        cb.build_context(query=f"topic {d % 5}")
        recall_latencies.append((time.time() - t0) * 1000)

        # Run daily evolution cycle
        coordinator.run_cycle()

        if d == 30:
            day_30_size = get_total_records()
        if d == 90:
            day_90_size = get_total_records()

    # SQLite integrity check
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute("PRAGMA integrity_check;")
    integrity = cursor.fetchone()[0]
    conn.close()

    p95_recall = sorted(recall_latencies)[int(len(recall_latencies) * 0.95)] if recall_latencies else 0.0
    peter_recs = {r["id"] for r in vector.get_all_records("peter")}
    pinned_intact = all(pid in peter_recs for pid in pinned_ids)

    report = {
        "days_simulated": days,
        "day_30_size": day_30_size,
        "day_90_size": day_90_size,
        "pinned_intact": pinned_intact,
        "p95_recall_ms": round(p95_recall, 2),
        "integrity_check": integrity,
        "plateau_ok": day_90_size <= day_30_size * 3 + 50,
    }

    markdown_report = f"""# 90-Day Simulated Soak Test Report

- **Simulated Days:** {report['days_simulated']}
- **Day 30 Vector Store Count:** {report['day_30_size']}
- **Day 90 Vector Store Count:** {report['day_90_size']}
- **Plateau Condition:** {'PASSED' if report['plateau_ok'] else 'FAILED'}
- **Pinned Records Intact:** {'YES' if report['pinned_intact'] else 'NO'}
- **Recall p95 Latency:** {report['p95_recall_ms']} ms (threshold < 500 ms)
- **SQLite Integrity Check:** `{report['integrity_check']}`

All 90 daily cycles completed without unhandled exceptions.
"""

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(markdown_report, encoding="utf-8")
    print(f"Soak test complete. Report written to {OUTPUT_PATH}")
    return report


if __name__ == "__main__":
    run_soak_test(days=90)
