"""Cognitive-memory soak (Stage 8, S8.3): N simulated days, offline and deterministic.

Every day Peter gets questions about a pool of recurring topics plus the odd
one-off topic, researches gaps (against a synthetic web), queues curiosity,
and sleeps once. The soak checks that memory reaches a steady state instead of
growing forever:

  * one-off topics are forgotten (pruned) while recurring ones stay
  * knowledge-collection count plateaus (last 30 days vs the 30 before)
  * recall p95 < 500 ms
  * SQLite integrity_check is ok, no sleep-cycle errors

Run:  python scripts/soak_cognitive.py --days 90
Writes nothing into the repo (temp dir; events/journal redirected).
"""
from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DAY = 86400.0
RECURRING = [
    "aster gardening", "basalt geology", "cobalt chemistry", "delta rivers", "ember forging",
    "fjord ecology", "glacier physics", "harbor shipping", "indigo dyeing", "juniper botany",
    "kelp farming", "lantern festivals",
]


def _web():
    def search(query: str, n: int) -> List[Dict[str, str]]:
        slug = "-".join(query.lower().split()[:2])
        return [{"title": f"{query} guide {i}", "url": f"https://{slug}-{i}.example/page", "snippet": ""}
                for i in range(2)][:n]

    def fetch(url: str) -> str:
        words = url.split("//")[1].split(".")[0].rsplit("-", 1)[0].replace("-", " ")
        return " ".join(f"{words} fact number {k} explains one detail about {words} in plain terms."
                        for k in range(3))
    return search, fetch


def run(days: int = 90, seed: int = 7, verbose: bool = False) -> Dict[str, Any]:
    import core.events as ev
    import engine.evolution as evolution
    from core.clock import FakeClock
    from core.cognition_config import set_cognition_config
    from engine.curiosity import CuriosityEngine
    from engine.evolution import EvolutionCoordinator
    from engine.research_pipeline import ResearchPipeline
    from memory.cognition import CognitiveMemory
    from memory.graph_store import GraphMemory
    from memory.snapshot import snapshot_manager_for
    from memory.vector_store import VectorMemory
    from tests.fakes.word_embedder import WordHashEmbedder

    tmp = Path(tempfile.mkdtemp(prefix="peter-soak-"))
    counters = {"spawned": 0, "pruned": 0}

    def capture(uid: str, payload: Dict[str, Any]) -> None:
        if payload["type"] == "node_spawned" and payload.get("kind") == "entity":
            counters["spawned"] += 1
        if payload["type"] == "node_pruned":
            counters["pruned"] += 1

    orig_log, orig_journal = ev._write_event_log, evolution.JOURNAL_PATH
    ev._write_event_log = capture
    evolution.JOURNAL_PATH = tmp / "journal.jsonl"
    set_cognition_config({"resolver": {"use_llm": False}, "sleep": {"research_items_per_cycle": 1}})
    try:
        rng = random.Random(seed)
        clock = FakeClock(start=10_000 * DAY)
        vector = VectorMemory(str(tmp / "chroma"), clock=clock, user_id="soak", embedding_function=WordHashEmbedder())
        graph = GraphMemory(str(tmp / "graph.db"), clock=clock, user_id="soak")
        cog = CognitiveMemory(vector, graph, router=None, async_research=False)
        search, fetch = _web()
        cog.research = ResearchPipeline(cog.knowledge, router=None, search_fn=search, fetch_fn=fetch)
        cog.curiosity = CuriosityEngine(cog.knowledge, resolver=cog.resolver, router=None)
        ledger = MagicMock()
        ledger._conn.side_effect = Exception("no ledger in soak")
        skills = MagicMock()
        skills.list_skills.return_value = []

        latencies: List[float] = []
        per_day: List[int] = []
        errors: List[str] = []
        for day in range(1, days + 1):
            questions = [f"what is {rng.choice(RECURRING[:6] if rng.random() < 0.7 else RECURRING)}?" for _ in range(2)]
            if rng.random() < 0.35:
                questions.append(f"what is {'quixote' + str(day)} trivia?")
            for q in questions:
                t0 = time.perf_counter()
                result, verdict = cog.recall(q)
                latencies.append((time.perf_counter() - t0) * 1000)
                if not verdict.sufficient:
                    cog.handle_gap(q, result, verdict)
            report = EvolutionCoordinator(
                vector=vector, graph=graph, router=None, guardrails=MagicMock(), ledger=ledger, skill_store=skills,
                consolidation_config={"min_age_hours": 10_000_000}, mode="active", clock=clock, cognition=cog,
                snapshot_manager=snapshot_manager_for(vector, graph, clock=clock, max_snapshots=2),
            ).run_cycle()
            errors.extend(report.get("errors", []))
            per_day.append(len(vector.list_knowledge_namespaces()))
            if verbose and day % 10 == 0:
                print(f"day {day}: collections={per_day[-1]} spawned={counters['spawned']} pruned={counters['pruned']}")
            clock.advance(days=1)

        lat = sorted(latencies)
        p95 = lat[int(0.95 * (len(lat) - 1))] if lat else 0.0
        with sqlite3.connect(str(graph.db_path)) as conn:
            integrity = conn.execute("PRAGMA integrity_check;").fetchone()[0]
        recurring_alive = sum(1 for t in RECURRING[:6] if cog.knowledge.find_node(t))
        last = max(per_day[-30:]) if len(per_day) >= 30 else max(per_day)
        prev = max(per_day[-60:-30]) if len(per_day) >= 60 else last
        result = {
            "days": days, "recall_p95_ms": round(p95, 2), "integrity": integrity, "errors": errors,
            "entities_spawned": counters["spawned"], "nodes_pruned": counters["pruned"],
            "collections_final": per_day[-1], "collections_max_last30": last, "collections_max_prev30": prev,
            "frequent_topics_alive": recurring_alive,
        }
        checks = {
            "recall_p95_under_500ms": p95 < 500,
            "integrity_ok": integrity == "ok",
            "no_errors": not errors,
            "one_off_topics_forgotten": counters["pruned"] > 0 and per_day[-1] < counters["spawned"],
            "frequent_topics_kept": recurring_alive == 6,
            "collections_plateau": last <= prev * 1.15 + 2,
        }
        result["checks"] = checks
        result["passed"] = all(checks.values())
        graph.close()
        return result
    finally:
        ev._write_event_log = orig_log
        evolution.JOURNAL_PATH = orig_journal
        set_cognition_config(None)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()
    import logging
    logging.getLogger("peter").setLevel(logging.WARNING)
    res = run(args.days, args.seed, verbose=True)
    text = json.dumps(res, indent=2)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
    failed = [k for k, ok in res["checks"].items() if not ok]
    print("SOAK: PASS" if res["passed"] else f"SOAK: FAIL ({', '.join(failed)})")
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
