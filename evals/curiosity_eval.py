"""Curiosity evaluation (Stage 8, S8.4).

Claim under test (from the project proposal): a curious Peter answers related,
NOT-YET-ASKED questions better than a memory-only Peter.

Protocol (same for both arms):
  1. Peter is asked 4 seed questions (henna, saree, poruwa, kiribath) and researches them.
  2. Curious arm only: after each turn the curiosity engine queues adjacent questions,
     then one sleep cycle researches the queue.
  3. Both arms get 10 held-out related questions. A question counts as answerable when
     graph recall finds its topic and the sufficiency check passes WITHOUT new research.

Modes:
  default (offline)  scripted proposals + a fake web: proves the wiring end to end,
                     NOT that Peter got smarter.
  --live             real LLM router (config/providers.yaml, keys from .env) and the real
                     web: the actual evidence. Small sample; report it as measured.

Run:  python evals/curiosity_eval.py            (offline)
      python evals/curiosity_eval.py --live     (needs API keys + internet; ~5–10 min)
Writes validation/runs/cognitive/curiosity_eval[_live].json
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SEED_QUESTIONS = ["what is henna?", "what is a kandyan saree?", "what is the poruwa ceremony?", "what is kiribath?"]
HELD_OUT = [
    "is black henna safe for children?",
    "how long does a henna stain last?",
    "what jewellery is worn with a kandyan saree?",
    "who ties the gold thread at a poruwa ceremony?",
    "what is the nekath auspicious time?",
    "what sweets are served at a sinhala wedding?",
    "what is the meaning of the white cloth at a poruwa?",
    "how is kiribath cooked?",
    "what is ashtaka chanting at weddings?",
    "what does the bride wear for the homecoming?",
]

# Offline arm: what a curious LLM would plausibly propose, and a web that has the answers
PROPOSALS = {
    "henna": ["Is black henna safe for children?", "How long does a henna stain last?"],
    "kandyan saree": ["What jewellery is worn with a kandyan saree?", "What does the bride wear for the homecoming?"],
    "poruwa ceremony": ["Who ties the gold thread at a poruwa ceremony?", "What is ashtaka chanting at weddings?",
                        "What is the nekath auspicious time?"],
    "kiribath": ["How is kiribath cooked?", "What sweets are served at a sinhala wedding?"],
}
FACTS = {
    "henna": "Henna is a plant dye from Lawsonia inermis applied to hands before weddings.",
    "kandyan saree": "The kandyan saree is the traditional Sri Lankan bridal drape with a frilled waist.",
    "poruwa ceremony": "The poruwa ceremony is a Sinhalese wedding ritual on a decorated platform.",
    "kiribath": "Kiribath is milk rice served at Sri Lankan celebrations and weddings.",
    "black henna": "Black henna contains PPD and is unsafe for children because it causes skin burns.",
    "henna stain": "A henna stain lasts one to three weeks as the skin naturally exfoliates.",
    "jewellery": "Kandyan brides wear the seven-tiered necklace and gold jewellery with the kandyan saree.",
    "homecoming": "For the homecoming the bride wears a saree in a bright colour, often a going-away saree.",
    "gold thread": "At a poruwa ceremony an elder ties the gold thread around the couple's little fingers.",
    "ashtaka": "Ashtaka chanting at weddings is a blessing recited by a chanter during the poruwa ceremony.",
    "nekath": "The nekath auspicious time is set by an astrologer for each wedding ritual.",
    "sweets": "Sweets served at a sinhala wedding include kavum, kokis and aluwa.",
    "kiribath cooked": "Kiribath is cooked by boiling rice and adding thick coconut milk, then cut into diamonds.",
}


def _offline_web():
    def search(query: str, n: int):
        q = query.lower()
        hits = [k for k in FACTS if all(w in q for w in k.split())]
        return [{"title": k, "url": f"https://{k.replace(' ', '-')}-{i}.example/", "snippet": FACTS[k]}
                for k in hits for i in range(2)][:n]

    def fetch(url: str) -> str:
        key = url.split("//")[1].rsplit("-", 1)[0].replace("-", " ")
        return FACTS.get(key, "")
    return search, fetch


class _ScriptedCuriosityRouter:
    """Returns the scripted proposals for whichever seed topic is in the prompt."""

    def complete(self, messages, **kw) -> Dict[str, Any]:
        text = " ".join(m.content for m in messages if m.content).lower()
        for topic, qs in PROPOSALS.items():
            if topic in text:
                return {"content": "\n".join(qs)}
        return {"content": ""}


def build(tmp: Path, live: bool):
    from core.clock import FakeClock, SystemClock
    from engine.curiosity import CuriosityEngine
    from engine.research_pipeline import ResearchPipeline
    from memory.cognition import CognitiveMemory
    from memory.graph_store import GraphMemory
    from memory.vector_store import VectorMemory

    clock = SystemClock() if live else FakeClock(start=1e9)
    if live:
        from llm.router import LLMRouter
        router = LLMRouter(str(ROOT / "config" / "providers.yaml"))
        vector = VectorMemory(str(tmp / "chroma"), clock=clock, user_id="eval")
        search = fetch = None
    else:
        from tests.fakes.word_embedder import WordHashEmbedder
        router = None
        vector = VectorMemory(str(tmp / "chroma"), clock=clock, user_id="eval", embedding_function=WordHashEmbedder())
        search, fetch = _offline_web()
    graph = GraphMemory(str(tmp / "graph.db"), clock=clock, user_id="eval")
    cog = CognitiveMemory(vector, graph, router=router, async_research=False)
    cog.research = ResearchPipeline(cog.knowledge, router=router, search_fn=search, fetch_fn=fetch)
    cog.curiosity = CuriosityEngine(cog.knowledge, resolver=cog.resolver,
                                    router=router if live else _ScriptedCuriosityRouter())
    return cog


def arm(curious: bool, live: bool) -> Dict[str, Any]:
    import engine.evolution as evolution
    from core.cognition_config import set_cognition_config
    from engine.evolution import EvolutionCoordinator
    from memory.snapshot import snapshot_manager_for

    tmp = Path(tempfile.mkdtemp(prefix="peter-eval-"))
    evolution.JOURNAL_PATH = tmp / "journal.jsonl"
    cfg = {"sleep": {"research_items_per_cycle": 20, "max_new_collections": 12},
           "sufficiency": {"use_llm_judge": live}}
    if not live:
        cfg["resolver"] = {"use_llm": False}
        cfg["sufficiency"]["min_records"] = 1
    set_cognition_config(cfg)
    cog = build(tmp, live)
    for q in SEED_QUESTIONS:
        result, verdict = cog.recall(q)
        if not verdict.sufficient:
            cog.handle_gap(q, result, verdict)
        if curious:
            cog.after_turn(q, "")
    if curious:
        ledger = MagicMock()
        ledger._conn.side_effect = Exception("no ledger")
        skills = MagicMock()
        skills.list_skills.return_value = []
        EvolutionCoordinator(vector=cog.vector, graph=cog.graph, router=cog.router, guardrails=MagicMock(),
                             ledger=ledger, skill_store=skills, consolidation_config={"min_age_hours": 1e9},
                             mode="active", clock=cog._clock, cognition=cog,
                             snapshot_manager=snapshot_manager_for(cog.vector, cog.graph, clock=cog._clock)).run_cycle()
    rows: List[Dict[str, Any]] = []
    for q in HELD_OUT:
        result, verdict = cog.recall(q, touch=False)
        rows.append({"question": q, "answerable": verdict.sufficient, "reason": verdict.reason,
                     "nodes": [n["label"] for n in result.matched_nodes], "best_score": round(result.best_score, 3)})
    set_cognition_config(None)
    return {"answerable": sum(r["answerable"] for r in rows), "of": len(rows), "rows": rows,
            "topics_known": len(cog.graph.get_nodes_by_kind("entity"))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="real LLMs + real web (needs keys and internet)")
    args = ap.parse_args()
    import core.events as ev
    ev._write_event_log = lambda uid, payload: None   # keep the repo's event logs clean
    memory_only = arm(curious=False, live=args.live)
    curious = arm(curious=True, live=args.live)
    report = {
        "mode": "live" if args.live else "offline (wiring only — not evidence of real improvement)",
        "memory_only": memory_only, "curious": curious,
        "delta_answerable": curious["answerable"] - memory_only["answerable"],
        "caveats": [] if args.live else [
            "Offline mode uses scripted curiosity proposals and a fake web, so it proves the pipeline "
            "(curiosity -> sleep research -> recall) works, not that Peter is smarter.",
            "Offline mode has no LLM judge: 'answerable' only means the topic was found and its records "
            "are similar to the question, which can over-count (e.g. a generic saree fact for a jewellery "
            "question). Live mode turns the judge on for those ambiguous cases.",
        ],
    }
    out = ROOT / "validation" / "runs" / "cognitive" / ("curiosity_eval_live.json" if args.live else "curiosity_eval.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"memory-only: {memory_only['answerable']}/{memory_only['of']} answerable without new research")
    print(f"curious:     {curious['answerable']}/{curious['of']} answerable without new research")
    print(f"written to {out}")
    return 0 if args.live or report["delta_answerable"] > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
