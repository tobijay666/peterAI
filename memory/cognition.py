"""CognitiveMemory — one object that bundles the cognitive-memory parts for a user.

    recall(query)      graph lookup -> node collection -> sufficiency verdict   (Stage 2)
    handle_gap(...)    on-demand research for missing knowledge                (Stage 3)
    after_turn(...)    curiosity questions for the queue                       (Stage 4)

Built once per user in peter.bootstrap_for_user / bootstrap_system and attached
to the ContextBuilder, so BaseAgent picks it up automatically.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from core.clock import Clock
from core.cognition_config import cognition_config
from core.events import emit
from core.logger import logger
from memory.entity_resolver import EntityResolver
from memory.graph_recall import GraphRecall, GraphRecallResult
from memory.graph_store import GraphMemory
from memory.knowledge import KnowledgeStore
from memory.sufficiency import SufficiencyChecker, SufficiencyVerdict
from memory.usage import UsageTracker
from memory.vector_store import VectorMemory


@dataclass
class GapOutcome:
    """What handle_gap did: researched now (records > 0), started in background, or nothing."""
    mode: str                 # "sync" | "async" | "skipped"
    topics: List[str]
    records: int = 0
    note: str = ""


class CognitiveMemory:
    def __init__(self, vector: VectorMemory, graph: GraphMemory, router: Any = None,
                 clock: Optional[Clock] = None, async_research: Optional[bool] = None,
                 research: Any = None, curiosity: Any = None) -> None:
        self.vector = vector
        self.graph = graph
        self.router = router
        self._clock = clock or graph._clock
        self.usage = UsageTracker(graph, clock=self._clock)
        self.knowledge = KnowledgeStore(vector, graph, usage=self.usage, clock=self._clock)
        from memory.layout import LayoutEngine
        self.layout = LayoutEngine(vector, graph)
        self.knowledge.layout = self.layout
        self.resolver = EntityResolver(vector, graph, router=router)
        self.graph_recall = GraphRecall(vector, graph, self.resolver, self.usage)
        self.sufficiency = SufficiencyChecker(router=router)
        self._async_override = async_research
        self.research = research          # ResearchPipeline (Stage 3), optional
        self.curiosity = curiosity        # CuriosityEngine (Stage 4), optional
        self._inflight: set = set()
        self._lock = threading.Lock()
        self.last_research_thread: Optional[threading.Thread] = None

    @property
    def user_id(self) -> str:
        return self.graph.user_id

    @property
    def enabled(self) -> bool:
        return bool(cognition_config().get("enabled", True))

    @property
    def async_research(self) -> bool:
        if self._async_override is not None:
            return self._async_override
        return bool(cognition_config()["research"].get("async_in_chat", True))

    # ------------------------------------------------------------------ Stage 2

    def recall(self, query: str, turn_id: str = "", touch: bool = True
               ) -> "tuple[GraphRecallResult, SufficiencyVerdict]":
        result = self.graph_recall.recall(query, turn_id=turn_id, touch=touch)
        verdict = self.sufficiency.check(query, result)
        return result, verdict

    # ------------------------------------------------------------------ Stage 3

    def gap_topics(self, result: GraphRecallResult, verdict: SufficiencyVerdict) -> List[str]:
        """What to research: unknown terms when nothing matched; otherwise the matched
        topic(s) — unless the judge named an unknown term as the missing piece."""
        unknown = set(result.unmatched_terms)
        if verdict.reason == "no_node":
            return list(result.unmatched_terms)
        if verdict.reason == "llm_judge":
            named = [m for m in verdict.missing_aspects if m in unknown]
            if named:
                return named
        if verdict.reason in ("weak_evidence", "llm_judge"):
            return [n["label"] for n in result.matched_nodes] or list(verdict.missing_aspects)
        return []

    def handle_gap(self, query: str, result: GraphRecallResult, verdict: SufficiencyVerdict,
                   turn_id: str = "") -> GapOutcome:
        topics = self.gap_topics(result, verdict)
        for t in topics:
            emit("gap_detected", user_id=self.user_id, turn_id=turn_id, term=t, reason=verdict.reason,
                 missing_aspects=verdict.missing_aspects)
        if not topics or self.research is None or not cognition_config()["research"].get("enabled", True):
            return GapOutcome("skipped", topics, note="no research pipeline" if topics else "")

        with self._lock:
            fresh = [t for t in topics if t.lower() not in self._inflight]
            for t in fresh:
                self._inflight.add(t.lower())
        if not fresh:
            return GapOutcome("skipped", topics, note="already researching")

        def run() -> int:
            total = 0
            try:
                for t in fresh:
                    out = self.research.research(t, question=query, missing_aspects=verdict.missing_aspects,
                                                 turn_id=turn_id)
                    total += out.records_stored
                if self.async_research and total:
                    self._deliver_followup(query, turn_id)
            except Exception as e:
                logger.error(f"CognitiveMemory: research failed: {e}")
            finally:
                with self._lock:
                    for t in fresh:
                        self._inflight.discard(t.lower())
            return total

        if self.async_research:
            th = threading.Thread(target=run, daemon=True, name="ResearchThread")
            self.last_research_thread = th
            th.start()
            return GapOutcome("async", fresh, note="research started in background")
        n = run()
        return GapOutcome("sync", fresh, records=n)

    def _deliver_followup(self, query: str, turn_id: str) -> None:
        """Async mode: after background research, send the cited answer as a follow-up."""
        try:
            answer = self.research.answer(query)
            emit("research_answer", user_id=self.user_id, turn_id=turn_id, question=query[:300],
                 answer=answer)
        except Exception as e:
            logger.warning(f"CognitiveMemory: follow-up answer failed: {e}")

    # ------------------------------------------------------------------ Stage 4

    def after_turn(self, query: str, reply: str, turn_id: str = "") -> List[Dict[str, Any]]:
        if self.curiosity is None:
            return []
        try:
            return self.curiosity.generate(query, reply, turn_id=turn_id)
        except Exception as e:
            logger.warning(f"CognitiveMemory: curiosity generation failed: {e}")
            return []
