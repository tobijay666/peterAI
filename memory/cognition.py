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


from memory.hierarchy import Hierarchy, is_subject as _is_subject, node_data as _node_data  # noqa: E402


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
        # Peter's own node, main subjects under it, similarity links between topics
        self.hierarchy = Hierarchy(vector, graph, knowledge=self.knowledge)
        # Edges grounded in the vectors: weights, record-level links, similar_to kNN (Part C)
        from memory.vector_graph import VectorGraph
        self.vgraph = VectorGraph(vector, graph, knowledge=self.knowledge)
        self.hierarchy.vgraph = self.vgraph
        self.knowledge.on_linked = self.vgraph.weight_data
        self.knowledge.after_linked = lambda eid, s, d, rel: (
            self.vgraph.link_records(eid, graph.get_node(s), graph.get_node(d))
            if rel != "has_subject" and graph.get_node(s) and graph.get_node(d) else 0)
        self.knowledge.on_records_changed = self.vgraph.on_records_changed
        self.knowledge.on_refreshed = self.vgraph.refresh_node
        self.knowledge.on_created = self.hierarchy.link_similar
        self.graph_recall = GraphRecall(vector, graph, self.resolver, self.usage, hierarchy=self.hierarchy)
        self.sufficiency = SufficiencyChecker(router=router)
        self._async_override = async_research
        self.research = research          # ResearchPipeline (Stage 3), optional
        self.curiosity = curiosity        # CuriosityEngine (Stage 4), optional
        self._inflight: set = set()
        self._lock = threading.Lock()
        self.last_research_thread: Optional[threading.Thread] = None
        # Existing users' graphs: alias keys, hierarchy repair, alias index (once per graph)
        from memory.upgrade import run_upgrades
        run_upgrades(self)

    @property
    def user_id(self) -> str:
        return self.graph.user_id

    @property
    def semantic_ok(self) -> bool:
        """False when embeddings are the random-hash fallback: similarity is meaningless."""
        try:
            return bool(self.vector.semantic_ok)
        except Exception:
            return True

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
        if touch and result.matched_nodes:
            # Known topics asked about together are related; the field hub links them too
            try:
                self.link_context(result, result.matched_nodes[0]["id"], turn_id=turn_id)
            except Exception as e:
                logger.debug(f"CognitiveMemory: link_context failed: {e}")
        return result, verdict

    # ------------------------------------------------------------------ Stage 3

    def gap_topics(self, result: GraphRecallResult, verdict: SufficiencyVerdict) -> List[str]:
        """The ONE topic to research now: the main unknown entity when nothing matched,
        otherwise the matched topic (unless the judge named an unknown entity as missing).
        Other unknown entities in the question are queued for curiosity, not researched now."""
        unknown = list(result.unmatched_terms)
        if verdict.reason == "no_node":
            return unknown[:1]
        if verdict.reason == "llm_judge":
            named = [m for m in verdict.missing_aspects if m in unknown]
            if named:
                return named[:1]
        if verdict.reason in ("weak_evidence", "llm_judge"):
            labels = [n["label"] for n in result.matched_nodes]
            return labels[:1] or list(verdict.missing_aspects)[:1]
        return []

    # ------------------------------------------------------------------ logical links

    def link_context(self, result: GraphRecallResult, primary_id: Optional[str], turn_id: str = "") -> None:
        """Connect what the question connects.

        * other known entities in the same question  -> related_to the main topic
          ("The Picture of Dorian Gray" <-> "Oscar Wilde")
        * the topic's field (resolver category)       -> an abstraction that abstracts it
          ("Victorian literature" -> "The Picture of Dorian Gray")
        """
        if not primary_id or self.graph.get_node(primary_id) is None:
            return
        hubs: List[str] = []
        try:
            self._link_context(result, primary_id, turn_id, hubs)
        finally:
            if hubs:
                self.fill_hubs_now(hubs, turn_id=turn_id)     # Part B: never an empty hub

    def _link_context(self, result: GraphRecallResult, primary_id: str, turn_id: str, hubs: List[str]) -> None:
        for n in result.matched_nodes:
            if n["id"] != primary_id:
                self.knowledge.link(primary_id, n["id"], "related_to", turn_id=turn_id)
        category = (getattr(result.resolution, "category", "") or "").strip()
        subject = (getattr(result.resolution, "subject", "") or "").strip()
        if not cognition_config()["resolver"].get("link_category", True):
            return
        under = primary_id               # what the subject hub sits above
        if category and category.lower() != subject.lower():
            try:
                hub, _ = self.knowledge.ensure_node(category, kind="abstraction", turn_id=turn_id)
                if hub["id"] != primary_id:
                    self.knowledge.link(hub["id"], primary_id, "abstracts", turn_id=turn_id)
                    under = hub["id"]
                    hubs.append(hub["id"])
            except Exception as e:
                logger.debug(f"CognitiveMemory: category hub skipped: {e}")
        # Main subject ("Writing", "Investing"): the top of the tree, linked to Peter himself
        if subject:
            try:
                top, _ = self.knowledge.ensure_node(subject, kind="abstraction", turn_id=turn_id,
                                                    data={"role": "subject"})
                self.hierarchy.mark_subject(top, turn_id=turn_id)
                if top["id"] != under:
                    self.knowledge.link(top["id"], under, "abstracts", turn_id=turn_id)
                hubs.append(top["id"])
            except Exception as e:
                logger.debug(f"CognitiveMemory: subject hub skipped: {e}")

    def attach_orphans(self, turn_id: str = "", limit: int = 12, budget: Any = None,
                       created: Optional[List[str]] = None) -> int:
        """Every topic must hang under a main subject that is linked to Peter.

        Finds trees whose top is not a main subject (a topic or question the
        resolver could not place, a hub made during sleep) and puts them under a
        subject: an existing subject when the topic is close to one, otherwise one
        named by the LLM. Returns how many trees were attached.

        ``budget`` (sleep): a new subject's collection counts against the cycle's
        CollectionBudget; once it is used up, trees wait for the next cycle unless an
        existing subject fits. ``created`` collects the collections made."""
        from memory.vector_store import BudgetExceeded
        if not cognition_config()["resolver"].get("link_category", True):
            return 0
        from memory.mind_view import _AGENT_NAMES
        from memory.tree_layout import build_forest
        # Only knowledge (topics and hubs) hangs under subjects; questions stay with their
        # topic (curious_about) and skills are not topics at all
        nodes = [n for n in self.graph.all_nodes(["entity", "abstraction"])
                 if n.get("status") != "pruned" and (n.get("label") or "").strip().lower() not in _AGENT_NAMES]
        if not nodes:
            return 0
        edges = self.graph.list_edges([n["id"] for n in nodes])
        roots, children = build_forest(nodes, edges)
        by_id = {n["id"]: n for n in nodes}
        parent = {c: p for p, kids in children.items() for c in kids}
        subjects = {n["id"]: n for n in nodes if _is_subject(n)}

        def subject_above(nid: str) -> Optional[str]:
            seen = set()
            while nid and nid not in seen:
                if nid in subjects:
                    return nid
                seen.add(nid)
                nid = parent.get(nid)
            return None

        done = 0
        filled: List[str] = []
        for r in roots:
            if r in subjects or done >= limit:
                continue
            node = by_id[r]
            label = node.get("label") or ""
            target = None
            # 1) a close known topic already under a subject -> same subject
            try:
                for hit in self.vector.search_entities(label, n_results=4):
                    if hit["node_id"] == r or hit["score"] < 0.55:
                        continue
                    target = subject_above(hit["node_id"])
                    if target:
                        break
            except Exception:
                pass
            # 2) ask the LLM, reusing an existing subject name when it fits
            if target is None:
                name = self.resolver.subject_for(label, [s["label"] for s in subjects.values()])
                if name and name.lower() != label.strip().lower():
                    try:
                        top, made = self.knowledge.ensure_node(name, kind="abstraction", turn_id=turn_id,
                                                               data={"role": "subject"}, budget=budget)
                        if made and created is not None:
                            created.append(top["collection"])
                        self.hierarchy.mark_subject(top, turn_id=turn_id)
                        top = self.graph.get_node(top["id"]) or top
                        subjects[top["id"]] = top
                        target = top["id"]
                    except BudgetExceeded:
                        logger.info(f"CognitiveMemory: no collection budget left for subject '{name}'; "
                                    f"'{label}' waits for the next cycle")
                        continue
                    except Exception as e:
                        logger.debug(f"CognitiveMemory: subject for '{label}' skipped: {e}")
                elif name and name.lower() == label.strip().lower():
                    # the root *is* a main subject ("Investing" asked about directly)
                    self.hierarchy.mark_subject(node, turn_id=turn_id)
                    subjects[r] = node
                    done += 1
                    continue
            if target and target != r:
                eid, _ = self.knowledge.link(target, r, "abstracts", turn_id=turn_id)
                if eid:
                    done += 1
                    filled.append(target)
        if filled:
            self.fill_hubs_now(filled, turn_id=turn_id)
        if done:
            emit("layout_updated", user_id=self.user_id, turn_id=turn_id, moves=[])
        return done

    # ------------------------------------------------------------------ Part B: every node is filled

    def is_hub(self, node: Dict[str, Any]) -> bool:
        if node.get("kind") == "abstraction" or _is_subject(node):
            return True
        return any(e["relation"] == "abstracts" and e["src"] == node["id"] for e in self.graph.edges_of(node["id"]))

    def record_count(self, node: Dict[str, Any]) -> int:
        ns = node.get("collection")
        return self.vector.count(ns) if ns and self.vector.has_collection(ns) else 0

    def own_record_count(self, node: Dict[str, Any]) -> int:
        """Records the node learned itself (not copies aggregated from its members)."""
        n = self.record_count(node)
        if not n:
            return 0
        return n - len(self.vector.get_where(node["collection"], {"kind": "migrated"}))

    def aggregate_hub(self, hub: Dict[str, Any], turn_id: str = "") -> int:
        """A hub/subject holds knowledge of its own: each member's most central records
        (closest to the member's centroid), copied with provenance (origin_id / origin_node),
        deduped, at most ``fill.max_hub_records``. Returns how many records were copied."""
        import numpy as np
        fcfg = cognition_config()["fill"]
        per, cap = int(fcfg.get("member_records", 3)), int(fcfg.get("max_hub_records", 40))
        if per <= 0 or not hub.get("collection"):
            return 0
        kids = [self.graph.get_node(e["dst"]) for e in sorted(self.graph.edges_of(hub["id"]), key=lambda e: e["dst"])
                if e["relation"] == "abstracts" and e["src"] == hub["id"]]
        room = cap - self.record_count(hub)
        copied = 0
        for kid in kids:
            if room <= 0:
                break
            if not kid or kid.get("kind") not in ("entity", "abstraction") or not kid.get("collection"):
                continue
            recs = [r for r in self.vector.get_all_records(kid["collection"]) if r.get("embedding")]
            if not recs:
                continue
            arr = np.asarray([r["embedding"] for r in recs], dtype=float)
            arr /= np.maximum(np.linalg.norm(arr, axis=1, keepdims=True), 1e-12)
            c = arr.mean(axis=0)
            order = np.argsort(-(arr @ c), kind="stable")
            chosen = [recs[int(i)] for i in order[: min(per, room) + per]]   # spares for already-copied ones
            n = self.knowledge.copy_records(hub, kid, chosen, limit=min(per, room))
            if n:
                copied += n
                room -= n
                emit("embedding_migrated", user_id=self.user_id, turn_id=turn_id, from_node=kid["id"],
                     to_node=hub["id"], count=n, reason="aggregate")
        if copied:
            self.knowledge.refresh_stats(hub)
        return copied

    def overview_question(self, node: Dict[str, Any]) -> str:
        label = node.get("label") or ""
        if _is_subject(node):
            return f"{label}: overview, main areas and key concepts"
        if self.is_hub(node):
            return f"{label}: overview, main areas"
        return f"What are the key facts about {label}?"

    def fill_node(self, node: Dict[str, Any], turn_id: str = "", budget: Any = None,
                  research: bool = True) -> Dict[str, Any]:
        """Give a node knowledge: hubs aggregate their members first; then, if the collection
        is still thin and research is available, research it (an overview query suited to its
        level), into this node's own collection (no new collection is made)."""
        out = {"node_id": node["id"], "aggregated": 0, "researched": 0}
        if self.is_hub(node):
            out["aggregated"] = self.aggregate_hub(node, turn_id=turn_id)
        thin = int(cognition_config()["fill"].get("thin_records", 3))
        if (research and self.research is not None and cognition_config()["research"].get("enabled", True)
                and self.own_record_count(node) < thin):
            try:
                r = self.research.research(node["label"], question=self.overview_question(node), turn_id=turn_id,
                                           budget=budget, target_node=self.graph.get_node(node["id"]) or node)
                out["researched"] = r.records_stored
            except Exception as e:
                logger.warning(f"CognitiveMemory: filling '{node.get('label')}' failed: {e}")
        return out

    def fill_hubs_now(self, hub_ids: List[str], turn_id: str = "") -> None:
        """Chat: hubs/subjects made for this turn get their members' knowledge at once; with
        fill.research_in_chat they are also researched now (in the research thread when async).
        Otherwise the next sleep's fill step researches them (it finds every thin collection)."""
        research_now = bool(cognition_config()["fill"].get("research_in_chat", False))
        if research_now and self.async_research and threading.current_thread().name != "ResearchThread":
            # chat must not wait for web research: aggregate now, research in the background
            ids = list(dict.fromkeys(hub_ids))
            for hid in ids:
                node = self.graph.get_node(hid)
                if node is not None:
                    try:
                        self.aggregate_hub(node, turn_id=turn_id)
                    except Exception as e:
                        logger.debug(f"CognitiveMemory: aggregating hub {hid} failed: {e}")

            def run() -> None:
                for hid in ids:
                    node = self.graph.get_node(hid)
                    if node is not None:
                        self.fill_node(node, turn_id=turn_id)
            th = threading.Thread(target=run, daemon=True, name="ResearchThread")
            self.last_research_thread = th
            th.start()
            return
        for hid in dict.fromkeys(hub_ids):
            node = self.graph.get_node(hid)
            if node is None:
                continue
            try:
                if research_now:
                    self.fill_node(node, turn_id=turn_id)
                elif self.record_count(node) < int(cognition_config()["fill"].get("thin_records", 3)):
                    self.aggregate_hub(node, turn_id=turn_id)
            except Exception as e:
                logger.debug(f"CognitiveMemory: filling hub {hid} failed: {e}")

    def queue_secondary(self, result: GraphRecallResult, primary_label: str, primary_id: Optional[str],
                        turn_id: str = "") -> None:
        """Unknown entities other than the main one become curiosity items linked to it;
        the sleep cycle researches them and draws the related_to edge."""
        if self.curiosity is None:
            return
        for term in result.unmatched_terms:
            if term.lower() == primary_label.lower():
                continue
            q = f"Who or what is {term}, and how is it connected to {primary_label}?"
            try:
                self.curiosity.add(q, primary_id, base_priority=0.7, origin_turn_id=turn_id, gap=True, topic=term)
            except Exception as e:
                logger.debug(f"CognitiveMemory: could not queue '{term}': {e}")

    def handle_gap(self, query: str, result: GraphRecallResult, verdict: SufficiencyVerdict,
                   turn_id: str = "") -> GapOutcome:
        topics = self.gap_topics(result, verdict)
        aspect = (getattr(result.resolution, "aspect", "") or "").strip()
        for t in topics:
            emit("gap_detected", user_id=self.user_id, turn_id=turn_id, term=t, reason=verdict.reason,
                 missing_aspects=[aspect] if aspect else [])
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
                    out = self.research.research(t, question=query, missing_aspects=[aspect] if aspect else [],
                                                 turn_id=turn_id)
                    total += out.records_stored
                    node_id = getattr(out, "node_id", None)
                    if node_id:
                        self.link_context(result, node_id, turn_id=turn_id)
                    self.queue_secondary(result, t, node_id, turn_id=turn_id)
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
            items = self.curiosity.generate(query, reply, turn_id=turn_id)
        except Exception as e:
            logger.warning(f"CognitiveMemory: curiosity generation failed: {e}")
            items = []
        try:
            self.attach_orphans(turn_id=turn_id)
        except Exception as e:
            logger.debug(f"CognitiveMemory: attach_orphans failed: {e}")
        return items
