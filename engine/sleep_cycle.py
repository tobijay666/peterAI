"""SleepCycle — the cognitive parts of Peter's sleep (spec Phase 4).

Called by EvolutionCoordinator in this order (pruning last, so nothing is
deleted before it could be consolidated or bridged; FR-14.5 still holds):

    snapshot -> usage audit (mark fading) -> consolidation -> abstraction
             -> gap scan + background research -> skills -> prune -> eviction -> journal

  A. usage audit + forgetting
       nodes below the usage floor and untouched for the window become 'fading';
       a later cycle (after the grace period) prunes them if they are still below.
  B. cross-domain abstraction
       related-but-distinct collections (centroid similarity band) are grouped;
       each group gets an abstraction node + collection, a copy of the relevant
       records (stored embeddings, nothing re-embedded) and abstracts/bridges edges.
  C. gap scan + background research
       weak nodes add curiosity items; the top items are researched.

One CollectionBudget (max 5 new collections) is shared by B and C per cycle.
Every step supports dry_run (shadow mode) and reports what it would do.
"""
from __future__ import annotations

import json
import math
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from core.cognition_config import cognition_config
from core.events import emit
from core.logger import logger
from memory.vector_store import BudgetExceeded, CollectionBudget

COGNITIVE_KINDS = ["entity", "abstraction"]
_DAY = 86400.0


def _unit(v: List[float]) -> List[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def _dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


@dataclass
class SleepReport:
    cycle_id: str
    dry_run: bool
    faded: List[str] = field(default_factory=list)
    pruned: List[str] = field(default_factory=list)
    abstractions: List[Dict[str, Any]] = field(default_factory=list)
    migrated: int = 0
    gaps_flagged: List[str] = field(default_factory=list)
    researched: List[Dict[str, Any]] = field(default_factory=list)
    collections_created: List[str] = field(default_factory=list)
    paused: bool = False
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cycle_id": self.cycle_id, "dry_run": self.dry_run, "faded": self.faded, "pruned": self.pruned,
            "abstractions": self.abstractions, "migrated": self.migrated, "gaps_flagged": self.gaps_flagged,
            "researched": self.researched, "collections_created": self.collections_created,
            "paused": self.paused, "errors": self.errors,
        }


class SleepCycle:
    def __init__(self, cognition: Any, dry_run: bool = False,
                 should_pause: Optional[Callable[[], bool]] = None,
                 cfg: Optional[Dict[str, Any]] = None) -> None:
        self.cog = cognition
        self.graph = cognition.graph
        self.vector = cognition.vector
        self.knowledge = cognition.knowledge
        self.usage = cognition.usage
        self._clock = cognition._clock
        self.dry_run = dry_run
        self.should_pause = should_pause or (lambda: False)
        self._cfg = cfg
        c = self.cfg
        self.cycle_id = uuid.uuid4().hex[:12]
        self.budget = CollectionBudget(max_new=int(c["max_new_collections"]), cycle_id=self.cycle_id)
        self.report = SleepReport(cycle_id=self.cycle_id, dry_run=dry_run)

    @property
    def cfg(self) -> Dict[str, Any]:
        return (self._cfg or cognition_config())["sleep"]

    @property
    def user_id(self) -> str:
        return self.graph.user_id

    def _phase(self, name: str) -> bool:
        """Announce a phase; returns False if the cycle should stop (user came back)."""
        if self.should_pause():
            self.report.paused = True
            return False
        emit("sleep_phase", user_id=self.user_id, name=name, cycle_id=self.cycle_id)
        return True

    # ================================================================== A. audit + prune

    def _protected_bridges(self, nodes: List[Dict[str, Any]]) -> Set[str]:
        """Nodes that are the only link between two kept nodes (cut vertices of the
        entity/abstraction subgraph). Pruning them would split the web."""
        ids = {n["id"] for n in nodes}
        adj: Dict[str, Set[str]] = {i: set() for i in ids}
        for e in self.graph.list_edges(list(ids)):
            if e["src"] in ids and e["dst"] in ids and e["src"] != e["dst"]:
                adj[e["src"]].add(e["dst"])
                adj[e["dst"]].add(e["src"])
        protected: Set[str] = set()
        for x in ids:
            nbrs = list(adj[x])
            if len(nbrs) < 2:
                continue
            start = nbrs[0]
            seen, stack = {start, x}, [start]
            while stack:
                cur = stack.pop()
                for nb in adj[cur]:
                    if nb not in seen:
                        seen.add(nb)
                        stack.append(nb)
            if any(nb not in seen for nb in nbrs[1:]):
                protected.add(x)
        return protected

    def _prune_candidates(self) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, float]]]:
        p = self.cfg["prune"]
        now = self._clock.now()
        scored = self.usage.recompute_all(COGNITIVE_KINDS, persist=not self.dry_run)
        nodes = [n for n in self.graph.all_nodes(COGNITIVE_KINDS) if n.get("status") != "pruned"]
        protected = self._protected_bridges(nodes)
        out = []
        for n in nodes:
            if n.get("pinned") or n["id"] in protected:
                continue
            last = n.get("last_accessed")
            if last is None:
                last = n.get("created_at") if n.get("created_at") is not None else now
            if now - float(last) < float(p["window_days"]) * _DAY:
                continue
            s = scored.get(n["id"], {"score": 0.0, "percentile": 0.0})
            if s["percentile"] < float(p["percentile_floor"]) or s["score"] < float(p.get("score_floor", 0.05)):
                out.append(n)
        return out, scored

    def audit(self) -> None:
        """A1: mark low-usage, long-untouched nodes as fading (no deletion here)."""
        if not self._phase("usage_audit"):
            return
        candidates, scored = self._prune_candidates()
        now = self._clock.now()
        for n in candidates:
            if n.get("status") == "fading":
                continue
            self.report.faded.append(n["id"])
            if self.dry_run:
                continue
            self.graph.set_node_fields(n["id"], status="fading", fading_since=now)
            style = self.usage.style_of({**n, "status": "fading"}, scored.get(n["id"], {}).get("percentile", 0.0))
            emit("node_fading", user_id=self.user_id, node_id=n["id"], label=n.get("label"),
                 usage_score=round(scored.get(n["id"], {}).get("score", 0.0), 4), style=style)

    def prune(self) -> None:
        """A2: prune nodes still below the floor after the grace period; clean orphans."""
        if not self._phase("prune"):
            return
        grace = float(self.cfg["prune"]["grace_days"]) * _DAY
        now = self._clock.now()
        candidates, _ = self._prune_candidates()
        cand_ids = {n["id"] for n in candidates}
        for n in self.graph.all_nodes(COGNITIVE_KINDS):
            if n.get("status") != "fading":
                continue
            since = n.get("fading_since")
            if since is None:
                since = now
            if now - float(since) < grace or n["id"] not in cand_ids:
                continue
            self.report.pruned.append(n["id"])
            if not self.dry_run:
                self.knowledge.remove_node(n["id"])
        if self.dry_run:
            return
        # Orphans: source nodes nothing cites any more; collections whose node is gone
        for s in self.graph.get_nodes_by_kind("source"):
            if self.graph.degree(s["id"]) == 0:
                self.graph.delete_node(s["id"])
        for c in self.graph.list_collections():
            if c.get("node_id") and self.graph.get_node(c["node_id"]) is None:
                self.vector.drop_collection(c["name"])
                self.graph.unregister_collection(c["name"])
        registered = {c["name"] for c in self.graph.list_collections()}
        for ns in self.vector.list_knowledge_namespaces():
            if ns not in registered and self.vector.count(ns) == 0:
                self.vector.drop_collection(ns)

    # ================================================================== B. abstraction

    def _groups(self) -> List[Dict[str, Any]]:
        a = self.cfg["abstraction"]
        lo, hi = float(a["min_similarity"]), float(a["max_similarity"])
        members = [n for n in self.graph.get_nodes_by_kind("entity")
                   if n.get("status") == "active" and n.get("collection")]
        cents: Dict[str, List[float]] = {}
        for n in members:
            c = self.vector.centroid(n["collection"])
            if c:
                cents[n["id"]] = c
        ids = sorted(cents)
        parent = {i: i for i in ids}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        pairs: List[Tuple[str, str, float]] = []
        for i, x in enumerate(ids):
            for y in ids[i + 1:]:
                sim = _dot(cents[x], cents[y])
                if lo <= sim <= hi:
                    pairs.append((x, y, sim))
                    parent[find(x)] = find(y)
        comps: Dict[str, List[str]] = {}
        for i in ids:
            comps.setdefault(find(i), []).append(i)
        node_by_id = {n["id"]: n for n in members}
        groups = []
        for comp in comps.values():
            if len(comp) < 2:
                continue
            comp_set = set(comp)
            # Already covered: one abstraction already abstracts every member
            covered = False
            for absn in self.graph.get_nodes_by_kind("abstraction"):
                kids = {e["dst"] for e in self.graph.edges_of(absn["id"])
                        if e["relation"] == "abstracts" and e["src"] == absn["id"]}
                if comp_set <= kids:
                    covered = True
                    break
            if covered:
                continue
            nodes = [node_by_id[i] for i in sorted(comp)]
            usage = sum(float(n.get("usage_score") or 0.0) for n in nodes)
            group_c = _unit([sum(vals) for vals in zip(*[cents[i] for i in comp])])
            groups.append({
                "members": nodes, "centroid": group_c,
                "pairs": [(x, y, s) for x, y, s in pairs if x in comp_set and y in comp_set],
                "rank": len(comp) * (1.0 + usage),
            })
        groups.sort(key=lambda g: (-g["rank"], [m["id"] for m in g["members"]]))
        return groups

    def _name_group(self, labels: List[str]) -> str:
        router = self.cog.router
        if router is not None:
            try:
                from core.interfaces import Message
                res = router.complete(
                    messages=[
                        Message(role="system", content=(
                            "Name the single umbrella theme that connects these topics in 1 to 4 words "
                            "(e.g. 'wedding cultures'). Reply with the name only."
                        )),
                        Message(role="user", content=", ".join(labels)),
                    ],
                    temperature=0.2, max_tokens=600,
                )
                name = (res.get("content") or "").strip().strip('."\'').splitlines()[0][:60] if res.get("content") else ""
                if name and 1 <= len(name.split()) <= 6:
                    return name
            except Exception as e:
                logger.debug(f"SleepCycle: naming failed ({e})")
        return " & ".join(labels[:2]) + (" themes" if len(labels) <= 2 else " and related")

    def abstract(self) -> None:
        if not self._phase("abstraction"):
            return
        a = self.cfg["abstraction"]
        floor = float(a["migrate_min_similarity"])
        per_member = int(a["max_records_per_member"])
        for g in self._groups():
            labels = [m["label"] for m in g["members"]]
            name = self._name_group(labels)
            existing = self.knowledge.find_node(name)
            plan = {"name": name, "members": [m["id"] for m in g["members"]], "node_id": None,
                    "migrated": 0, "reused": bool(existing)}
            if existing is not None and existing["kind"] != "abstraction":
                continue  # the name is already an entity; don't hijack it
            if self.dry_run:
                self.report.abstractions.append(plan)
                continue
            if existing is None:
                if not self.budget.can_create():
                    break
                try:
                    node, created = self.knowledge.ensure_node(name, kind="abstraction", budget=self.budget,
                                                              cycle_id=self.cycle_id, aliases=[])
                except BudgetExceeded:
                    break
                if created:
                    self.report.collections_created.append(node["collection"])
            else:
                node = existing
            plan["node_id"] = node["id"]
            # Embedding migration: copy (D1) the records that sit close to the group's centroid
            already = {r["metadata"].get("origin_id") for r in self.vector.get_all_records(node["collection"])}
            for m in g["members"]:
                recs = self.vector.get_all_records(m["collection"])
                ranked = sorted(((_dot(_unit(r["embedding"]), g["centroid"]), r) for r in recs if r.get("embedding")),
                                key=lambda t: -t[0])
                copied = 0
                for sim, r in ranked:
                    if copied >= per_member or sim < floor:
                        break
                    if r["id"] in already:
                        continue
                    meta = {k: v for k, v in (r["metadata"] or {}).items()
                            if k not in ("access_count", "last_accessed", "created_at", "agent", "node_id")}
                    meta.update({"origin_collection": m["collection"], "origin_id": r["id"], "origin_node": m["id"],
                                 "kind": "migrated"})
                    self.knowledge.add_record(node, r["document"], meta, embedding=r["embedding"])
                    already.add(r["id"])
                    copied += 1
                if copied:
                    plan["migrated"] += copied
                    self.report.migrated += copied
                    emit("embedding_migrated", user_id=self.user_id, from_node=m["id"], to_node=node["id"],
                         count=copied, cycle_id=self.cycle_id)
            for m in g["members"]:
                self.knowledge.link(node["id"], m["id"], "abstracts")
            for x, y, _ in g["pairs"]:
                self.knowledge.link(x, y, "bridges")
            self.knowledge.refresh_stats(node)
            self.report.abstractions.append(plan)

    # ================================================================== C. gaps + research

    def scan_gaps(self) -> None:
        if not self._phase("gap_scan"):
            return
        gcfg = self.cfg["gaps"]
        curiosity = self.cog.curiosity
        for n in self.graph.all_nodes(COGNITIVE_KINDS):
            if n.get("status") != "active":
                continue
            coll = self.graph.get_collection(n.get("collection") or "") or {}
            degree = sum(1 for e in self.graph.edges_of(n["id"]) if e["relation"] != "curious_about")
            reasons = []
            if degree < int(gcfg["min_degree"]):
                reasons.append("few_edges")
            if int(coll.get("record_count") or 0) < int(gcfg["min_records"]):
                reasons.append("few_records")
            if float(n.get("confidence") if n.get("confidence") is not None else 0.5) < float(gcfg["min_confidence"]):
                reasons.append("low_confidence")
            if coll.get("created_cycle_id") == self.cycle_id:
                reasons.append("new_this_cycle")
            if not reasons:
                continue
            self.report.gaps_flagged.append(n["id"])
            if not self.dry_run and curiosity is not None:
                curiosity.add_gap_item(n, ",".join(reasons))

    def research(self) -> None:
        if not self._phase("background_research"):
            return
        curiosity, research = self.cog.curiosity, self.cog.research
        if curiosity is None or research is None:
            return
        k = int(self.cfg["research_items_per_cycle"])
        if self.dry_run:
            self.report.researched = [{"question": it["question"], "dry_run": True} for it in curiosity.list(limit=k)]
            return
        searched_topics: Set[str] = set()
        for item in curiosity.pop_top(k):
            if self.should_pause():
                curiosity.complete(item["id"], ok=False)
                self.report.paused = True
                break
            topic = curiosity.topic_for(item)
            first = topic.lower() not in searched_topics   # search the bare topic name once per cycle
            searched_topics.add(topic.lower())
            try:
                out = research.research(topic, question=item["question"], budget=self.budget,
                                        include_topic=first)
            except Exception as e:
                self.report.errors.append(f"research '{topic}': {e}")
                curiosity.complete(item["id"], ok=False)
                continue
            ok = out.records_stored > 0
            curiosity.complete(item["id"], ok=ok)
            origin = item.get("topic_node_id")
            if ok and out.node_id and origin and origin != out.node_id and self.graph.get_node(origin):
                # "Oscar Wilde" researched from a Dorian Gray question -> related_to edge
                self.knowledge.link(out.node_id, origin, "related_to")
            if ok and out.node_id and getattr(self.cog, "hierarchy", None) is not None:
                # now that it has facts, link it to topics close in meaning
                try:
                    n = self.graph.get_node(out.node_id)
                    if n:
                        self.cog.hierarchy.link_similar(n)
                except Exception as e:
                    self.report.errors.append(f"link_similar: {e}")
            if out.created_node and out.node_id:
                node = self.graph.get_node(out.node_id)
                if node and node.get("collection"):
                    self.report.collections_created.append(node["collection"])
            self.report.researched.append({"question": item["question"], "topic": topic, "node_id": out.node_id,
                                           "records": out.records_stored, "deferred": out.deferred})
        # Whatever sleep learned must hang under a main subject linked to Peter
        if not self.dry_run and hasattr(self.cog, "attach_orphans"):
            try:
                self.cog.attach_orphans()
            except Exception as e:
                self.report.errors.append(f"attach_orphans: {e}")

    # ================================================================== layout

    def relayout(self) -> None:
        """Refit the 3-D projection now that collections changed; emit the moves to animate."""
        layout = getattr(self.cog, "layout", None)
        if self.dry_run or layout is None or self.report.paused:
            return
        moves = layout.refit()
        if moves:
            emit("layout_updated", user_id=self.user_id, cycle_id=self.cycle_id, moves=moves)
