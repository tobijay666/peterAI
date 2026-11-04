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


def _symbol_signature(label: str) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """The digit runs and symbol characters of a label. Two labels that differ in these are
    different topics however close their words: C / C++ / C# / "C ++", Python 2 / Python 3,
    Windows 10 / Windows 11."""
    import re
    low = (label or "").lower()
    digits = tuple(sorted(re.findall(r"\d+", low)))
    syms = tuple(sorted(c for c in low if not c.isalnum() and not c.isspace()))
    return digits, syms


def _symbol_compatible(a_label: str, b_label: str) -> bool:
    return _symbol_signature(a_label) == _symbol_signature(b_label)


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
    repaired: int = 0
    merged: List[Dict[str, Any]] = field(default_factory=list)
    filled: List[Dict[str, Any]] = field(default_factory=list)
    copies_removed: int = 0
    copies_updated: int = 0
    counts_fixed: int = 0
    vectors: Dict[str, int] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cycle_id": self.cycle_id, "dry_run": self.dry_run, "faded": self.faded, "pruned": self.pruned,
            "abstractions": self.abstractions, "migrated": self.migrated, "gaps_flagged": self.gaps_flagged,
            "researched": self.researched, "collections_created": self.collections_created,
            "paused": self.paused, "repaired": self.repaired, "merged": self.merged, "filled": self.filled, "copies_removed": self.copies_removed,
            "copies_updated": self.copies_updated, "counts_fixed": self.counts_fixed, "vectors": self.vectors,
            "errors": self.errors,
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

    def _structural(self, nodes: List[Dict[str, Any]]) -> Set[str]:
        """Never pruned: main subjects (linked to Peter) and any node that still abstracts a
        live topic (a hub is only as dead as its last child)."""
        from memory.hierarchy import is_subject
        live = {n["id"] for n in nodes}
        keep = {n["id"] for n in nodes if is_subject(n)}
        for e in self.graph.list_edges(list(live)):
            if e["relation"] == "abstracts" and e["src"] in live and e["dst"] in live:
                keep.add(e["src"])
        return keep

    def _prune_candidates(self) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, float]]]:
        p = self.cfg["prune"]
        now = self._clock.now()
        scored = self.usage.recompute_all(COGNITIVE_KINDS, persist=not self.dry_run)
        nodes = [n for n in self.graph.all_nodes(COGNITIVE_KINDS) if n.get("status") != "pruned"]
        protected = self._protected_bridges(nodes) | self._structural(nodes)
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
        # question nodes whose queue item is gone or finished (older graphs left them behind)
        live = {f"cur_{r['id']}" for r in self._curiosity_rows("WHERE status = 'pending' OR status = 'researching'")}
        for c in self.graph.get_nodes_by_kind("curiosity"):
            if c["id"] not in live and self.graph.delete_node(c["id"]):
                emit("node_pruned", user_id=self.user_id, node_id=c["id"], label=c.get("label"), collection=None)
        for c in self.graph.list_collections():
            if c.get("node_id") and self.graph.get_node(c["node_id"]) is None:
                self.vector.drop_collection(c["name"])
                self.graph.unregister_collection(c["name"])
        registered = {c["name"] for c in self.graph.list_collections()}
        for ns in self.vector.list_knowledge_namespaces():
            if ns not in registered and self.vector.count(ns) == 0:
                self.vector.drop_collection(ns)

    def _curiosity_rows(self, where: str) -> List[Dict[str, Any]]:
        with self.graph._get_connection() as conn:
            return [dict(r) for r in conn.execute(f"SELECT * FROM curiosity_queue {where};").fetchall()]

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
        groups = self._groups()
        with self.knowledge.batch():      # D3: one recount at the end, weights/links done in maintain
            self._abstract_groups(groups, floor, per_member)

    def _abstract_groups(self, groups: List[Dict[str, Any]], floor: float, per_member: int) -> None:
        for g in groups:
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
            # Embedding migration: copy (D1) the records that sit close to the group's centroid,
            # with provenance, never twice (same origin or same text) - KnowledgeStore.copy_records
            for m in g["members"]:
                recs = self.vector.get_all_records(m["collection"])
                ranked = sorted(((_dot(_unit(r["embedding"]), g["centroid"]), r) for r in recs if r.get("embedding")),
                                key=lambda t: -t[0])
                chosen = [r for sim, r in ranked if sim >= floor][:per_member]
                copied = self.knowledge.copy_records(node, m, chosen)
                if copied:
                    plan["migrated"] += copied
                    self.report.migrated += copied
                    emit("embedding_migrated", user_id=self.user_id, from_node=m["id"], to_node=node["id"],
                         count=copied, cycle_id=self.cycle_id)
            # Bulk-create the hub->member (abstracts) and member<->member (bridges) edges in one
            # transaction (D3): the hub is fresh and the members are entities, so there is no cycle
            # or subject to guard and no per-edge validation query is needed. Weights and record
            # links are filled by the end-of-sleep refresh_all, like every other sleep edge.
            self._bulk_link(node, g["members"], g["pairs"])
            self.knowledge.refresh_stats(node)
            self.report.abstractions.append(plan)

    def _bulk_link(self, hub: Dict[str, Any], members: List[Dict[str, Any]],
                   pairs: List[Tuple[str, str, float]]) -> None:
        member_ids = [m["id"] for m in members]
        scope = [hub["id"]] + member_ids
        existing = self.graph.list_edges(scope)
        has_parent = {m: False for m in member_ids}        # does the member already have a primary hub?
        present: Set[Tuple[str, str, str]] = set()
        for e in existing:
            present.add((e["src"], e["dst"], e["relation"]))
            if e["relation"] == "abstracts" and e["dst"] in has_parent and (e.get("data") or {}).get("primary"):
                has_parent[e["dst"]] = True
        triples: List[Tuple[str, str, str, Any]] = []
        drawn: List[Tuple[str, str, str, Any]] = []
        for mid in member_ids:
            if (hub["id"], mid, "abstracts") in present or hub["id"] == mid:
                continue
            data = {"primary": True} if not has_parent[mid] else None
            has_parent[mid] = True
            triples.append((hub["id"], mid, "abstracts", data))
            drawn.append((hub["id"], mid, "abstracts", data))
            present.add((hub["id"], mid, "abstracts"))
        for x, y, _ in pairs:
            if x == y:
                continue
            if (x, y, "bridges") in present or (y, x, "bridges") in present:
                continue
            # a hub and its own member are joined by abstracts, never also by a symmetric bridge
            if (hub["id"], x, "abstracts") in present or (hub["id"], y, "abstracts") in present:
                if x in member_ids and y in member_ids:
                    pass   # both are members: a bridge between two members is fine
            triples.append((x, y, "bridges", None))
            drawn.append((x, y, "bridges", None))
            present.add((x, y, "bridges"))
        eids = self.graph.link_many(triples)
        for eid, (src, dst, rel, data) in zip(eids, drawn):
            emit("edge_drawn", user_id=self.user_id, edge_id=eid, src=src, dst=dst, relation=rel,
                 weight=None, primary=bool((data or {}).get("primary")))

    # ================================================================== B2. near-duplicate merge

    def merge_duplicates(self) -> None:
        """Two topics that are the same thing under two names ('Colombo Stock Exchange' /
        'Stock Exchange of Colombo') are merged: label similarity >= merge.label_similarity
        and, when both have facts, centroid similarity >= merge.centroid_similarity."""
        if not self._phase("merge"):
            return
        from memory.hierarchy import is_subject
        m = self.cfg.get("merge", {}) or {}
        lab_min, cen_min = float(m.get("label_similarity", 0.92)), float(m.get("centroid_similarity", 0.9))
        cap = int(m.get("max_per_cycle", 5))
        nodes = [n for n in self.graph.all_nodes(COGNITIVE_KINDS)
                 if n.get("status") != "pruned" and n.get("collection") and not is_subject(n)]
        if len(nodes) < 2 or cap <= 0:
            return
        import numpy as np
        L = np.asarray(self.vector.embed([n.get("label") or "" for n in nodes]), dtype=float)
        L /= np.maximum(np.linalg.norm(L, axis=1, keepdims=True), 1e-12)
        S = L @ L.T
        cents = {n["id"]: self.vector.centroid(n["collection"]) for n in nodes}
        gone: Set[str] = set()
        pairs = sorted(((float(S[i, j]), i, j) for i in range(len(nodes)) for j in range(i + 1, len(nodes))
                        if S[i, j] >= lab_min and nodes[i]["kind"] == nodes[j]["kind"]), key=lambda t: -t[0])
        done = 0
        for sim, i, j in pairs:
            a, b = nodes[i], nodes[j]
            if done >= cap or a["id"] in gone or b["id"] in gone:
                continue
            # D1/D8: never fold topics that differ only by symbols or digits, even at label
            # similarity 1.0 under a bag-of-words embedder (C / C++ / C# / "C ++", Python 2/3)
            if not _symbol_compatible(a.get("label") or "", b.get("label") or ""):
                continue
            ca, cb = cents.get(a["id"]), cents.get(b["id"])
            # D1: require real centroid evidence on BOTH sides — never merge on the label alone
            # (an empty "C#" must not fold into "C"); the vectors must actually agree
            if ca is None or cb is None or _dot(ca, cb) < cen_min:
                continue
            if self.knowledge.is_descendant(a["id"], of=b["id"]) or self.knowledge.is_descendant(b["id"], of=a["id"]):
                continue                          # a hub and its own member are not duplicates
            def rank(n):
                return (not n.get("pinned"), -self.vector.count(n["collection"]), -int(n.get("access_count") or 0),
                        float(n.get("created_at") or 0.0), n["id"])
            keeper, loser = sorted((a, b), key=rank)
            if loser.get("pinned"):
                continue
            entry = {"keeper": keeper["id"], "merged": loser["id"], "label_similarity": round(sim, 4)}
            if not self.dry_run:
                try:
                    entry.update(self.knowledge.merge_nodes(keeper, loser, dedupe_similarity=float(
                        cognition_config()["research"].get("corroboration_similarity", 0.9))))
                except Exception as e:
                    self.report.errors.append(f"merge {loser['id']} -> {keeper['id']}: {e}")
                    continue
            gone.add(loser["id"])
            self.report.merged.append(entry)
            done += 1

    # ================================================================== B3. fill empty / thin hubs

    def fill(self) -> None:
        """Every node is a vector store with knowledge (Part B). Hubs and subjects whose
        collection is empty or thin first take their members' most central records, then up to
        ``fill.max_per_cycle`` of them are researched with an overview query into their own
        collection. Thin topics (and hubs over the cap) are queued by the gap scan that follows,
        so nothing stays empty silently."""
        if not self._phase("fill"):
            return
        fcfg = cognition_config()["fill"]
        thin, cap = int(fcfg.get("thin_records", 3)), int(fcfg.get("max_per_cycle", 3))
        from memory.hierarchy import is_subject
        hubs = []
        for n in self.graph.all_nodes(COGNITIVE_KINDS):
            if n.get("status") == "pruned" or not n.get("collection") or not self.cog.is_hub(n):
                continue
            count = self.cog.own_record_count(n)       # knowledge of its own, copies not counted
            if count < thin:
                hubs.append((count, n))
        # subjects first, then the emptiest, then the most used
        hubs.sort(key=lambda t: (not is_subject(t[1]), t[0], -float(t[1].get("usage_score") or 0.0), t[1]["id"]))
        if self.dry_run:
            self.report.filled.extend({"node_id": n["id"], "before": c, "aggregated": 0, "researched": 0}
                                      for c, n in hubs)
            return
        with self.knowledge.batch():      # D3: bulk record inserts recount once; links done in maintain
            self._fill_hubs(hubs, cap)

    def _fill_hubs(self, hubs: List[Tuple[int, Dict[str, Any]]], cap: int) -> None:
        researched = 0
        for count, n in hubs:
            if self.should_pause():
                self.report.paused = True
                return
            entry = {"node_id": n["id"], "before": count, "aggregated": 0, "researched": 0}
            do_research = researched < cap and self.cog.research is not None
            try:
                r = self.cog.fill_node(n, budget=self.budget, research=do_research)
            except Exception as e:
                self.report.errors.append(f"fill {n['id']}: {e}")
                continue
            if do_research:
                researched += 1
            entry.update(aggregated=r["aggregated"], researched=r["researched"], after=self.cog.record_count(n))
            self.report.filled.append(entry)

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
            if out.deferred:
                # budget used up: the item waits for the next cycle (never filed under a wrong node)
                curiosity.defer(item["id"])
                self.report.researched.append({"question": item["question"], "topic": topic, "node_id": None,
                                               "records": 0, "deferred": True})
                continue
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
                self.cog.attach_orphans(budget=self.budget, created=self.report.collections_created)
            except Exception as e:
                self.report.errors.append(f"attach_orphans: {e}")

    # ================================================================== D. maintenance

    def maintain(self) -> None:
        """Keep the stored graph consistent after everything else changed it:
        hierarchy repair, stale hub copies, registry record counts."""
        if not self._phase("maintenance") or self.dry_run:
            return
        hier = getattr(self.cog, "hierarchy", None)
        if hier is not None:
            try:
                self.report.repaired += hier.repair()
            except Exception as e:
                self.report.errors.append(f"repair: {e}")
        for n in self.graph.get_nodes_by_kind("abstraction"):
            if n.get("status") == "pruned" or not n.get("collection"):
                continue
            try:
                r = self.knowledge.refresh_copies(n)
                self.report.copies_removed += r["removed"]
                self.report.copies_updated += r["updated"]
            except Exception as e:
                self.report.errors.append(f"refresh_copies {n['id']}: {e}")
        self.reconcile_counts()
        vg = getattr(self.cog, "vgraph", None)
        if vg is not None:
            try:      # edge weights, record-level vector links, Peter's vector, similar_to kNN
                self.report.vectors = vg.refresh_all()
            except Exception as e:
                self.report.errors.append(f"vectors: {e}")

    def reconcile_counts(self) -> int:
        """Registry record_count == the collection's real size (audit item 12). Returns fixes."""
        fixed = 0
        for c in self.graph.list_collections():
            real = self.vector.count(c["name"])
            if int(c.get("record_count") or 0) != real:
                self.graph.update_collection_stats(c["name"], record_count=real)
                fixed += 1
        self.report.counts_fixed += fixed
        return fixed

    # ================================================================== layout

    def relayout(self) -> None:
        """Refit the 3-D projection now that collections changed; emit the moves to animate."""
        layout = getattr(self.cog, "layout", None)
        if self.dry_run or layout is None or self.report.paused:
            return
        moves = layout.refit()
        if moves:
            emit("layout_updated", user_id=self.user_id, cycle_id=self.cycle_id, moves=moves)
