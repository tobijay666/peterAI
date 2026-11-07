"""Curiosity queue (spec Phase 3).

After each turn Peter proposes 3–5 adjacent questions it would like to learn
about, ranks them and stores them durably (curiosity_queue table in the user's
graph DB, so it is per-user, snapshotted and survives restarts). Each pending
question is also a faint 'curiosity' node linked 'curious_about' to its topic,
so it shows up in the mind view.

    priority = w_usage * topic usage percentile
             + w_novelty * (1 - similarity to the nearest existing node)
             + w_gap * (1 if it came from a knowledge gap)
             + w_recency * exp(-age_days / 7)      (applied when ranking)

The sleep cycle pops the top items and researches them.
"""
from __future__ import annotations

import math
import re
import uuid
from typing import Any, Dict, List, Optional

from core.clock import Clock
from core.cognition_config import cognition_config
from core.events import emit
from core.logger import logger
from memory.entity_resolver import EntityResolver
from memory.knowledge import KnowledgeStore

_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")
MAX_ATTEMPTS = 3


def _cos(a: List[float], b: List[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    da = math.sqrt(sum(x * x for x in a)) or 1.0
    db = math.sqrt(sum(y * y for y in b)) or 1.0
    return num / (da * db)


class CuriosityEngine:
    def __init__(self, knowledge: KnowledgeStore, resolver: Optional[EntityResolver] = None,
                 router: Any = None, clock: Optional[Clock] = None,
                 cfg: Optional[Dict[str, Any]] = None) -> None:
        self.knowledge = knowledge
        self.graph = knowledge.graph
        self.vector = knowledge.vector
        self.resolver = resolver or EntityResolver(knowledge.vector, knowledge.graph, router=router)
        self.router = router
        self._clock = clock or knowledge._clock
        self._cfg = cfg

    @property
    def cfg(self) -> Dict[str, Any]:
        return (self._cfg or cognition_config())["curiosity"]

    @property
    def user_id(self) -> str:
        return self.graph.user_id

    # ------------------------------------------------------------------ storage

    def _rows(self, where: str = "", args: tuple = ()) -> List[Dict[str, Any]]:
        with self.graph._get_connection() as conn:
            rows = conn.execute(f"SELECT * FROM curiosity_queue {where};", args).fetchall()
            return [dict(r) for r in rows]

    def effective_priority(self, item: Dict[str, Any]) -> float:
        w = self.cfg["weights"]
        age_days = max(0.0, (self._clock.now() - float(item["created_at"])) / 86400.0)
        return float(item["priority"]) + float(w["recency"]) * math.exp(-age_days / 7.0)

    def list(self, status: str = "pending", limit: int = 50) -> List[Dict[str, Any]]:
        items = self._rows("WHERE status = ?", (status,))
        for it in items:
            it["effective_priority"] = round(self.effective_priority(it), 4)
        items.sort(key=lambda it: (-it["effective_priority"], it["created_at"], it["id"]))
        return items[:limit]

    def get(self, item_id: str) -> Optional[Dict[str, Any]]:
        rows = self._rows("WHERE id = ?", (item_id,))
        return rows[0] if rows else None

    def _set(self, item_id: str, **fields: Any) -> None:
        fields["updated_at"] = self._clock.now()
        sets = ", ".join(f"{k} = ?" for k in fields)
        with self.graph._get_connection() as conn:
            conn.execute(f"UPDATE curiosity_queue SET {sets} WHERE id = ?;", (*fields.values(), item_id))
            conn.commit()

    def _drop_node(self, item_id: str) -> None:
        self.graph.delete_node(f"cur_{item_id}")

    # ------------------------------------------------------------------ add

    def add(self, question: str, topic_node_id: Optional[str] = None, base_priority: float = 0.0,
            origin_turn_id: str = "", gap: bool = False, topic: Optional[str] = None) -> Dict[str, Any]:
        now = self._clock.now()
        item_id = uuid.uuid4().hex[:16]
        with self.graph._get_connection() as conn:
            conn.execute(
                "INSERT INTO curiosity_queue (id, question, topic_node_id, topic, priority, status, origin_turn_id,"
                " attempts, gap, created_at, updated_at) VALUES (?, ?, ?, ?, ?, 'pending', ?, 0, ?, ?, ?);",
                (item_id, question, topic_node_id, topic, float(base_priority), origin_turn_id,
                 1 if gap else 0, now, now),
            )
            conn.commit()
        node_id = f"cur_{item_id}"
        self.graph.upsert_node(node_id, kind="curiosity", label=question[:200],
                               data={"curiosity_id": item_id}, importance=0.2)
        edge_id = None
        if topic_node_id and self.graph.get_node(topic_node_id):
            edge_id, _ = self.graph.link_once(node_id, topic_node_id, "curious_about")
        item = self.get(item_id)
        # edge_id: the stored curious_about edge, so the mind view draws exactly that edge
        emit("curiosity_added", user_id=self.user_id, turn_id=origin_turn_id, id=item_id, question=question,
             topic_node_id=topic_node_id, priority=round(float(base_priority), 4), node_id=node_id, edge_id=edge_id)
        self._enforce_cap()
        return item

    def _enforce_cap(self) -> None:
        cap = int(self.cfg["max_pending"])
        pending = self.list("pending", limit=10_000)
        for it in pending[cap:]:
            self._set(it["id"], status="dropped")
            self._drop_node(it["id"])

    # ------------------------------------------------------------------ generate

    def propose(self, query: str, reply: str, labels: List[str]) -> List[str]:
        n = int(self.cfg["per_turn"])
        if self.router is not None:
            try:
                from core.interfaces import Message
                res = self.router.complete(
                    messages=[
                        Message(role="system", content=(
                            f"You are Peter's curiosity. Given a conversation and the topics involved, propose {n} "
                            "short, specific questions about ADJACENT themes the user did not ask yet but would "
                            "plausibly want next. One question per line, each ending with '?'. No numbering."
                        )),
                        Message(role="user", content=(
                            f"Topics: {', '.join(labels) or 'none'}\nUser asked: {query[:500]}\n"
                            f"Peter answered: {reply[:800]}"
                        )),
                    ],
                    temperature=0.7, max_tokens=1200,
                )
                qs = [_BULLET.sub("", l).strip() for l in (res.get("content") or "").splitlines()]
                qs = [q for q in qs if q.endswith("?") and len(q) > 10]
                if qs:
                    return qs[:n]
            except Exception as e:
                logger.debug(f"CuriosityEngine: LLM proposal failed ({e}); templates")
        out: List[str] = []
        for label in labels or []:
            out += [f"What is the history and origin of {label}?",
                    f"How is {label} used or practised today?",
                    f"What are common misconceptions about {label}?",
                    f"What safety or ethical issues are linked to {label}?"]
        return out[:n]

    def _is_duplicate(self, emb: List[float], others: List[List[float]]) -> bool:
        thr = float(self.cfg["dedupe_similarity"])
        return any(_cos(emb, o) >= thr for o in others)

    def generate(self, query: str, reply: str = "", turn_id: str = "") -> List[Dict[str, Any]]:
        res = self.resolver.resolve(query)
        topics = res.nodes
        labels = [n["label"] for n in topics] + list(res.unmatched)
        gap_terms = set(t.lower() for t in res.unmatched)
        questions = self.propose(query, reply, labels)
        if not questions:
            return []

        pending = self._rows("WHERE status IN ('pending', 'researching', 'dismissed')")
        pending_embs = self.vector.embed([p["question"] for p in pending]) if pending else []
        new_embs = self.vector.embed(questions)
        thr = float(self.cfg["dedupe_similarity"])
        w = self.cfg["weights"]
        pct = {k: v["percentile"] for k, v in self.knowledge.usage.recompute_all(persist=False).items()}

        added: List[Dict[str, Any]] = []
        accepted_embs: List[List[float]] = []
        for q, emb in zip(questions, new_embs):
            if self._is_duplicate(emb, pending_embs) or self._is_duplicate(emb, accepted_embs):
                continue
            hits = self.vector.search_entities(q, n_results=1)
            if hits and hits[0]["score"] >= thr:
                continue  # an existing collection already covers it
            topic = topics[0] if topics else None
            for t in topics:
                if t["label"].lower() in q.lower():
                    topic = t
                    break
            gap_hit = next((t for t in res.unmatched if t.lower() in q.lower()), None)
            is_gap = gap_hit is not None or (topic is None and bool(gap_terms))
            topic_text = topic["label"] if topic is not None and gap_hit is None else (
                gap_hit or (res.unmatched[0] if res.unmatched else None))
            usage_p = pct.get(topic["id"], 0.5) if topic else 0.5
            novelty = 1.0 - max(0.0, hits[0]["score"]) if hits else 1.0
            base = float(w["usage"]) * usage_p + float(w["novelty"]) * novelty + float(w["gap"]) * (1.0 if is_gap else 0.0)
            added.append(self.add(q, topic["id"] if topic and gap_hit is None else None, base, turn_id,
                                  gap=is_gap, topic=topic_text))
            accepted_embs.append(emb)
        return added

    def add_gap_item(self, node: Dict[str, Any], reason: str, turn_id: str = "") -> Dict[str, Any]:
        """Sleep gap scan: queue (or boost) a question about a weakly supported node."""
        for it in self._rows("WHERE status = 'pending' AND topic_node_id = ?", (node["id"],)):
            if it["gap"]:
                self._set(it["id"], priority=float(it["priority"]) + 0.1)
                return self.get(it["id"])
        q = f"What are the key facts about {node['label']}?"
        w = self.cfg["weights"]
        return self.add(q, node["id"], float(w["gap"]) + float(w["novelty"]) * 0.5, turn_id, gap=True,
                        topic=node["label"])

    # ------------------------------------------------------------------ lifecycle

    def boost(self, item_id: str, delta: float = 0.25) -> Optional[Dict[str, Any]]:
        it = self.get(item_id)
        if not it:
            return None
        self._set(item_id, priority=float(it["priority"]) + delta)
        return self.get(item_id)

    def dismiss(self, item_id: str) -> Optional[Dict[str, Any]]:
        """Dismiss one question; near-duplicates lose half their priority, and new
        twins are blocked because dismissed questions stay in the dedupe set."""
        it = self.get(item_id)
        if not it:
            return None
        self._set(item_id, status="dismissed")
        self._drop_node(item_id)
        others = self._rows("WHERE status = 'pending'")
        if others:
            emb = self.vector.embed([it["question"]])[0]
            embs = self.vector.embed([o["question"] for o in others])
            for o, e in zip(others, embs):
                if _cos(emb, e) >= float(self.cfg["dedupe_similarity"]) * 0.8:
                    self._set(o["id"], priority=float(o["priority"]) * 0.5)
        return self.get(item_id)

    def pop_top(self, k: int) -> List[Dict[str, Any]]:
        items = self.list("pending", limit=k)
        for it in items:
            self._set(it["id"], status="researching", attempts=int(it["attempts"]) + 1)
        return [self.get(it["id"]) for it in items]

    def complete(self, item_id: str, ok: bool) -> None:
        it = self.get(item_id)
        if not it:
            return
        if ok:
            self._set(item_id, status="done")
            self._drop_node(item_id)
        elif int(it["attempts"]) >= MAX_ATTEMPTS:
            self._set(item_id, status="dropped")
            self._drop_node(item_id)
        else:
            self._set(item_id, status="pending")

    def defer(self, item_id: str) -> None:
        """Not researched for a reason that is not the question's fault (collection budget
        used up): back to pending, and the attempt does not count."""
        it = self.get(item_id)
        if not it:
            return
        self._set(item_id, status="pending", attempts=max(0, int(it["attempts"]) - 1))

    def topic_for(self, item: Dict[str, Any]) -> str:
        """Research topic for an item: the stored topic term, its node label, else the main entity."""
        if item.get("topic"):
            return item["topic"]
        if item.get("topic_node_id"):
            node = self.graph.get_node(item["topic_node_id"])
            if node:
                return node["label"]
        from memory.entity_resolver import heuristic_entities
        ents = heuristic_entities(item["question"], 1)
        return ents[0] if ents else item["question"]
