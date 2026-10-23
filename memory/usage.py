"""Usage tracking for cognitive-memory nodes (spec Phase 1.3 activation counter).

    u = ln(1 + n) * exp(-dt / h) * (0.5 + importance)

n = access count, dt = days since last access (or creation), h = half-life in
days (config ``usage.half_life_days``). The raw score is stored on the node;
the UI tiers use the node's *percentile* among the user's entity and
abstraction nodes, so tiers stay meaningful as memory grows.
"""
from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from typing import Any, Dict, Iterable, List, Optional

from core.clock import Clock, SystemClock
from core.cognition_config import cognition_config
from core.events import emit
from memory.graph_store import GraphMemory
from memory.visual_style import visual_style

COGNITIVE_KINDS = ["entity", "abstraction"]
_DAY = 86400.0


def usage_score(access_count: int, last_accessed: Optional[float], importance: float,
                now: float, half_life_days: float) -> float:
    n = max(0, int(access_count or 0))
    if n == 0:
        return 0.0
    last = last_accessed if last_accessed is not None else now
    dt_days = max(0.0, (now - float(last)) / _DAY)
    h = max(1e-6, float(half_life_days))
    imp = float(importance if importance is not None else 0.5)
    return math.log1p(n) * math.exp(-dt_days / h) * (0.5 + imp)


def percentile_ranks(scores: Dict[str, float]) -> Dict[str, float]:
    """Mid-rank percentile in [0, 1]. A single node sits at 0.5 (moderate)."""
    n = len(scores)
    if n == 0:
        return {}
    if n == 1:
        return {k: 0.5 for k in scores}
    ordered = sorted(scores.values())
    out = {}
    for k, s in scores.items():
        below = bisect_left(ordered, s)
        equal = bisect_right(ordered, s) - below
        out[k] = (below + 0.5 * (equal - 1)) / (n - 1)
    return out


class UsageTracker:
    """Activation + scoring over the graph's cognitive nodes."""

    def __init__(self, graph: GraphMemory, clock: Optional[Clock] = None,
                 cfg: Optional[Dict[str, Any]] = None) -> None:
        self.graph = graph
        self._clock = clock or getattr(graph, "_clock", None) or SystemClock()
        self._cfg = cfg

    @property
    def cfg(self) -> Dict[str, Any]:
        return self._cfg or cognition_config()

    @property
    def half_life(self) -> float:
        return float(self.cfg["usage"]["half_life_days"])

    def score_node(self, node: Dict[str, Any], now: Optional[float] = None) -> float:
        now = self._clock.now() if now is None else now
        last = node.get("last_accessed")
        if last is None:
            last = node.get("created_at")
        return usage_score(node.get("access_count", 0), last, node.get("importance", 0.5), now, self.half_life)

    def recompute_all(self, kinds: Optional[List[str]] = None, persist: bool = True) -> Dict[str, Dict[str, float]]:
        """Score every cognitive node. Returns {node_id: {"score", "percentile"}}."""
        now = self._clock.now()
        nodes = [n for n in self.graph.all_nodes(kinds or COGNITIVE_KINDS) if n.get("status") != "pruned"]
        scores = {n["id"]: self.score_node(n, now) for n in nodes}
        pct = percentile_ranks(scores)
        if persist:
            for nid, sc in scores.items():
                self.graph.set_node_fields(nid, usage_score=sc)
        return {nid: {"score": scores[nid], "percentile": pct[nid]} for nid in scores}

    def percentile_of(self, node_id: str) -> float:
        return self.recompute_all(persist=False).get(node_id, {}).get("percentile", 0.5)

    def style_of(self, node: Dict[str, Any], percentile: Optional[float] = None) -> Dict[str, Any]:
        p = self.percentile_of(node["id"]) if percentile is None else percentile
        return visual_style(p, node.get("status", "active"), self.cfg)

    def touch(self, node_id: str, turn_id: str = "", emit_event: bool = True) -> Optional[Dict[str, Any]]:
        """Activation counter increment (+ last-accessed timestamp), then rescore."""
        node = self.graph.touch(node_id)
        if node is None:
            return None
        score = self.score_node(node)
        self.graph.set_node_fields(node_id, usage_score=score)
        node["usage_score"] = score
        if emit_event:
            pct = self.percentile_of(node_id)
            style = visual_style(pct, node.get("status", "active"), self.cfg)
            emit("node_touched", user_id=self.graph.user_id, turn_id=turn_id, node_id=node_id,
                 usage_score=round(score, 4), percentile=round(pct, 4), tier=style["tier"],
                 access_count=node.get("access_count", 0), style=style)
        return node

    def touch_many(self, node_ids: Iterable[str], turn_id: str = "") -> List[Dict[str, Any]]:
        out = []
        for nid in dict.fromkeys(node_ids):
            n = self.touch(nid, turn_id=turn_id)
            if n:
                out.append(n)
        return out
