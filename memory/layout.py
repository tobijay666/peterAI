"""Stable 3-D layout for the mind view (Stage 6).

A node's position is its collection centroid projected to 3-D. The projection
(PCA basis: mean + 3 components) is fitted once, saved next to the user's graph
DB (layout.json), and reused, so new nodes land in place without reshuffling
the others. The basis is refitted only during sleep (``refit()``), and the
moves are emitted so the browser can animate them.

Nodes without a collection are placed relative to what they belong to:
curiosity items orbit their topic, sources orbit the node citing them, agents
and skills sit on fixed rings. Everything is deterministic (no randomness).
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.logger import logger

RADIUS = 12.0
Vec3 = Tuple[float, float, float]


def _hash_unit(key: str) -> Vec3:
    """Deterministic point on the unit sphere for a string."""
    h = hashlib.md5(key.encode()).digest()
    u = int.from_bytes(h[:4], "little") / 0xFFFFFFFF
    v = int.from_bytes(h[4:8], "little") / 0xFFFFFFFF
    theta = 2 * math.pi * u
    phi = math.acos(2 * v - 1)
    return (math.sin(phi) * math.cos(theta), math.sin(phi) * math.sin(theta), math.cos(phi))


class LayoutEngine:
    def __init__(self, vector: Any, graph: Any, path: Optional[Path] = None) -> None:
        self.vector = vector
        self.graph = graph
        self.path = Path(path) if path else Path(graph.db_path).parent / "layout.json"
        self._basis: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------ basis

    def _load(self) -> Optional[Dict[str, Any]]:
        if self._basis is None and self.path.exists():
            try:
                self._basis = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                self._basis = None
        return self._basis

    def _save(self, basis: Dict[str, Any]) -> None:
        self._basis = basis
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(basis), encoding="utf-8")
        except Exception as e:
            logger.warning(f"LayoutEngine: could not save basis: {e}")

    def _centroids(self) -> Dict[str, List[float]]:
        out = {}
        for n in self.graph.all_nodes(["entity", "abstraction"]):
            if n.get("status") == "pruned" or not n.get("collection"):
                continue
            c = self.vector.centroid(n["collection"])
            if c:
                out[n["id"]] = c
        return out

    def fit(self, centroids: Dict[str, List[float]]) -> Optional[Dict[str, Any]]:
        if len(centroids) < 2:
            return None
        ids = sorted(centroids)
        X = np.asarray([centroids[i] for i in ids], dtype=float)
        mean = X.mean(axis=0)
        Xc = X - mean
        _, _, vt = np.linalg.svd(Xc, full_matrices=False)
        comps = np.zeros((3, X.shape[1]))
        k = min(3, vt.shape[0])
        comps[:k] = vt[:k]
        # deterministic sign: the largest-magnitude loading of each component is positive
        for i in range(k):
            j = int(np.argmax(np.abs(comps[i])))
            if comps[i, j] < 0:
                comps[i] *= -1
        proj = Xc @ comps.T
        scale = float(np.max(np.linalg.norm(proj, axis=1))) or 1.0
        basis = {"mean": mean.tolist(), "components": comps.tolist(), "scale": scale, "n": len(ids)}
        self._save(basis)
        return basis

    def project(self, centroid: List[float]) -> Optional[Vec3]:
        basis = self._load()
        if not basis:
            return None
        v = np.asarray(centroid, dtype=float) - np.asarray(basis["mean"], dtype=float)
        p = (np.asarray(basis["components"], dtype=float) @ v) / float(basis["scale"]) * RADIUS
        return (float(p[0]), float(p[1]), float(p[2]))

    # ------------------------------------------------------------------ placement

    def _store(self, node_id: str, pos: Vec3) -> Vec3:
        self.graph.set_node_fields(node_id, pos_x=pos[0], pos_y=pos[1], pos_z=pos[2])
        return pos

    def place(self, node: Dict[str, Any]) -> Vec3:
        """Position for one node; computed once and stored, then reused."""
        if node.get("pos_x") is not None:
            return (float(node["pos_x"]), float(node["pos_y"]), float(node["pos_z"]))
        kind = node.get("kind")
        pos: Optional[Vec3] = None
        if kind in ("entity", "abstraction") and node.get("collection"):
            c = self.vector.centroid(node["collection"])
            if c is not None:
                if self._load() is None:
                    self.fit(self._centroids())
                pos = self.project(c)
            if pos is not None and kind == "abstraction":
                pos = (pos[0] * 0.8, pos[1] * 0.8 + 2.0, pos[2] * 0.8)  # hover above members
        if pos is None:
            pos = self._relative(node)
            if kind in ("entity", "abstraction"):
                return pos  # provisional (empty collection): not stored, placed properly once it has records
        return self._store(node["id"], pos)

    def _anchor(self, node_id: str, relations: Tuple[str, ...]) -> Optional[Dict[str, Any]]:
        for e in self.graph.edges_of(node_id):
            if e["relation"] in relations:
                other = e["dst"] if e["src"] == node_id else e["src"]
                n = self.graph.get_node(other)
                if n:
                    return n
        return None

    def _relative(self, node: Dict[str, Any]) -> Vec3:
        kind = node.get("kind")
        u = _hash_unit(node["id"])
        if kind in ("curiosity", "source"):
            anchor = self._anchor(node["id"], ("curious_about", "cites"))
            if anchor is not None:
                ax, ay, az = self.place(anchor)
                r = 2.2 if kind == "curiosity" else 1.4
                return (ax + u[0] * r, ay + u[1] * r, az + u[2] * r)
            return (u[0] * RADIUS * 1.1, u[1] * RADIUS * 1.1, u[2] * RADIUS * 1.1)
        if kind == "agent":
            return (u[0] * 3.0, 0.0, u[2] * 3.0)
        if kind == "skill":
            return (u[0] * RADIUS * 0.5, -RADIUS * 0.6, u[2] * RADIUS * 0.5)
        return (u[0] * RADIUS, u[1] * RADIUS, u[2] * RADIUS)

    def refit(self) -> List[Dict[str, Any]]:
        """Sleep-time refit: new basis, every entity/abstraction moved. Returns the moves."""
        cents = self._centroids()
        if self.fit(cents) is None:
            return []
        moves = []
        for nid, c in cents.items():
            node = self.graph.get_node(nid)
            p = self.project(c)
            if node is None or p is None:
                continue
            if node["kind"] == "abstraction":
                p = (p[0] * 0.8, p[1] * 0.8 + 2.0, p[2] * 0.8)
            self._store(nid, p)
            moves.append({"node_id": nid, "position": [round(x, 3) for x in p]})
        # dependants follow their anchors on next placement
        for n in self.graph.all_nodes(["curiosity", "source"]):
            self.graph.set_node_fields(n["id"], pos_x=None, pos_y=None, pos_z=None)
        return moves
