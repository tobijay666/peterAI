"""3-D force-directed layout of the stored vector graph (mind view layout "graph").

The picture is the graph itself, not a tree drawn over it:
  * a spring on every stored edge between drawn nodes (Peter's has_subject edges too);
    its rest length shrinks with the edge's vector weight (cosine similarity), so closely
    related topics sit close together and weak links are long;
  * weak global repulsion so nodes do not overlap (all pairs for small graphs, a fixed,
    seeded sample of partners per node for big ones);
  * a weak pull towards the origin, where Peter's node is fixed.

Stability (the owner asked for an accurate *and* stable picture):
  * a FRESH layout (no stored positions) runs enough iterations to converge, seeded
    deterministically, so it is already near-equilibrium;
  * when the graph has not changed since the stored layout the caller returns the stored
    positions unchanged (see mind_view.graph_snapshot) — no iteration, no rewrite;
  * when a few nodes/edges changed, only the new nodes and their neighbours are relaxed
    (``pinned`` holds the rest fixed) at a low temperature, so the rest of the picture does
    not slide around.

Deterministic: initial positions come from the previous layout when there is one
(stored on the nodes), else from the semantic projection (memory/layout.py), else a
hash of the id; the repulsion sample uses a fixed seed.
"""
from __future__ import annotations

import hashlib
import math
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

Vec3 = Tuple[float, float, float]

ORIGIN_ID = "self_peter"
REST_MIN = 3.0          # rest length of an edge with weight 1
REST_MAX = 13.0         # ...and with weight 0
SPRING_K = 0.06
REPULSION = 6.0         # weak: c / d^2
GRAVITY = 0.004
MAX_STEP = 1.5          # cap on a node's move per iteration in a fresh layout
SETTLE_STEP = 0.35      # ...and in an incremental settle (small, so nothing jumps)
FRESH_ITERS = 160       # a fresh layout iterates to near-equilibrium
SETTLE_ITERS = 12       # an incremental relax of a few nodes
SAMPLE_ABOVE = 200      # nodes: switch from all pairs to a sampled repulsion
SAMPLE_PARTNERS = 48
DEFAULT_WEIGHT = 0.4    # edges without a vector weight (curious_about, cites, ...)


def _hash_point(key: str, radius: float = 10.0) -> Vec3:
    h = hashlib.md5(key.encode()).digest()
    u = int.from_bytes(h[:4], "little") / 0xFFFFFFFF
    v = int.from_bytes(h[4:8], "little") / 0xFFFFFFFF
    theta, phi = 2 * math.pi * u, math.acos(2 * v - 1)
    return (radius * math.sin(phi) * math.cos(theta), radius * math.sin(phi) * math.sin(theta), radius * math.cos(phi))


def _weight(edge: Dict) -> float:
    d = edge.get("data") or {}
    w = d.get("weight") if isinstance(d, dict) else None
    try:
        w = float(w) if w is not None else DEFAULT_WEIGHT
    except (TypeError, ValueError):
        w = DEFAULT_WEIGHT
    return min(1.0, max(0.0, w))


def rest_length(weight: float) -> float:
    return REST_MIN + (REST_MAX - REST_MIN) * (1.0 - min(1.0, max(0.0, weight)))


def graph_positions(node_ids: List[str], edges: List[Dict], initial: Dict[str, Vec3],
                    previous: Optional[Dict[str, Vec3]] = None, seed: int = 7,
                    iterations: Optional[int] = None, pinned: Optional[Set[str]] = None,
                    fresh: Optional[bool] = None) -> Dict[str, Vec3]:
    """node id -> (x, y, z). Peter (``self_peter``) is fixed at the origin and takes part
    through his edges; he is not in the result.

    ``previous`` seeds positions; ``pinned`` holds those nodes fixed (their neighbours relax
    around them). ``fresh`` (default: true when nothing is pinned and few nodes have a
    previous position) picks the annealed, convergent schedule; otherwise a short low-step
    settle that barely moves anything."""
    ids = sorted(set(node_ids) - {ORIGIN_ID})
    if not ids:
        return {}
    previous = previous or {}
    pinned = set(pinned or ())
    index = {nid: i + 1 for i, nid in enumerate(ids)}          # 0 = Peter
    index[ORIGIN_ID] = 0
    n = len(ids) + 1
    P = np.zeros((n, 3), dtype=np.float64)
    placed = 0
    free_mask = np.ones(n, dtype=bool)
    free_mask[0] = False                                       # Peter is always fixed at the origin
    for nid, i in index.items():
        if nid == ORIGIN_ID:
            continue
        p = previous.get(nid)
        if p is None:
            p = initial.get(nid) or _hash_point(nid)
        else:
            placed += 1
        if nid in pinned:
            free_mask[i] = False
    # second pass so free_mask is complete before we jitter only the free nodes
    for nid, i in index.items():
        if nid == ORIGIN_ID:
            continue
        p = previous.get(nid) or initial.get(nid) or _hash_point(nid)
        P[i] = p
    # coincident starting points (e.g. many hubs projected to one spot) get a deterministic nudge,
    # but only for nodes that are free to move (pinned nodes keep their exact stored position)
    jitter = np.array([_hash_point(nid, 0.05) for nid in ids])
    for k, nid in enumerate(ids):
        if free_mask[index[nid]]:
            P[index[nid]] += jitter[k]

    if fresh is None:
        fresh = not pinned and placed <= max(2, 0.5 * len(ids))
    if iterations is None:
        iterations = FRESH_ITERS if fresh else SETTLE_ITERS
    max_step = MAX_STEP if fresh else SETTLE_STEP

    src, dst, rest, k = [], [], [], []
    seen = set()
    for e in edges:
        a, b = index.get(e["src"]), index.get(e["dst"])
        if a is None or b is None or a == b:
            continue
        key = (min(a, b), max(a, b))
        if key in seen:
            continue
        seen.add(key)
        w = _weight(e)
        src.append(a); dst.append(b); rest.append(rest_length(w)); k.append(SPRING_K * (0.5 + w))
    src_a, dst_a = np.asarray(src, dtype=np.int64), np.asarray(dst, dtype=np.int64)
    rest_a, k_a = np.asarray(rest), np.asarray(k)

    rng = np.random.default_rng(seed)
    partners = rng.integers(0, n, size=(n, SAMPLE_PARTNERS)) if n > SAMPLE_ABOVE else None
    free_col = free_mask[:, None]

    for it in range(iterations):
        F = np.zeros_like(P)
        if len(src_a):
            D = P[dst_a] - P[src_a]
            L = np.linalg.norm(D, axis=1) + 1e-9
            f = (k_a * (L - rest_a) / L)[:, None] * D
            np.add.at(F, src_a, f)
            np.add.at(F, dst_a, -f)
        if partners is None:
            D = P[:, None, :] - P[None, :, :]
            d2 = (D * D).sum(-1) + 0.01
            np.fill_diagonal(d2, np.inf)
            F += (REPULSION * D / (d2 ** 1.5)[:, :, None]).sum(axis=1)
        else:
            D = P[:, None, :] - P[partners]                      # (n, m, 3)
            d2 = (D * D).sum(-1) + 0.01
            d2[partners == np.arange(n)[:, None]] = np.inf
            F += (REPULSION * (n / SAMPLE_PARTNERS) * D / (d2 ** 1.5)[:, :, None]).sum(axis=1)
        F -= GRAVITY * P
        # fresh: anneal from max_step down to ~0 so the layout settles; settle: constant small cap
        temp = (max_step * (1.0 - it / max(1, iterations)) + 0.03) if fresh else max_step
        norm = np.linalg.norm(F, axis=1, keepdims=True) + 1e-12
        step = F * np.minimum(1.0, temp / norm)
        P += step * free_col                                     # pinned / Peter never move
    return {nid: (round(float(P[i, 0]), 3), round(float(P[i, 1]), 3), round(float(P[i, 2]), 3))
            for nid, i in index.items() if nid != ORIGIN_ID}
