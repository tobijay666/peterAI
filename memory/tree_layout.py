"""Tree layout for the mind view: connected topics drawn as top-down tree diagrams.

  Peter (core)
    └── Writing                          <- main subjects, linked straight to Peter
          └── Victorian literature       <- field hubs (abstractions)
      ├── The Picture of Dorian Gray     <- topics under the field that abstracts them
      │     ├── Oscar Wilde              <- related topics under the better-known one
      │     └── "Who is Basil…?"         <- curiosity questions as leaves
      └── …

Each node gets at most one parent, chosen by relation priority, so the result is a
forest of real trees. Links that don't fit the tree (a second parent, a cross link)
are still drawn as edges; they just don't decide positions. Trees are laid out
left to right and wrap into rows; the whole picture lies in one vertical plane
(z = 0) so it reads like a diagram. Deterministic: same graph, same picture.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

Vec3 = Tuple[float, float, float]

LEVEL_GAP = 5.0       # vertical distance between tree levels
LEAF_GAP = 3.2        # horizontal space per topic leaf
SMALL_LEAF_GAP = 1.8  # curiosity questions / skills / agents take less room
TREE_GAP = 4.0        # space between neighbouring trees
ROW_GAP = 8.0         # space between rows of trees
MAX_ROW_WIDTH = 70.0  # wrap trees onto a new row past this width

def _is_subject(n: Dict) -> bool:
    d = n.get("data")
    if isinstance(d, str):
        import json
        try:
            d = json.loads(d or "{}")
        except Exception:
            d = {}
    return isinstance(d, dict) and d.get("role") == "subject"


_SMALL = {"curiosity", "skill", "agent", "source"}
_KIND_ORDER = {"abstraction": 0, "entity": 1, "skill": 2, "agent": 3, "curiosity": 4, "source": 5}


def _parent_candidates(edges: Iterable[Dict], ids: set, kind: Dict[str, str],
                       usage: Dict[str, float]) -> Dict[str, List[Tuple[int, str]]]:
    """child -> [(priority, parent)], lower priority wins."""
    cand: Dict[str, List[Tuple[int, str]]] = {}

    def add(child: str, parent: str, prio: int) -> None:
        if child in ids and parent in ids and child != parent:
            cand.setdefault(child, []).append((prio, parent))

    for e in edges:
        s, d, rel = e["src"], e["dst"], e["relation"]
        if rel == "abstracts":                       # hub above the topic it abstracts
            data = e.get("data") or {}
            add(d, s, -1 if isinstance(data, dict) and data.get("primary") else 0)   # primary hub first
        elif rel == "curious_about":                 # the question hangs under its topic
            add(s, d, 1)
        elif rel in ("similar_to", "has_subject", "helper_of", "spawned_by"):
            continue                                 # cross links / plumbing: drawn, never a parent
        elif rel == "cites":
            add(d, s, 3)
        elif rel in ("related_to", "bridges"):       # the less-used topic hangs under the more-used one
            if s in ids and d in ids:
                us, ud = usage.get(s, 0.0), usage.get(d, 0.0)
                if (us, s) >= (ud, d):
                    add(d, s, 2)
                else:
                    add(s, d, 2)
        else:
            add(d, s, 4)
    return cand


def build_forest(nodes: List[Dict], edges: List[Dict]) -> Tuple[List[str], Dict[str, List[str]]]:
    """Roots (in display order) and children lists. Cycles are broken; every node appears once."""
    ids = {n["id"] for n in nodes}
    kind = {n["id"]: n.get("kind", "entity") for n in nodes}
    usage = {n["id"]: float(n.get("usage_score") or 0.0) for n in nodes}
    label = {n["id"]: (n.get("label") or "").lower() for n in nodes}
    cand = _parent_candidates(edges, ids, kind, usage)

    subjects = {n["id"] for n in nodes if _is_subject(n)}
    parent: Dict[str, Optional[str]] = {}
    for child in sorted(ids):
        if child in subjects:            # main subjects always sit at the top, under Peter
            parent[child] = None
            continue
        opts = sorted(cand.get(child, []), key=lambda t: (t[0], -usage.get(t[1], 0.0), t[1]))
        parent[child] = opts[0][1] if opts else None

    # Break cycles: walk up from each node; if we come back to it, cut the weakest link (its own)
    for start in sorted(ids):
        seen, cur = set(), start
        while cur is not None and cur not in seen:
            seen.add(cur)
            cur = parent.get(cur)
        if cur is not None:              # cur is on a cycle
            loop, x = [cur], parent[cur]
            while x != cur:
                loop.append(x)
                x = parent[x]
            # the node with the highest usage (then hub kinds) becomes a root
            top = sorted(loop, key=lambda i: (_KIND_ORDER.get(kind[i], 9), -usage[i], i))[0]
            parent[top] = None

    children: Dict[str, List[str]] = {i: [] for i in ids}
    for c, p in parent.items():
        if p is not None:
            children[p].append(c)
    order = lambda i: (_KIND_ORDER.get(kind[i], 9), -usage[i], label[i], i)
    for p in children:
        children[p].sort(key=order)
    subj = {n["id"] for n in nodes if _is_subject(n)}
    roots = sorted([i for i in ids if parent[i] is None],
                   key=lambda i: (i not in subj, -_subtree_size(i, children), order(i)))
    return roots, children


def _subtree_size(root: str, children: Dict[str, List[str]]) -> int:
    n, stack = 0, [root]
    while stack:
        x = stack.pop()
        n += 1
        stack.extend(children.get(x, []))
    return n


def tree_positions(nodes: List[Dict], edges: List[Dict]) -> Dict[str, Vec3]:
    """node id -> (x, y, z) for a forest of top-down trees, centred on the origin."""
    if not nodes:
        return {}
    kind = {n["id"]: n.get("kind", "entity") for n in nodes}
    roots, children = build_forest(nodes, edges)

    width: Dict[str, float] = {}

    def measure(root: str) -> float:
        """Subtree widths, iteratively (long chains must not hit the recursion limit)."""
        order, stack = [], [root]
        while stack:
            x = stack.pop()
            order.append(x)
            stack.extend(children.get(x, []))
        for i in reversed(order):
            kids = children.get(i, [])
            own = SMALL_LEAF_GAP if kind[i] in _SMALL else LEAF_GAP
            width[i] = max(own, sum(width[c] for c in kids)) if kids else own
        return width[root]

    local: Dict[str, Tuple[float, float]] = {}

    def place(root: str, left0: float, depth0: int) -> int:
        """Position subtree root inside [left0, left0 + width]; returns its deepest level."""
        deepest, order, stack = depth0, [], [(root, left0, depth0)]
        while stack:
            i, left, depth = stack.pop()
            order.append(i)
            deepest = max(deepest, depth)
            local[i] = (left + width[i] / 2.0, -depth * LEVEL_GAP)
            x = left
            for c in children.get(i, []):
                stack.append((c, x, depth + 1))
                x += width[c]
        for i in reversed(order):     # centre each parent over its first and last child
            kids = children.get(i, [])
            if kids:
                local[i] = ((local[kids[0]][0] + local[kids[-1]][0]) / 2.0, local[i][1])
        return deepest

    # Lay trees out left to right, wrapping into rows
    pos: Dict[str, Vec3] = {}
    row_x, row_top, row_depth = 0.0, 0.0, 0
    rows: List[List[str]] = [[]]
    for r in roots:
        measure(r)
    for r in roots:
        if row_x > 0 and row_x + width[r] > MAX_ROW_WIDTH:
            row_top -= (row_depth + 1) * LEVEL_GAP + ROW_GAP
            row_x, row_depth = 0.0, 0
            rows.append([])
        local.clear()
        depth = place(r, row_x, 0)
        row_depth = max(row_depth, depth)
        for i, (x, y) in local.items():
            pos[i] = (x, y + row_top, 0.0)
        rows[-1].append(r)
        row_x += width[r] + TREE_GAP

    # Centre horizontally and hang the forest just below Peter's core at the origin,
    # so the core reads as the root every recall pulse starts from
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    cx, top = (min(xs) + max(xs)) / 2.0, max(ys)
    return {i: (round(x - cx, 3), round(y - top - LEVEL_GAP, 3), 0.0) for i, (x, y, _) in pos.items()}


def core_subjects(nodes: List[Dict], edges: List[Dict]) -> List[str]:
    """Main subjects (role=subject), in display order: the nodes linked straight to Peter."""
    subj = {n["id"] for n in nodes if _is_subject(n)}
    roots, _ = build_forest(nodes, edges)
    return [r for r in roots if r in subj]


# --------------------------------------------------------------------------- 3-D "cluster" layout
#
# Same hierarchy as the tree (Peter -> main subjects -> topics -> details), but in 3-D
# and shaped by meaning: each node's semantic position (embedding projected to 3-D)
# decides which way it grows from its parent, so similar subjects sit next to each
# other around Peter and similar topics bunch together under their subject.

SUBJECT_RADIUS = 14.0   # distance of the main subjects from Peter's core
CHILD_LEN = 8.0         # parent -> child distance at the first level (shrinks per level)
CHILD_DECAY = 0.72
MIN_GAP = 2.6           # minimum distance between sibling topics
SMALL_MIN_GAP = 1.5     # ...between small nodes (questions, skills)
OUTWARD = 0.55          # how much branches lean away from Peter (vs. following meaning)


def _unit(v):
    import math
    n = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    return (v[0] / n, v[1] / n, v[2] / n) if n > 1e-9 else None


def _fib(i: int, n: int):
    """Point i of n spread evenly on the unit sphere (fallback direction)."""
    import math
    y = 1 - 2 * (i + 0.5) / max(n, 1)
    r = math.sqrt(max(0.0, 1 - y * y))
    t = math.pi * (3 - math.sqrt(5)) * i
    return (math.cos(t) * r, y, math.sin(t) * r)


def _spread(points: Dict[str, List[float]], centre, min_gap: Dict[str, float], iters: int = 30,
            radius: Optional[float] = None) -> None:
    """Push points apart until no two are closer than their gaps (in place).
    With radius, points stay on the sphere of that radius around centre."""
    import math
    ids = sorted(points)
    if len(ids) < 2:
        return
    if len(ids) > 40:
        _spread_np(points, ids, centre, min_gap, iters, radius)
        return
    for _ in range(iters):
        moved = False
        for a_i in range(len(ids)):
            a = points[ids[a_i]]
            for b_i in range(a_i + 1, len(ids)):
                b = points[ids[b_i]]
                need = (min_gap[ids[a_i]] + min_gap[ids[b_i]]) / 2.0
                dx, dy, dz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
                d = math.sqrt(dx * dx + dy * dy + dz * dz)
                if d >= need:
                    continue
                if d < 1e-6:
                    dx, dy, dz = _fib(b_i, len(ids)); d = 1.0
                push = (need - d) / 2.0 / d
                for k, v in enumerate((dx, dy, dz)):
                    a[k] -= v * push
                    b[k] += v * push
                moved = True
        if radius is not None:
            for i in ids:
                p = points[i]
                u = _unit((p[0] - centre[0], p[1] - centre[1], p[2] - centre[2])) or (0.0, 1.0, 0.0)
                points[i] = [centre[0] + u[0] * radius, centre[1] + u[1] * radius, centre[2] + u[2] * radius]
        if not moved:
            break


def _spread_np(points, ids, centre, min_gap, iters, radius) -> None:
    """Vectorised _spread for large groups (hundreds of loose topics or siblings)."""
    import numpy as np
    P = np.array([points[i] for i in ids], dtype=np.float32)
    g = np.array([min_gap[i] for i in ids], dtype=np.float32)
    need = (g[:, None] + g[None, :]) / 2.0
    np.fill_diagonal(need, 0.0)
    c = np.array(centre, dtype=np.float32)
    jitter = np.array([_fib(k, len(ids)) for k in range(len(ids))], dtype=np.float32) * 1e-3
    P += jitter                                   # coincident points get a deterministic nudge
    n_ = len(ids)
    for _ in range(min(iters, 15 if n_ <= 300 else 6 if n_ <= 800 else 2)):
        D = P[None, :, :] - P[:, None, :]          # D[a, b] = P[b] - P[a]
        d = np.sqrt((D * D).sum(-1)) + 1e-6
        over = np.clip(need - d, 0.0, None)
        if not over.any():
            break
        push = (over / d / 2.0)[:, :, None] * D
        P -= push.sum(axis=1)
        if radius is not None:
            v = P - c
            n = np.linalg.norm(v, axis=1, keepdims=True)
            n[n < 1e-6] = 1.0
            P = c + v / n * radius
    for k, i in enumerate(ids):
        points[i] = [float(x) for x in P[k]]


def cluster_positions(nodes: List[Dict], edges: List[Dict], semantic: Dict[str, Vec3]) -> Dict[str, Vec3]:
    """node id -> (x, y, z): the tree hierarchy in 3-D, arranged by similarity.

    semantic: each node's meaning-based position (e.g. LayoutEngine.place); nodes
    missing from it get an even spread. Deterministic."""
    if not nodes:
        return {}
    kind = {n["id"]: n.get("kind", "entity") for n in nodes}
    roots, children = build_forest(nodes, edges)
    sem = {i: semantic.get(i) for i in kind}

    # Main subjects (and any loose roots) around Peter, in the direction of their meaning
    centre_of_meaning = [0.0, 0.0, 0.0]
    known = [v for v in sem.values() if v is not None]
    if known:
        centre_of_meaning = [sum(v[k] for v in known) / len(known) for k in range(3)]
    pos: Dict[str, List[float]] = {}
    for idx, r in enumerate(roots):
        s = sem.get(r)
        u = _unit((s[0] - centre_of_meaning[0], s[1] - centre_of_meaning[1], s[2] - centre_of_meaning[2])) \
            if s is not None else None
        u = u or _fib(idx, len(roots))
        pos[r] = [u[0] * SUBJECT_RADIUS, u[1] * SUBJECT_RADIUS, u[2] * SUBJECT_RADIUS]
    # bigger trees need more room on the sphere
    gaps = {r: min(SUBJECT_RADIUS, 4.0 + 1.2 * _subtree_size(r, children) ** 0.5) for r in roots}
    _spread(pos, (0.0, 0.0, 0.0), gaps, radius=SUBJECT_RADIUS)

    # Grow each tree outward; a child's direction = its meaning relative to its parent,
    # leaning away from Peter so branches don't fold back into the centre
    level = [(r, 1) for r in roots]
    while level:
        nxt = []
        for p, depth in level:
            kids = children.get(p, [])
            if not kids:
                continue
            pp = pos[p]
            out = _unit(tuple(pp)) or (0.0, 1.0, 0.0)
            length = max(2.4, CHILD_LEN * (CHILD_DECAY ** (depth - 1)))
            sib: Dict[str, List[float]] = {}
            for j, c in enumerate(kids):
                sc, sp = sem.get(c), sem.get(p)
                d = _unit((sc[0] - sp[0], sc[1] - sp[1], sc[2] - sp[2])) if sc is not None and sp is not None else None
                d = d or _fib(j, len(kids))
                v = _unit(tuple(OUTWARD * out[k] + (1 - OUTWARD) * d[k] for k in range(3))) or out
                ln = length * (0.6 if kind[c] in _SMALL else 1.0)
                sib[c] = [pp[0] + v[0] * ln, pp[1] + v[1] * ln, pp[2] + v[2] * ln]
            gaps = {c: SMALL_MIN_GAP if kind[c] in _SMALL else MIN_GAP for c in kids}
            _spread(sib, tuple(pp), gaps, iters=20)
            pos.update(sib)
            nxt.extend((c, depth + 1) for c in kids)
        level = nxt
    return {i: (round(p[0], 3), round(p[1], 3), round(p[2], 3)) for i, p in pos.items()}
