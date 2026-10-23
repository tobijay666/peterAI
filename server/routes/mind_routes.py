"""
Mind-view endpoints.

GET /api/mind/snapshot
    Returns the user's current memory as 3-D projected points (for the
    memory cloud) and graph nodes/edges (for the knowledge graph).

GET /api/mind/replay
    Returns the user's event log (JSON-lines) as a list, for the timeline
    and replay feature in the frontend.

Cognitive memory (Stage 6):
GET  /api/mind/snapshot?view=graph     knowledge web: entity / abstraction / curiosity /
                                       skill / agent nodes with server-decided style and
                                       stable positions, plus edges   (default view)
GET  /api/mind/snapshot?view=records   the original per-record point cloud
GET  /api/mind/node/{node_id}          inspection panel data
POST /api/mind/sleep                   start a sleep cycle now (body: {"mode": "active"|"shadow"})

3-D projection uses PCA here (Phase 2).  Phase 6 adds UMAP with persisted
coordinates so new points land without reshuffling.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional

from core.logger import logger
from core.user_context import UserContext
from server.auth import get_current_user
from server.peter_pool import pool

router = APIRouter(prefix="/api/mind", tags=["mind"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _demo_seed_points() -> List[Dict[str, Any]]:
    """Returns synthetic seed points shown when the DB is empty.

    These let you see the 3-D scene immediately.  Each point carries
    demo=True so the frontend can render them at reduced opacity.
    When real memories exist they replace these entirely.
    """
    import math
    seeds = [
        # (label, agent, kind, importance, theta_deg, phi_deg, r)
        ("Peter is ready to learn",       "peter",      "finding",  0.9,   0,   60, 10),
        ("Researcher awaiting queries",   "researcher",  "finding",  0.6,  45,   80, 12),
        ("Librarian indexing knowledge",  "librarian",   "finding",  0.6,  90,   70, 11),
        ("QA agent standing by",          "qa",          "finding",  0.5, 135,   90, 13),
        ("Voice agent ready",             "voice",       "finding",  0.5, 180,   50, 10),
        ("Filesystem tools mounted",      "filesystem",  "skill",    0.7, 225,  100, 12),
        ("DevOps pipeline connected",     "devops",      "skill",    0.6, 270,   65, 11),
        ("First skill will appear here",  "peter",       "skill",    0.4, 315,  120,  9),
        ("Memories grow with each chat",  "peter",       "episode",  0.5,  22,   40,  8),
        ("Graph edges form over time",    "peter",       "semantic", 0.4, 200,   30,  9),
        ("Trust score calibrated",        "peter",       "finding",  0.5, 110,  140, 11),
        ("Latent space initialised",      "peter",       "semantic", 0.8,  67,   55, 14),
    ]
    points = []
    for label, agent, kind, imp, theta_deg, phi_deg, r in seeds:
        theta = math.radians(theta_deg)
        phi   = math.radians(phi_deg)
        points.append({
            "id":         f"demo-{len(points)}",
            "document":   label,
            "agent":      agent,
            "importance": imp,
            "kind":       kind,
            "trust":      "demo",
            "pinned":     False,
            "embedding":  [],
            "demo":       True,
            "position":   [
                r * math.sin(phi) * math.cos(theta),
                r * math.sin(phi) * math.sin(theta),
                r * math.cos(phi),
            ],
        })
    return points


def _scatter_sphere(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Fallback: scatter records randomly on a sphere so they are always visible."""
    import math, random
    rng = random.Random(42)  # stable seed so layout does not shuffle on every refresh
    for r in records:
        theta = rng.uniform(0, 2 * math.pi)
        phi   = rng.uniform(0, math.pi)
        r_val = rng.uniform(6, 14)
        r["position"] = [
            r_val * math.sin(phi) * math.cos(theta),
            r_val * math.sin(phi) * math.sin(theta),
            r_val * math.cos(phi),
        ]
    return records


def _project_3d(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Projects embeddings to 3-D using PCA.

    Falls back to a spherical scatter (always visible) when scikit-learn is
    absent, fewer than 4 records exist, or embeddings are not present.
    Demo seed points already have positions set and skip projection entirely.
    """
    # Demo points arrive pre-positioned — pass them through unchanged
    if records and records[0].get("demo"):
        return records

    if len(records) < 4:
        return _scatter_sphere(records)

    try:
        import numpy as np
        from sklearn.decomposition import PCA

        vecs = [r.get("embedding", []) for r in records]
        # If embeddings are empty / short, fall through to scatter
        if not vecs[0] or len(vecs[0]) < 3:
            raise ValueError("embeddings missing or too short")

        arr = np.array(vecs, dtype=float)
        pca = PCA(n_components=3)
        proj = pca.fit_transform(arr) * 10   # scale to roughly ±10 units
        for i, r in enumerate(records):
            r["position"] = proj[i].tolist()
    except Exception as e:
        logger.debug(f"PCA projection skipped ({e}); using spherical scatter.")
        return _scatter_sphere(records)

    return records


def _read_event_log(log_path: Path, last_n: int = 500) -> List[Dict[str, Any]]:
    """Read the last N events from a JSON-lines log file."""
    if not log_path.exists():
        return []
    try:
        lines = log_path.read_text(encoding="utf-8").splitlines()
        recent = lines[-last_n:]
        events = []
        for line in recent:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return events
    except Exception as e:
        logger.warning(f"mind_routes: could not read event log at {log_path}: {e}")
        return []


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/snapshot")
async def snapshot(view: str = "graph", sources: bool = False, ns: Optional[str] = None,
                   user: UserContext = Depends(get_current_user)) -> Dict[str, Any]:
    """Returns the memory snapshot for the 3-D mind view (graph web or record cloud)."""
    import traceback
    try:
        if view == "graph":
            peter, _g, _r, _l = pool.get_or_create(user)
            cognition = getattr(peter, "cognition", None)
            if cognition is not None:
                from memory.mind_view import graph_snapshot
                return {"user_id": user.user_id, "view": "graph", **graph_snapshot(cognition, include_sources=sources)}
        return await _snapshot_impl(user, ns=ns)
    except Exception as exc:
        tb = traceback.format_exc()
        logger.error(f"[mind/snapshot] 500 error: {exc}\n{tb}")
        # Return error details so the browser can show them (dev mode)
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=500, content={"error": str(exc), "traceback": tb})


async def _snapshot_impl(user: UserContext, ns: Optional[str] = None) -> Dict[str, Any]:
    peter, _guardrails, _router, _learning = pool.get_or_create(user)
    vector_store = peter.vector
    graph_store  = peter.graph

    # Collect records from every namespace (embeddings included for PCA projection)
    raw_points: List[Dict[str, Any]] = []
    namespaces = [ns] if ns else vector_store.list_namespaces()
    for namespace in namespaces:
        if namespace == vector_store.INDEX_NAMESPACE or not vector_store.has_collection(namespace):
            continue
        for rec in vector_store.get_all_records(namespace):
            raw_points.append({
                "id":         rec["id"],
                "namespace":  namespace,
                "document":   rec["document"][:200],
                "agent":      rec["metadata"].get("agent", namespace),
                "importance": float(rec["metadata"].get("importance", 0.5)),
                "kind":       rec["metadata"].get("kind", "finding"),
                "trust":      rec["metadata"].get("trust", "untrusted_web"),
                "pinned":     bool(rec["metadata"].get("pinned", False)),
                "embedding":  rec.get("embedding", []),
                "demo":       False,
            })

    # If no real memories exist yet, inject demo seed points so the
    # visualization is never blank.  demo=True lets the frontend dim them.
    if not raw_points and not ns:
        raw_points = _demo_seed_points()

    points = _project_3d(raw_points)

    # Graph nodes and edges
    all_nodes: List[Dict[str, Any]] = []
    for kind in ["task", "agent", "skill", "finding", "episode", "semantic"]:
        for node in graph_store.get_nodes_by_kind(kind):
            all_nodes.append({
                "id":         node["id"],
                "kind":       node["kind"],
                "label":      node["label"][:80],
                "importance": float(node.get("importance", 0.5)),
            })

    all_edges: List[Dict[str, Any]] = []
    seen_edges: set = set()
    for node in all_nodes:
        for nbr in graph_store.neighbors(node["id"], max_depth=1):
            eid = nbr["edge_id"]
            if eid not in seen_edges:
                seen_edges.add(eid)
                all_edges.append({
                    "id":       eid,
                    "src":      nbr["src"],
                    "dst":      nbr["dst"],
                    "relation": nbr["relation"],
                })

    return {
        "user_id": user.user_id,
        "view":    "records",
        "points":  points,
        "graph":   {"nodes": all_nodes, "edges": all_edges},
    }


@router.get("/replay")
async def replay(
    user: UserContext = Depends(get_current_user),
    last_n: int = 500,
) -> Dict[str, Any]:
    """Returns the user's recent event log for timeline / replay in the frontend."""
    events = _read_event_log(user.events_log_path, last_n=last_n)
    return {"user_id": user.user_id, "events": events}


@router.get("/node/{node_id}")
async def node(node_id: str, user: UserContext = Depends(get_current_user)) -> Dict[str, Any]:
    """Inspection panel: usage, collection records with sources/confidence, edges, curiosity."""
    peter, _g, _r, _l = pool.get_or_create(user)
    cognition = getattr(peter, "cognition", None)
    if cognition is None:
        raise HTTPException(status_code=503, detail="Cognitive memory is not enabled.")
    from memory.mind_view import node_detail
    detail = node_detail(cognition, node_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="Node not found.")
    return detail


class PinNodeRequest(BaseModel):
    pinned: bool = True


@router.post("/node/{node_id}/pin")
async def pin_node(node_id: str, body: PinNodeRequest, user: UserContext = Depends(get_current_user)) -> Dict[str, Any]:
    """Pin a node so pruning never removes it (or unpin it)."""
    peter, _g, _r, _l = pool.get_or_create(user)
    if not peter.graph.set_node_fields(node_id, pinned=body.pinned):
        raise HTTPException(status_code=404, detail="Node not found.")
    return {"ok": True, "node_id": node_id, "pinned": body.pinned}


@router.delete("/node/{node_id}")
async def forget_node(node_id: str, user: UserContext = Depends(get_current_user)) -> Dict[str, Any]:
    """Forget a topic now: the node, its collection, aliases and index vector (pinned nodes are refused)."""
    peter, _g, _r, _l = pool.get_or_create(user)
    cognition = getattr(peter, "cognition", None)
    if cognition is None:
        raise HTTPException(status_code=503, detail="Cognitive memory is not enabled.")
    node = peter.graph.get_node(node_id)
    if node is None or node["kind"] not in ("entity", "abstraction"):
        raise HTTPException(status_code=404, detail="Topic node not found.")
    if node.get("pinned"):
        raise HTTPException(status_code=409, detail="Node is pinned; unpin it first.")
    cognition.knowledge.remove_node(node_id)
    return {"ok": True, "node_id": node_id}


class SleepRequest(BaseModel):
    mode: Optional[str] = None   # "active" | "shadow"; default = EVOLUTION_MODE


@router.post("/sleep")
async def sleep_now(body: SleepRequest, user: UserContext = Depends(get_current_user)) -> Dict[str, Any]:
    """Start a sleep cycle for this user in the background (progress arrives as events)."""
    import threading
    from server.idle import idle_tracker
    from server.scheduler import run_cycle_for_user

    if body.mode not in (None, "active", "shadow"):
        raise HTTPException(status_code=400, detail="mode must be 'active' or 'shadow'.")
    pool.get_or_create(user)  # make sure the user's Peter is loaded
    if idle_tracker.is_sleeping(user.user_id):
        return {"started": False, "reason": "already sleeping"}
    threading.Thread(target=idle_tracker.run_sleep,
                     args=(user.user_id, run_cycle_for_user, "manual", body.mode),
                     daemon=True, name=f"Sleep-{user.user_id}").start()
    return {"started": True, "mode": body.mode or "default"}
