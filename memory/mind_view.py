"""Mind-view data for the 3-D knowledge web (Stage 6).

``graph_snapshot``   every visible node with its server-decided style and stable
                     position, plus the edges between them (one edge query).
``node_detail``      the inspection panel: usage, collection, records with
                     sources and confidence, an embedding preview, edges,
                     curiosity items.

Pure functions over a CognitiveMemory, so they are tested without HTTP.
"""
from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional

from core.cognition_config import cognition_config
from memory.visual_style import visual_style

# Agent nodes (researcher, librarian…) are plumbing, not knowledge: not drawn
GRAPH_KINDS = ["entity", "abstraction", "curiosity", "skill"]
from memory.hierarchy import HELPER_NAMES as _AGENT_NAMES  # noqa: E402  (helpers are plumbing)

# Fixed looks for nodes that are not usage-weighted
FIXED_STYLES = {
    "curiosity": {"tier": "curiosity", "scale": 0.35, "emissive": 0.25, "saturation": 0.6, "opacity": 0.45},
    "source":    {"tier": "source", "scale": 0.25, "emissive": 0.1, "saturation": 0.4, "opacity": 0.5},
    "agent":     {"tier": "agent", "scale": 1.1, "emissive": 0.6, "saturation": 0.9, "opacity": 1.0},
    "skill":     {"tier": "skill", "scale": 0.8, "emissive": 0.5, "saturation": 0.9, "opacity": 0.9},
}


def _style_for(node: Dict[str, Any], pct: Dict[str, float]) -> Dict[str, Any]:
    if node["kind"] in ("entity", "abstraction"):
        return visual_style(pct.get(node["id"], 0.5), node.get("status", "active"))
    return dict(FIXED_STYLES.get(node["kind"], FIXED_STYLES["skill"]))


def graph_snapshot(cognition: Any, include_sources: bool = False,
                   layout_mode: Optional[str] = None) -> Dict[str, Any]:
    """layout_mode: "cluster" (3-D tree arranged by similarity, default), "tree" (flat
    top-down tree diagrams) or "semantic" (meaning only, no hierarchy);
    default from config visual.layout."""
    graph = cognition.graph
    kinds = GRAPH_KINDS + (["source"] if include_sources else [])
    nodes = [n for n in graph.all_nodes(kinds) if n.get("status") != "pruned"]
    scored = cognition.usage.recompute_all(persist=False)
    pct = {k: v["percentile"] for k, v in scored.items()}
    colls = {c["name"]: c for c in graph.list_collections()}
    layout = getattr(cognition, "layout", None)
    mode = layout_mode or cognition_config()["visual"].get("layout", "cluster")
    # Helper agents (researcher, librarian…) are plumbing, never drawn even as a topic node
    nodes = [n for n in nodes if (n.get("label") or "").strip().lower() not in _AGENT_NAMES]
    from memory.tree_layout import core_subjects, tree_positions
    all_edges = graph.list_edges([n["id"] for n in nodes])
    # Main subjects = stored Peter --has_subject--> edges (memory/hierarchy.py), in layout order
    from memory.hierarchy import SELF_ID
    stored = {e["dst"] for e in graph.edges_of(SELF_ID) if e["relation"] == "has_subject" and e["src"] == SELF_ID}
    core_links: List[str] = [i for i in core_subjects(nodes, all_edges) if i in stored]
    if mode == "tree":
        tree_pos = tree_positions(nodes, all_edges)
    elif mode == "cluster":
        # 3-D: the tree hierarchy, with similar meanings placed close together
        from memory.tree_layout import cluster_positions
        semantic = {n["id"]: layout.place(n) for n in nodes} if layout is not None else {}
        tree_pos = cluster_positions(nodes, all_edges, semantic)
    else:
        tree_pos = {}

    out_nodes: List[Dict[str, Any]] = []
    for n in nodes:
        style = _style_for(n, pct)
        if mode in ("tree", "cluster"):
            pos = tree_pos.get(n["id"], (0.0, 0.0, 0.0))
        else:
            pos = layout.place(n) if layout is not None else (0.0, 0.0, 0.0)
        coll = colls.get(n.get("collection") or "")
        out_nodes.append({
            "id": n["id"], "kind": n["kind"], "label": (n.get("label") or "")[:120],
            "status": n.get("status", "active"), "pinned": bool(n.get("pinned")),
            "access_count": int(n.get("access_count") or 0),
            "usage_score": round(float(scored.get(n["id"], {}).get("score", n.get("usage_score") or 0.0)), 4),
            "percentile": round(pct.get(n["id"], 0.0), 4) if n["id"] in pct else None,
            "tier": style["tier"], "style": style,
            "position": [round(float(x), 3) for x in pos],
            "collection": n.get("collection"),
            "record_count": int(coll["record_count"]) if coll else 0,
            "confidence": round(float(n.get("confidence") if n.get("confidence") is not None else 0.5), 3),
        })
    ids = [n["id"] for n in out_nodes]
    edges = [{"id": e["id"], "src": e["src"], "dst": e["dst"], "relation": e["relation"]}
             for e in graph.list_edges(ids)]
    by_kind: Dict[str, int] = {}
    for n in out_nodes:
        by_kind[n["kind"]] = by_kind.get(n["kind"], 0) + 1
    return {
        "layout": mode,
        "core_links": core_links,   # main subjects, drawn as lines from Peter's core
        "nodes": out_nodes,
        "edges": edges,
        "stats": {"nodes": len(out_nodes), "edges": len(edges), "by_kind": by_kind,
                  "collections": len(colls)},
    }


def node_detail(cognition: Any, node_id: str, max_records: int = 10) -> Optional[Dict[str, Any]]:
    graph, vector = cognition.graph, cognition.vector
    node = graph.get_node(node_id)
    if node is None:
        return None
    scored = cognition.usage.recompute_all(persist=False)
    pct = {k: v["percentile"] for k, v in scored.items()}
    style = _style_for(node, pct)

    records: List[Dict[str, Any]] = []
    coll_info = None
    ns = node.get("collection")
    if ns and vector.has_collection(ns):
        recs = vector.get_all_records(ns)
        recs.sort(key=lambda r: (-float(r["metadata"].get("confidence", 0.5)), r["id"]))
        coll_info = {"name": ns, "record_count": len(recs)}
        for r in recs[:max_records]:
            meta = r.get("metadata") or {}
            emb = r.get("embedding") or []
            try:
                sources = json.loads(meta.get("sources", "[]")) if meta.get("sources") else []
            except Exception:
                sources = []
            if not sources and meta.get("source_url"):
                sources = [{"url": meta["source_url"], "title": meta.get("source_title", "")}]
            records.append({
                "id": r["id"], "text": r["document"], "kind": meta.get("kind", "claim"),
                "confidence": float(meta.get("confidence", 0.5)), "trust": meta.get("trust", "untrusted_web"),
                "sources": sources, "origin_node": meta.get("origin_node"),
                "embedding_preview": [round(float(x), 4) for x in emb[:8]],
                "embedding_norm": round(math.sqrt(sum(float(x) ** 2 for x in emb)), 4) if emb else 0.0,
                "embedding_dims": len(emb),
            })

    edges = []
    for e in graph.edges_of(node_id):
        other_id = e["dst"] if e["src"] == node_id else e["src"]
        other = graph.get_node(other_id)
        edges.append({"id": e["id"], "relation": e["relation"],
                      "direction": "out" if e["src"] == node_id else "in",
                      "node_id": other_id, "label": other["label"] if other else other_id,
                      "kind": other["kind"] if other else None})
    curiosity = []
    if getattr(cognition, "curiosity", None) is not None:
        curiosity = [{"id": c["id"], "question": c["question"], "status": c["status"],
                      "priority": c.get("effective_priority", c["priority"])}
                     for c in cognition.curiosity.list("pending", limit=200) if c.get("topic_node_id") == node_id]
    sources = sorted({s["url"] for r in records for s in r["sources"] if s.get("url")})
    return {
        "id": node["id"], "kind": node["kind"], "label": node["label"], "status": node.get("status", "active"),
        "pinned": bool(node.get("pinned")),
        "usage": {"score": round(float(scored.get(node_id, {}).get("score", 0.0)), 4),
                  "percentile": round(pct[node_id], 4) if node_id in pct else None,
                  "access_count": int(node.get("access_count") or 0),
                  "last_accessed": node.get("last_accessed"), "created_at": node.get("created_at")},
        "tier": style["tier"], "style": style,
        "confidence": round(float(node.get("confidence") if node.get("confidence") is not None else 0.5), 3),
        "collection": coll_info, "records": records, "sources": sources,
        "aliases": graph.aliases_of(node_id), "edges": edges, "curiosity": curiosity,
    }
