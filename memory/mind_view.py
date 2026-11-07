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


LAYOUTS = ("graph", "cluster", "tree", "semantic")
TOPIC_KINDS = ("entity", "abstraction")


def _layout_signature(ids: List[str], edges: List[Dict[str, Any]]) -> str:
    """A hash of the drawn node set and edge set (src|dst|relation). Weights are left out on
    purpose: a weight refresh must not move the picture. Unchanged signature -> stored positions."""
    import hashlib
    h = hashlib.sha1()
    h.update("\x1f".join(sorted(ids)).encode())
    h.update(b"\x1e")
    h.update("\x1f".join(sorted(f"{e['src']}|{e['dst']}|{e['relation']}" for e in edges)).encode())
    return h.hexdigest()


def _graph_layout(cognition: Any, graph: Any, layout: Any, nodes: List[Dict[str, Any]], ids: List[str],
                  all_edges: List[Dict[str, Any]], self_edges: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Force-directed positions for the "graph" layout, stable between snapshots (D2):
      * signature unchanged and every node already placed -> return stored positions, no rewrite;
      * a few new nodes / changed edges -> relax only those and their neighbours (rest pinned);
      * otherwise (fresh, or most nodes new) -> one convergent deterministic layout.
    """
    from memory.graph_layout import graph_positions
    from memory.hierarchy import SELF_ID
    layout_edges = all_edges + self_edges
    previous = {n["id"]: (float(n["gpos_x"]), float(n["gpos_y"]), float(n["gpos_z"]))
                for n in nodes if n.get("gpos_x") is not None}
    placed = set(previous)
    sig = _layout_signature(ids, layout_edges)
    stored_sig = graph.get_meta("graph_layout_sig")
    if stored_sig == sig and all(i in placed for i in ids):
        return previous                                        # nothing changed: hold the picture still

    edge_key = lambda e: f"{e['src']}|{e['dst']}|{e['relation']}"
    cur_keys = {edge_key(e) for e in layout_edges}
    old_keys_raw = graph.get_meta("graph_layout_edges")
    old_keys = set(old_keys_raw.split("\n")) if old_keys_raw else set()
    changed_edges = cur_keys ^ old_keys if old_keys else cur_keys

    new_nodes = {i for i in ids if i not in placed}
    initial: Dict[str, Any] = {}
    for n in nodes:
        if n["id"] in new_nodes:
            try:
                initial[n["id"]] = layout.place(n) if layout is not None else None
            except Exception:
                initial[n["id"]] = None

    fresh = not placed or len(new_nodes) > 0.5 * len(ids)
    if fresh:
        tree_pos = graph_positions(ids + [SELF_ID], layout_edges, initial, previous, fresh=True)
    else:
        # free = new nodes, the endpoints of added/removed edges, and their 1-hop neighbours;
        # everything else keeps its stored position so the picture does not drift
        touched = set(new_nodes)
        for key in changed_edges:
            s, d, _ = key.split("|", 2)
            if s in placed or s in new_nodes:
                touched.add(s)
            if d in placed or d in new_nodes:
                touched.add(d)
        adj: Dict[str, set] = {}
        for e in layout_edges:
            adj.setdefault(e["src"], set()).add(e["dst"])
            adj.setdefault(e["dst"], set()).add(e["src"])
        free = set(touched)
        for t in touched:
            free |= adj.get(t, set())
        free.discard(SELF_ID)
        free &= set(ids)
        if not free:
            graph.set_meta("graph_layout_sig", sig)            # structure changed but nothing to move
            graph.set_meta("graph_layout_edges", "\n".join(sorted(cur_keys)))
            return previous
        pinned = (placed | new_nodes) - free
        tree_pos = graph_positions(ids + [SELF_ID], layout_edges, initial, previous,
                                   pinned=pinned, fresh=False)

    moved = {i: p for i, p in tree_pos.items() if previous.get(i) != p}
    try:
        if moved:
            graph.set_graph_positions(moved)
        graph.set_meta("graph_layout_sig", sig)
        graph.set_meta("graph_layout_edges", "\n".join(sorted(cur_keys)))
    except Exception:
        pass
    return tree_pos


def graph_snapshot(cognition: Any, include_sources: bool = False,
                   layout_mode: Optional[str] = None) -> Dict[str, Any]:
    """layout_mode: "graph" (3-D force-directed layout of the stored vector graph: springs on
    stored edges, shorter for more similar nodes, Peter fixed at the origin; default),
    "cluster" (3-D tree arranged by similarity), "tree" (flat top-down tree diagrams) or
    "semantic" (meaning only); default from config visual.layout.

    Edges: every stored edge between drawn nodes, with relation, vector weight and the
    primary flag, plus Peter's own has_subject edges (src ``self_peter``; the view draws
    Peter as the core sphere at the origin). Nothing is drawn that is not stored."""
    from memory.hierarchy import SELF_ID
    graph = cognition.graph
    kinds = GRAPH_KINDS + (["source"] if include_sources else [])
    nodes = [n for n in graph.all_nodes(kinds) if n.get("status") != "pruned"]
    scored = cognition.usage.recompute_all(persist=False)
    pct = {k: v["percentile"] for k, v in scored.items()}
    colls = {c["name"]: c for c in graph.list_collections()}
    layout = getattr(cognition, "layout", None)
    mode = layout_mode or cognition_config()["visual"].get("layout", "graph")
    if mode not in LAYOUTS:
        mode = "graph"
    # Helper agents (researcher, librarian…) are plumbing, never drawn even as a topic node
    nodes = [n for n in nodes if (n.get("label") or "").strip().lower() not in _AGENT_NAMES]
    ids = [n["id"] for n in nodes]
    id_set = set(ids)
    all_edges = graph.list_edges(ids)
    # Peter's own stored edges (has_subject) to drawn subjects; Peter is the core at the origin
    self_edges = [e for e in graph.list_edges([SELF_ID] + ids) if SELF_ID in (e["src"], e["dst"])
                  and (e["dst"] if e["src"] == SELF_ID else e["src"]) in id_set and e["relation"] != "helper_of"]
    stored_subjects = sorted({e["dst"] for e in self_edges if e["relation"] == "has_subject" and e["src"] == SELF_ID})
    core_links: List[str] = stored_subjects            # only stored edges, never inferred
    if mode == "tree":
        from memory.tree_layout import tree_positions
        tree_pos = tree_positions(nodes, all_edges)
    elif mode == "cluster":
        # 3-D: the tree hierarchy, with similar meanings placed close together
        from memory.tree_layout import cluster_positions
        semantic = {n["id"]: layout.place(n) for n in nodes} if layout is not None else {}
        tree_pos = cluster_positions(nodes, all_edges, semantic)
    elif mode == "graph":
        tree_pos = _graph_layout(cognition, graph, layout, nodes, ids, all_edges, self_edges)
    else:
        tree_pos = {}

    out_nodes: List[Dict[str, Any]] = []
    empty = 0
    for n in nodes:
        style = _style_for(n, pct)
        if mode in ("tree", "cluster", "graph"):
            pos = tree_pos.get(n["id"], (0.0, 0.0, 0.0))
        else:
            pos = layout.place(n) if layout is not None else (0.0, 0.0, 0.0)
        coll = colls.get(n.get("collection") or "")
        count = int(coll["record_count"]) if coll else 0
        is_empty = n["kind"] in TOPIC_KINDS and count == 0
        empty += 1 if is_empty else 0
        out_nodes.append({
            "id": n["id"], "kind": n["kind"], "label": (n.get("label") or "")[:120],
            "status": n.get("status", "active"), "pinned": bool(n.get("pinned")),
            "access_count": int(n.get("access_count") or 0),
            "usage_score": round(float(scored.get(n["id"], {}).get("score", n.get("usage_score") or 0.0)), 4),
            "percentile": round(pct.get(n["id"], 0.0), 4) if n["id"] in pct else None,
            "tier": style["tier"], "style": style,
            "position": [round(float(x), 3) for x in pos],
            "collection": n.get("collection"),
            "record_count": count,
            "empty": is_empty,              # a topic/hub whose vector store has no knowledge yet
            "confidence": round(float(n.get("confidence") if n.get("confidence") is not None else 0.5), 3),
        })

    def edge_out(e: Dict[str, Any]) -> Dict[str, Any]:
        d = e.get("data") or {}
        w = d.get("weight") if isinstance(d, dict) else None
        return {"id": e["id"], "src": e["src"], "dst": e["dst"], "relation": e["relation"],
                "weight": round(float(w), 4) if w is not None else None,
                "primary": bool(d.get("primary")) if isinstance(d, dict) else False}

    edges = [edge_out(e) for e in all_edges] + [edge_out(e) for e in self_edges]
    by_kind: Dict[str, int] = {}
    for n in out_nodes:
        by_kind[n["kind"]] = by_kind.get(n["kind"], 0) + 1
    semantic_ok = bool(getattr(cognition, "semantic_ok", True))
    return {
        "layout": mode,
        "cognition": {"semantic_ok": semantic_ok},
        "self": {"id": SELF_ID, "position": [0.0, 0.0, 0.0]},
        "core_links": core_links,   # main subjects: Peter's stored has_subject edges
        "nodes": out_nodes,
        "edges": edges,
        "stats": {"nodes": len(out_nodes), "edges": len(edges), "by_kind": by_kind,
                  "collections": len(colls), "semantic_ok": semantic_ok, "empty_collections": empty},
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
        d = e.get("data") or {}
        edges.append({"id": e["id"], "relation": e["relation"],
                      "direction": "out" if e["src"] == node_id else "in",
                      "node_id": other_id, "label": other["label"] if other else other_id,
                      "kind": other["kind"] if other else None,
                      "weight": d.get("weight") if isinstance(d, dict) else None,
                      "primary": bool(d.get("primary")) if isinstance(d, dict) else False})
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
        "empty": node["kind"] in TOPIC_KINDS and not (coll_info and coll_info["record_count"]),
        "aliases": graph.aliases_of(node_id), "edges": edges, "curiosity": curiosity,
    }
