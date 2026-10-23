"""
Memory browser endpoints.

GET    /api/memory/search?q=<query>&n=<int>&ns=<namespace>
         Searches the user's vector memory and returns matching records.

GET    /api/memory/stats
         Returns per-namespace counts, total, kind breakdown, and limit.

POST   /api/memory/pin
         Pins or unpins a memory record.

DELETE /api/memory/clear[?namespace=<ns>]
         Deletes all unpinned records, optionally scoped to one namespace.

DELETE /api/memory/{memory_id}
         Deletes a single (unpinned) memory record.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from core.user_context import UserContext
from core.logger import logger
from server.auth import get_current_user
from server.peter_pool import pool

router = APIRouter(prefix="/api/memory", tags=["memory"])


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

@router.get("/search")
async def search_memory(
    q:    str,
    n:    int             = 10,
    ns:   Optional[str]  = Query(None),
    user: UserContext     = Depends(get_current_user),
) -> Dict[str, Any]:
    """Search the user's memory across all (or one) namespace(s)."""
    peter, _, _, _l = pool.get_or_create(user)
    vector = peter.vector

    namespaces = [ns] if ns else vector.list_namespaces()

    results: List[Dict[str, Any]] = []
    seen_ids: set = set()

    for namespace in namespaces:
        try:
            hits = vector.recall(namespace=namespace, query=q, n_results=n)
            for h in hits:
                if h["id"] not in seen_ids:
                    seen_ids.add(h["id"])
                    results.append({
                        "id":         h["id"],
                        "document":   h["document"],
                        "namespace":  namespace,
                        "agent":      h["metadata"].get("agent", namespace),
                        "importance": float(h["metadata"].get("importance", 0.5)),
                        "pinned":     bool(h["metadata"].get("pinned", False)),
                        "kind":       h["metadata"].get("kind", "finding"),
                        "score":      float(h.get("score", 0.0)),
                    })
        except Exception as e:
            logger.warning(f"memory search: namespace '{namespace}' error: {e}")

    # Sort by score desc, return top n
    results.sort(key=lambda r: -r["score"])
    return {"user_id": user.user_id, "query": q, "results": results[:n]}


# ---------------------------------------------------------------------------
# Memory stats  ← MUST be before /{memory_id}
# ---------------------------------------------------------------------------

@router.get("/stats")
async def memory_stats(
    user: UserContext = Depends(get_current_user),
) -> Dict[str, Any]:
    """Return per-namespace counts, total count, kind breakdown, and limit."""
    peter, _, _, _l = pool.get_or_create(user)
    from server.learning import get_memory_stats
    return get_memory_stats(peter, user)


# ---------------------------------------------------------------------------
# Pin / unpin
# ---------------------------------------------------------------------------

class PinRequest(BaseModel):
    memory_id: str
    pinned:    bool = True


@router.post("/pin")
async def pin_memory(
    body: PinRequest,
    user: UserContext = Depends(get_current_user),
) -> Dict[str, Any]:
    peter, _, _, _l = pool.get_or_create(user)
    vector = peter.vector
    success = False
    for ns in vector.list_namespaces():
        ok = vector.pin(namespace=ns, doc_id=body.memory_id, pinned=body.pinned)
        if ok:
            success = True
            break
    if not success:
        raise HTTPException(status_code=404, detail="Memory record not found.")
    return {"ok": True, "memory_id": body.memory_id, "pinned": body.pinned}


# ---------------------------------------------------------------------------
# Bulk clear  ← MUST be before /{memory_id}
# ---------------------------------------------------------------------------

@router.delete("/clear")
async def clear_memory(
    namespace: Optional[str] = Query(None),
    user: UserContext = Depends(get_current_user),
) -> Dict[str, Any]:
    """Delete all unpinned records.  Pass ?namespace=peter to clear one namespace."""
    peter, _, _, _l = pool.get_or_create(user)
    vector = peter.vector
    deleted = 0
    namespaces = [namespace] if namespace else vector.list_namespaces()
    for ns in namespaces:
        try:
            recs = vector.get_all_records(ns)
            ids_to_del = [r["id"] for r in recs if not r.get("metadata", {}).get("pinned", False)]
            deleted += vector.prune(namespace=ns, doc_ids=ids_to_del)
        except Exception as e:
            logger.warning(f"clear_memory: namespace '{ns}' error: {e}")
    return {"ok": True, "deleted": deleted}


# ---------------------------------------------------------------------------
# Delete single record  ← MUST be last (catches /{memory_id})
# ---------------------------------------------------------------------------

@router.delete("/{memory_id}")
async def delete_memory(
    memory_id: str,
    user:      UserContext = Depends(get_current_user),
) -> Dict[str, Any]:
    peter, _, _, _l = pool.get_or_create(user)
    vector = peter.vector
    deleted = 0
    for ns in vector.list_namespaces():
        deleted += vector.prune(namespace=ns, doc_ids=[memory_id])
    if not deleted:
        raise HTTPException(status_code=404, detail="Memory record not found or is pinned.")
    return {"ok": True, "memory_id": memory_id, "deleted": deleted}
