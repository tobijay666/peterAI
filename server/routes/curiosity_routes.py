"""
Curiosity queue endpoints (cognitive memory, Stage 4).

GET  /api/curiosity?status=pending&limit=50
        The user's curiosity items, highest effective priority first.

POST /api/curiosity/{item_id}   body: {"action": "boost" | "dismiss"}
        Boost raises an item's priority; dismiss removes it and halves the
        priority of near-duplicates (and blocks new twins).
"""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from core.user_context import UserContext
from server.auth import get_current_user
from server.peter_pool import pool

router = APIRouter(prefix="/api/curiosity", tags=["curiosity"])


def _engine(user: UserContext):
    peter, _, _, _ = pool.get_or_create(user)
    cognition = getattr(peter, "cognition", None)
    if cognition is None or cognition.curiosity is None:
        raise HTTPException(status_code=503, detail="Cognitive memory is not enabled.")
    return cognition.curiosity


@router.get("")
async def list_curiosity(status: str = "pending", limit: int = 50,
                         user: UserContext = Depends(get_current_user)) -> Dict[str, Any]:
    if status not in ("pending", "researching", "done", "dismissed", "dropped"):
        raise HTTPException(status_code=400, detail="Unknown status.")
    items = _engine(user).list(status=status, limit=max(1, min(limit, 500)))
    return {"user_id": user.user_id, "status": status, "items": items}


class CuriosityAction(BaseModel):
    action: str


@router.post("/{item_id}")
async def act_on_curiosity(item_id: str, body: CuriosityAction,
                           user: UserContext = Depends(get_current_user)) -> Dict[str, Any]:
    engine = _engine(user)
    if body.action == "boost":
        item = engine.boost(item_id)
    elif body.action == "dismiss":
        item = engine.dismiss(item_id)
    else:
        raise HTTPException(status_code=400, detail="action must be 'boost' or 'dismiss'.")
    if item is None:
        raise HTTPException(status_code=404, detail="Curiosity item not found.")
    return {"ok": True, "item": item}
