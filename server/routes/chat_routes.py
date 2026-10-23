"""
Chat endpoint.

POST /api/chat
    Sends a message to the user's private Peter instance and returns the
    response plus a task_id (used by the feedback buttons).
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import Optional

from core.user_context import UserContext
from server.auth import get_current_user
from server.peter_pool import pool

router = APIRouter(prefix="/api", tags=["chat"])


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str
    user_id:  str
    task_id:  Optional[str] = None


@router.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest, user: UserContext = Depends(get_current_user)) -> ChatResponse:
    peter, _guardrails, _router, learning = pool.get_or_create(user)

    # Activity resets the idle timer and pauses a running sleep cycle
    from server.idle import idle_tracker
    idle_tracker.mark_active(user.user_id)

    # Generate a task_id for this turn so the frontend can attach feedback to it
    task_id = str(uuid.uuid4())

    response = peter.run(user_input=body.message)

    # Phase 6: fire-and-forget learning
    try:
        learning.on_turn(body.message, response, task_id)
    except Exception:
        pass  # learning failures are non-fatal

    # Record a task-level outcome so feedback has something to update
    try:
        from memory.outcomes import OutcomeLedger
        ledger = OutcomeLedger(db_path=str(user.graph_db_path))
        ledger.record(
            task_id=task_id,
            level="task",
            agent="peter",
            status="success",
        )
    except Exception:
        pass  # ledger failures are non-fatal

    return ChatResponse(response=response, user_id=user.user_id, task_id=task_id)
