"""
Feedback and morning-report endpoints.

POST /api/feedback
    Accepts {task_id, rating} where rating is "good" or "bad".
    Routes to the FeedbackEngine which records it in the OutcomeLedger
    and adjusts memory importance accordingly.

GET /api/report/morning
    Returns the latest evolution journal entries — what Peter learned
    overnight — so the frontend can show the "While you were away" panel.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, List, Optional

from core.user_context import UserContext
from core.logger import logger
from server.auth import get_current_user
from server.peter_pool import pool

router = APIRouter(prefix="/api", tags=["feedback"])


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------

class FeedbackRequest(BaseModel):
    task_id: str
    rating:  str          # "good" | "bad"
    note:    Optional[str] = None


class FeedbackResponse(BaseModel):
    ok:      bool
    task_id: str
    rating:  str


@router.post("/feedback", response_model=FeedbackResponse)
async def feedback(
    body: FeedbackRequest,
    user: UserContext = Depends(get_current_user),
) -> FeedbackResponse:
    """Record good or bad feedback for a completed task."""
    if body.rating not in ("good", "bad"):
        raise HTTPException(status_code=422, detail="rating must be 'good' or 'bad'.")

    peter, _guardrails, _router, _learning = pool.get_or_create(user)

    try:
        from memory.outcomes import OutcomeLedger
        from engine.feedback import FeedbackEngine
        ledger  = OutcomeLedger(db_path=str(user.graph_db_path))
        fb_eng  = FeedbackEngine(ledger=ledger, vector=peter.vector)
        if body.rating == "good":
            fb_eng.record_good(body.task_id)
        else:
            fb_eng.record_bad(body.task_id, note=body.note)
    except Exception as e:
        logger.error(f"[feedback] Error recording feedback for {body.task_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Feedback recording failed: {e}")

    return FeedbackResponse(ok=True, task_id=body.task_id, rating=body.rating)


# ---------------------------------------------------------------------------
# Morning report
# ---------------------------------------------------------------------------

class ReportEntry(BaseModel):
    step:      str
    ts:        Optional[float] = None
    mode:      Optional[str]   = None
    note:      Optional[str]   = None
    path:      Optional[str]   = None
    skill_id:  Optional[str]   = None


@router.get("/report/morning")
async def morning_report(
    user: UserContext = Depends(get_current_user),
    last_n: int = 20,
) -> Dict[str, Any]:
    """Returns the latest evolution journal entries for the morning report panel."""
    journal_path = user.data_root / "users" / user.user_id / "evolution" / "journal.jsonl"
    if user.is_anonymous:
        journal_path = user.data_root / "evolution" / "journal.jsonl"

    entries: List[Dict[str, Any]] = []
    if journal_path.exists():
        try:
            import json
            lines = journal_path.read_text(encoding="utf-8").splitlines()
            for line in lines[-last_n:]:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except Exception:
                        pass
        except Exception as e:
            logger.warning(f"[morning_report] Could not read journal: {e}")

    return {"user_id": user.user_id, "entries": entries}
