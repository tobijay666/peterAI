"""
WebSocket endpoint — streams live mind events to the browser.

/ws/mind?token=<jwt>

Each authenticated user gets their own queue.  Events are JSON-serialised
dicts pushed by core.events.emit() (added in Phase 2).

Phase 1 stub: the socket connects and stays open but no events flow yet
because the emit() hooks haven't been wired into the agent loop.
The client is designed to handle this gracefully.
"""

from __future__ import annotations

import asyncio
import json
from typing import Dict, Optional

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

from core.logger import logger
from core.user_context import UserContext
from server.auth import get_ws_user

router = APIRouter(tags=["websocket"])

# Per-user asyncio queues populated by the event bridge (Phase 2)
_queues: Dict[str, asyncio.Queue] = {}

# The server's event loop, captured when the first socket connects.
# emit() is called from agent / worker threads, and asyncio.Queue is not
# thread-safe, so pushes from other threads are marshalled onto this loop.
_loop: Optional[asyncio.AbstractEventLoop] = None


def set_event_loop(loop: asyncio.AbstractEventLoop) -> None:
    """Record the server loop (called on startup and on each WS connect)."""
    global _loop
    _loop = loop


def get_user_queue(user_id: str) -> asyncio.Queue:
    """Returns (creating if necessary) the asyncio queue for a user."""
    if user_id not in _queues:
        _queues[user_id] = asyncio.Queue(maxsize=512)
    return _queues[user_id]


def push_event(user_id: str, event: dict) -> None:
    """
    Called from synchronous code (the agent loop, Phase 2) to push an event
    onto the user's queue.  Uses put_nowait and drops when the queue is full
    so the agent loop is never blocked.
    """
    q = get_user_queue(user_id)

    def _put() -> None:
        try:
            q.put_nowait(event)
        except asyncio.QueueFull:
            pass  # drop — the browser will re-fetch a snapshot on reconnect

    loop = _loop
    if loop is None or loop.is_closed():
        _put()  # no server loop yet (tests / REPL): plain put is safe
        return
    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        running = None
    if running is loop:
        _put()
    else:
        loop.call_soon_threadsafe(_put)


@router.websocket("/ws/mind")
async def ws_mind(websocket: WebSocket, user: UserContext = Depends(get_ws_user)) -> None:
    await websocket.accept()
    set_event_loop(asyncio.get_running_loop())
    logger.info(f"[ws] User '{user.user_id}' connected to mind stream.")
    q = get_user_queue(user.user_id)
    try:
        while True:
            # Wait up to 20 s for an event, then send a keepalive ping
            try:
                event = await asyncio.wait_for(q.get(), timeout=20.0)
                await websocket.send_text(json.dumps(event))
            except asyncio.TimeoutError:
                await websocket.send_text(json.dumps({"type": "ping"}))
    except WebSocketDisconnect:
        logger.info(f"[ws] User '{user.user_id}' disconnected.")
    except Exception as e:
        logger.error(f"[ws] Error for '{user.user_id}': {e}")
