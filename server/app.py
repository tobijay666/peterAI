"""
Peter AI — FastAPI application.

Start:
    uvicorn server.app:app --reload --port 8000

The frontend (web/dist/) is served as static files when it exists.
During development, Vite's dev server handles the frontend and the proxy
in vite.config.ts forwards /api and /ws requests here.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()  # must run before any os.getenv() calls in routers

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from core.logger import logger
from server.routes.auth_routes     import router as auth_router
from server.routes.chat_routes     import router as chat_router
from server.routes.curiosity_routes import router as curiosity_router
from server.routes.feedback_routes  import router as feedback_router
from server.routes.memory_routes    import router as memory_router
from server.routes.mind_routes     import router as mind_router
from server.ws                     import router as ws_router


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    application = FastAPI(
        title="Peter AI",
        description="Self-learning multi-agent assistant — web API",
        version="1.4.0",
    )

    # CORS — allow the Vite dev server during development
    _origins = os.getenv(
        "CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")

    application.add_middleware(
        CORSMiddleware,
        allow_origins=_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # API & WebSocket routes
    application.include_router(auth_router)
    application.include_router(chat_router)
    application.include_router(curiosity_router)
    application.include_router(feedback_router)
    application.include_router(memory_router)
    application.include_router(mind_router)
    application.include_router(ws_router)

    # Start the nightly evolution scheduler
    @application.on_event("startup")
    async def _start_scheduler() -> None:
        import asyncio
        from server.ws import set_event_loop
        set_event_loop(asyncio.get_running_loop())
        from server.scheduler import nightly
        nightly.start()
        logger.info("[server] NightlyScheduler started.")
        from server.idle import idle_tracker
        idle_tracker.start()
        logger.info("[server] Idle sleep trigger started.")

    @application.on_event("shutdown")
    async def _stop_scheduler() -> None:
        from server.scheduler import nightly
        nightly.stop()
        from server.idle import idle_tracker
        idle_tracker.stop()

    # Serve the built frontend when dist/ exists (production mode)
    dist = Path(__file__).resolve().parent.parent / "web" / "dist"
    if dist.exists():
        application.mount("/", StaticFiles(directory=str(dist), html=True), name="static")
        logger.info(f"[server] Serving frontend from {dist}")
    else:
        logger.info(
            "[server] No web/dist found — API-only mode. "
            "Run 'cd web && npm run build' for production."
        )

    return application


app = create_app()
