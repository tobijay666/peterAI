"""
Authentication helpers for the FastAPI server.

Supports two modes, chosen by environment variable:

  AUTH_MODE=jwt (default)
      Signs and verifies tokens locally using a secret key set in .env
      (JWT_SECRET).  Good for a small, self-hosted deployment.

  AUTH_MODE=supabase
      Verifies Supabase JWT tokens using the project's JWT secret
      (SUPABASE_JWT_SECRET).  Set this when you want Supabase Auth
      to handle sign-up, password reset and email verification.

In both modes the token is expected as a standard Bearer header or as
a `token` query parameter (for WebSocket connections).
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, Query, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

try:
    from jose import JWTError, jwt as jose_jwt
    _JOSE_AVAILABLE = True
except ImportError:
    _JOSE_AVAILABLE = False

from core.logger import logger
from core.user_context import UserContext, web_context

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

AUTH_MODE = os.getenv("AUTH_MODE", "jwt").lower()  # "jwt" | "supabase"

JWT_SECRET    = os.getenv("JWT_SECRET", "change-me-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_HOURS = int(os.getenv("JWT_EXPIRE_HOURS", "24"))

SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET", "")

_bearer = HTTPBearer(auto_error=False)

# ---------------------------------------------------------------------------
# Token creation (used by the login endpoint in jwt mode)
# ---------------------------------------------------------------------------

def create_access_token(user_id: str, display_name: str = "User") -> str:
    """Creates a signed JWT for the given user (jwt mode only)."""
    if not _JOSE_AVAILABLE:
        raise RuntimeError("python-jose is not installed. Run: pip install python-jose[cryptography]")
    expire = datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRE_HOURS)
    payload = {
        "sub":          user_id,
        "display_name": display_name,
        "exp":          expire,
    }
    return jose_jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

# ---------------------------------------------------------------------------
# Token verification
# ---------------------------------------------------------------------------

def _verify_token(token: str) -> dict:
    """Decodes and verifies the token; raises HTTPException on failure."""
    if not _JOSE_AVAILABLE:
        raise HTTPException(status_code=500, detail="python-jose not installed.")

    secret = SUPABASE_JWT_SECRET if AUTH_MODE == "supabase" else JWT_SECRET
    if not secret:
        raise HTTPException(status_code=500, detail="JWT secret not configured.")

    try:
        payload = jose_jwt.decode(token, secret, algorithms=[JWT_ALGORITHM])
        return payload
    except JWTError as exc:
        logger.warning(f"[auth] JWT verification failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

# ---------------------------------------------------------------------------
# FastAPI dependencies
# ---------------------------------------------------------------------------

def get_current_user(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> UserContext:
    """FastAPI dependency — extracts and verifies the Bearer token, returns UserContext."""
    if not creds:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    payload = _verify_token(creds.credentials)
    uid  = payload.get("sub") or payload.get("user_id") or ""
    name = payload.get("display_name") or payload.get("email") or "User"
    if not uid:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token payload.")
    return web_context(uid, name)


def get_ws_user(token: str = Query(default="")) -> UserContext:
    """FastAPI dependency for WebSocket connections — token passed as query param."""
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="WebSocket requires a token query parameter.",
        )
    payload = _verify_token(token)
    uid  = payload.get("sub") or payload.get("user_id") or ""
    name = payload.get("display_name") or payload.get("email") or "User"
    if not uid:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token payload.")
    return web_context(uid, name)
