"""
Authentication endpoints.

POST /api/auth/login
    Accepts {email, password} and returns {access_token}.

    In jwt mode this validates credentials against a tiny in-memory user
    store driven by environment variables — suitable for a small group of
    known users.  Replace this function with a real DB lookup when needed.

    In supabase mode the frontend should call Supabase directly; this
    endpoint is then unused.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from typing import Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from server.auth import AUTH_MODE, create_access_token

router = APIRouter(prefix="/api/auth", tags=["auth"])


# ---------------------------------------------------------------------------
# Simple in-process user store (jwt mode)
# ---------------------------------------------------------------------------
# Users are defined as a JSON string in the environment variable PETER_USERS:
#   [{"id": "u1", "email": "user@example.com", "password_hash": "<sha256hex>", "name": "Alice"}]
#
# Generate a hash:  python3 -c "import hashlib; print(hashlib.sha256(b'mypassword').hexdigest())"

def _load_users() -> list:
    raw = os.getenv("PETER_USERS", "[]")
    try:
        return json.loads(raw)
    except Exception:
        return []


def _find_user(email: str, password: str) -> Optional[dict]:
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    for u in _load_users():
        if u.get("email", "").lower() == email.lower():
            stored = u.get("password_hash", "")
            if hmac.compare_digest(stored, password_hash):
                return u
    return None


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class LoginRequest(BaseModel):
    email:    str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type:   str = "bearer"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/login", response_model=LoginResponse)
async def login(body: LoginRequest) -> LoginResponse:
    if AUTH_MODE == "supabase":
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Use Supabase Auth directly in supabase mode.",
        )

    user = _find_user(body.email, body.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password.",
        )
    token = create_access_token(
        user_id=user["id"],
        display_name=user.get("name", "User"),
    )
    return LoginResponse(access_token=token)
