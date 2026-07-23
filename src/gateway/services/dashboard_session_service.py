"""Dashboard sign-in sessions.

Signing in to the admin dashboard exchanges the master key for a server-issued
session: an opaque token handed to the browser in an HttpOnly cookie, with only
its SHA-256 hash stored in the ``dashboard_sessions`` table. This lets a
sign-in survive tab closes and browser restarts without ever persisting the
master key (or any JS-readable credential) in the browser.

Sessions live in the database, not process memory, so every worker and replica
accepts them and a revocation is seen everywhere. They expire on a TTL
(``dashboard_session_ttl_hours``) and are revoked on sign-out and on master-key
rotation.
"""

import hashlib
import secrets
from datetime import UTC, datetime, timedelta

from fastapi import Response
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.models.entities import DashboardSession

SESSION_COOKIE_NAME = "otari_dashboard_session"
_SESSION_TOKEN_PREFIX = "otari-sess-"


def hash_session_token(token: str) -> str:
    """SHA-256 hex of a session token; only the hash is ever stored."""
    return hashlib.sha256(token.encode()).hexdigest()


def _as_utc(value: datetime) -> datetime:
    """Treat a naive stored datetime as the UTC it was written as (SQLite)."""
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


async def create_dashboard_session(db: AsyncSession, ttl_hours: int) -> tuple[str, datetime]:
    """Stage a new session row and return ``(token, expires_at)``.

    Expired rows are pruned opportunistically here, so the table stays small
    without a background task. The caller owns the transaction and must commit
    before handing the token to the browser.
    """
    now = datetime.now(UTC)
    await db.execute(delete(DashboardSession).where(DashboardSession.expires_at < now))
    token = f"{_SESSION_TOKEN_PREFIX}{secrets.token_urlsafe(32)}"
    expires_at = now + timedelta(hours=ttl_hours)
    db.add(DashboardSession(token_hash=hash_session_token(token), created_at=now, expires_at=expires_at))
    return token, expires_at


async def is_valid_dashboard_session(db: AsyncSession, token: str) -> bool:
    """Whether a session token matches a stored, unexpired session."""
    row = await db.get(DashboardSession, hash_session_token(token))
    if row is None:
        return False
    return _as_utc(row.expires_at) >= datetime.now(UTC)


async def revoke_dashboard_session(db: AsyncSession, token: str) -> None:
    """Stage removal of one session (sign-out). The caller commits."""
    await db.execute(delete(DashboardSession).where(DashboardSession.token_hash == hash_session_token(token)))


async def revoke_all_dashboard_sessions(db: AsyncSession) -> None:
    """Stage removal of every session (master-key rotation). The caller commits."""
    await db.execute(delete(DashboardSession))


def apply_session_cookie(response: Response, token: str, expires_at: datetime, *, secure: bool) -> None:
    """Set the session cookie with its security attributes in one place.

    ``secure`` mirrors the request scheme rather than being hard-coded: a plain
    HTTP deployment (LAN, localhost without TLS) would otherwise never receive
    the cookie back. That is no worse than such a deployment already sending the
    raw master key in cleartext today. ``SameSite=Strict`` keeps cross-site
    requests from carrying the cookie, which is the primary CSRF control here
    (the dashboard and API are same-origin).
    """
    max_age = max(0, int((expires_at - datetime.now(UTC)).total_seconds()))
    response.set_cookie(
        SESSION_COOKIE_NAME,
        token,
        max_age=max_age,
        httponly=True,
        secure=secure,
        samesite="strict",
        path="/",
    )


def clear_session_cookie(response: Response) -> None:
    """Expire the session cookie in the browser (sign-out)."""
    response.delete_cookie(SESSION_COOKIE_NAME, path="/")
