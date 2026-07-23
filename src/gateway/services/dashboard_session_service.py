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
from typing import Any, cast

from fastapi import Request, Response
from sqlalchemy import delete, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.config import GatewayConfig
from gateway.log_config import logger
from gateway.models.entities import DashboardSession, RuntimeSetting
from gateway.services.master_key_service import hash_master_key

SESSION_COOKIE_NAME = "otari_dashboard_session"
_SESSION_TOKEN_PREFIX = "otari-sess-"
# Stored in runtime_settings; ignored by runtime_settings_service (not a
# SETTABLE_KEY). Hash of the master key that existing sessions were minted
# under, so a key change across a restart revokes them (see
# revoke_sessions_on_master_key_change).
SESSION_KEY_MARKER = "dashboard_session_master_key_hash"


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


async def record_session_key_marker(db: AsyncSession, key_hash: str) -> None:
    """Stage the marker naming the master key sessions are minted under.

    Update-then-insert so it works whether or not the row exists yet; the
    caller commits.
    """
    result = cast(
        CursorResult[Any],
        await db.execute(update(RuntimeSetting).where(RuntimeSetting.key == SESSION_KEY_MARKER).values(value=key_hash)),
    )
    if result.rowcount == 0:
        db.add(RuntimeSetting(key=SESSION_KEY_MARKER, value=key_hash))


async def revoke_sessions_on_master_key_change(config: GatewayConfig, db: AsyncSession) -> None:
    """At startup, revoke every dashboard session if the master key changed.

    A session only proves possession of the master key at mint time, so it must
    not outlive the key. The generated key rotates through the dashboard, which
    revokes sessions inline; a configured key rotates by changing
    ``OTARI_MASTER_KEY``/config and restarting, which no request handler
    observes. Comparing a stored hash of the effective key here closes that
    path (and any configured/generated regime switch).

    Best-effort like ``ensure_master_key``: a failure only skips this check for
    the boot, and concurrent workers racing the first marker INSERT are benign.
    """
    current = hash_master_key(config.master_key) if config.master_key is not None else config._master_key_hash
    if current is None:
        return
    try:
        row = await db.get(RuntimeSetting, SESSION_KEY_MARKER)
        if row is not None and row.value == current:
            return
        if row is not None:
            await revoke_all_dashboard_sessions(db)
        await record_session_key_marker(db, current)
        await db.commit()
    except SQLAlchemyError:
        await db.rollback()
        logger.warning("Could not check dashboard sessions against the current master key; skipping for this boot.")


def request_is_https(request: Request) -> bool:
    """Whether the browser leg of this request is HTTPS, for the Secure flag.

    Behind a TLS-terminating proxy, uvicorn only honors ``X-Forwarded-Proto``
    from trusted IPs (loopback by default), so ``request.url.scheme`` reads
    "http" on typical PaaS ingress despite an HTTPS browser leg. Honor the
    header here regardless of source: it only decides the cookie's ``Secure``
    attribute, and a spoofed "https" over plain HTTP merely denies the spoofer
    their own session cookie.
    """
    if request.url.scheme == "https":
        return True
    forwarded = request.headers.get("X-Forwarded-Proto", "")
    return forwarded.split(",")[0].strip().lower() == "https"


def apply_session_cookie(response: Response, token: str, expires_at: datetime, *, secure: bool) -> None:
    """Set the session cookie with its security attributes in one place.

    ``secure`` mirrors the effective request scheme (``request_is_https``)
    rather than being hard-coded: a plain HTTP deployment (LAN, localhost
    without TLS) would otherwise never receive the cookie back. That is no
    worse than such a deployment already sending the raw master key in
    cleartext today. ``SameSite=Strict`` keeps cross-site requests from
    carrying the cookie, which is the primary CSRF control here (the dashboard
    and API are same-origin). ``Path=/`` is as narrow as the surface allows:
    the management routes live directly under ``/v1`` beside inference, so the
    cookie reaches inference paths too; that grants nothing beyond what master
    authority already has, and cross-site use is blocked by SameSite.
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
