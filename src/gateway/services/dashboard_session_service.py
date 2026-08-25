"""Dashboard sign-in sessions.

Signing in to the admin dashboard exchanges the master key for a server-issued
session: an opaque token handed to the browser in an HttpOnly cookie, with only
its SHA-256 hash stored in the ``dashboard_sessions`` table. This lets a
sign-in survive tab closes and browser restarts without ever persisting the
master key (or any JS-readable credential) in the browser.

Every session names the identity it was minted for, so it resolves a caller and
that caller's active organization rather than only proving the master key was
presented once. Master-key sign-in binds the session to the deployment's
bootstrap operator (`services.tenancy.provisioning_service`); a per-user sign-in
flow binds it to whoever authenticated.

An opaque token, deliberately, rather than the platform's JWT ``Token``: this
session is revocable server-side, which a bearer JWT is not, and the cookie,
HTTPS and rate-limit machinery around it already works. otari-ai#1716 settled
that sessions are the steady-state dashboard login, so the platform's JWT does
not survive the rehome (see ``docs/access-control.md``).

Sessions live in the database, not process memory, so every worker and replica
accepts them and a revocation is seen everywhere. They expire on a TTL
(``dashboard_session_ttl_hours``) and are revoked on sign-out, on master-key
rotation, when their identity is deactivated, and (through the foreign key) when
it is deleted.
"""

import hashlib
import secrets
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any, cast

from fastapi import Request, Response
from sqlalchemy import delete, select, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col

from gateway.core.config import GatewayConfig
from gateway.core.database import create_session
from gateway.log_config import logger
from gateway.models.entities import DashboardSession, RuntimeSetting
from gateway.models.tenancy import User
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


async def create_dashboard_session(
    db: AsyncSession, ttl_hours: int, *, user_id: uuid.UUID
) -> tuple[str, datetime]:
    """Stage a new session row for ``user_id`` and return ``(token, expires_at)``.

    ``user_id`` names the tenancy identity the session speaks for, and is
    required rather than defaulted: a session that names nobody cannot resolve a
    caller, which is the reason the column exists.

    ``user.last_sign_in_at`` is stamped here rather than at the four sign-in
    routes that call this, so a fifth cannot forget to. The one call that is not
    a sign-in in the ordinary sense is the re-mint after a master-key rotation
    (``routes/settings.py``), and it is stamped too: the operator proved the new
    key to get the new session, which is the event the column records.

    Expired rows are pruned opportunistically here, so the table stays small
    without a background task. The caller owns the transaction and must commit
    before handing the token to the browser.
    """
    now = datetime.now(UTC)
    await db.execute(delete(DashboardSession).where(DashboardSession.expires_at < now))
    await db.execute(update(User).where(col(User.id) == user_id).values(last_sign_in_at=now))
    token = f"{_SESSION_TOKEN_PREFIX}{secrets.token_urlsafe(32)}"
    expires_at = now + timedelta(hours=ttl_hours)
    db.add(
        DashboardSession(
            token_hash=hash_session_token(token),
            user_id=user_id,
            created_at=now,
            expires_at=expires_at,
        )
    )
    return token, expires_at


async def resolve_dashboard_session(db: AsyncSession, token: str) -> User | None:
    """Return the identity a stored, unexpired session token speaks for.

    One query joining the session to its identity, rather than two lookups or a
    lazy ``relationship()`` (which raises ``MissingGreenlet`` on an
    ``AsyncSession``), because this runs on every cookie-authenticated request.

    Three things make a token resolve to nothing, and all three mean "not
    authenticated": no such session, an expired one, and one whose identity has
    been deactivated. That last is why the identity is loaded here rather than
    trusted from the session row: deactivating an operator has to end their
    dashboard access, not wait out the TTL.

    A deactivated identity's sessions are also deleted rather than only refused;
    see ``_revoke_deactivated_identity_sessions`` below for why.

    Expiry is compared in Python, as the rest of this module does: SQLite hands
    the stored timestamp back naive, so the check goes through ``_as_utc``.
    """
    row = (
        await db.execute(
            select(DashboardSession, User)
            .join(User, col(User.id) == DashboardSession.user_id)
            .where(DashboardSession.token_hash == hash_session_token(token))
        )
    ).first()
    if row is None:
        return None
    # Annotated rather than tuple-unpacked: a two-entity ``Row`` types as ``Any``
    # under mypy strict, and the return type is what callers rely on.
    session: DashboardSession = row[0]
    identity: User = row[1]
    if _as_utc(session.expires_at) < datetime.now(UTC):
        return None
    if not identity.is_active:
        await _revoke_deactivated_identity_sessions(identity.id)
        return None
    return identity


async def _revoke_deactivated_identity_sessions(user_id: uuid.UUID) -> None:
    """Delete a deactivated identity's sessions, on a session of its own.

    Refusing a session without deleting it leaves the row alive, so
    re-activating an identity hands back the access of every cookie it held,
    which is the opposite of what deactivating it for a lost laptop was for.
    ``DeploymentUserService.update_user`` is the flow that deactivates one, and
    it does call ``revoke_user_dashboard_sessions`` in its own transaction rather
    than leaving the rows to be found here: this path only runs when the browser
    next presents the cookie, and until then a session that should be gone is
    alive. This stays the backstop for a row deactivated some other way, an
    operator's own SQL among them.

    A short-lived session rather than the request's, following
    ``deps._bump_last_used_at``, which is the other best-effort write on the
    auth path: this one runs inside an auth dependency, so committing the
    caller's session would commit whatever else that request had staged. Today
    nothing is staged by the time this runs, but that is a property of every
    current call site rather than of this function, and a structure that does
    not depend on it cannot be broken by a later one.

    A failure is logged and swallowed. The caller's answer is "not
    authenticated" either way, and turning a 401 into a 503 because the cleanup
    could not be written would be the worse outcome; the rows die on their TTL.
    """
    try:
        async with create_session() as session:
            await revoke_user_dashboard_sessions(session, user_id)
            await session.commit()
    except SQLAlchemyError:
        logger.warning("Could not revoke the sessions of deactivated identity %s", user_id, exc_info=True)


async def revoke_dashboard_session(db: AsyncSession, token: str) -> None:
    """Stage removal of one session (sign-out). The caller commits."""
    await db.execute(delete(DashboardSession).where(DashboardSession.token_hash == hash_session_token(token)))


async def revoke_all_dashboard_sessions(db: AsyncSession) -> None:
    """Stage removal of every session (master-key rotation). The caller commits."""
    await db.execute(delete(DashboardSession))


async def revoke_user_dashboard_sessions(
    db: AsyncSession, user_id: uuid.UUID, *, keep_token_hash: str | None = None
) -> None:
    """Stage removal of one identity's sessions, optionally sparing one.

    A password change ends the sessions minted under the old password, the way
    master-key rotation ends the ones minted under the old key. It is scoped to
    the one identity rather than to the table, because another person's session
    was not minted under this password and signing them out would be collateral.

    ``keep_token_hash`` spares the caller's own session so changing a password
    does not sign the caller out of the page they changed it on. A caller with
    no session of their own (the master key in a header) passes nothing, and
    every session for that identity ends. The caller commits.
    """
    statement = delete(DashboardSession).where(DashboardSession.user_id == user_id)
    if keep_token_hash is not None:
        statement = statement.where(DashboardSession.token_hash != keep_token_hash)
    await db.execute(statement)


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
