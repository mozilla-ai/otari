"""Dashboard sign-in sessions (standalone mode only).

``POST /v1/auth/session`` exchanges the master key for a server-issued session
held in an HttpOnly cookie, so the dashboard never persists the raw key in the
browser and a sign-in survives tab closes and restarts. ``DELETE`` is sign-out.
The cookie is honored by the master-key auth dependencies in
``gateway.api.deps`` when a request carries no header credentials.
"""

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, Field
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import get_config, get_db, is_valid_master_key
from gateway.core.config import GatewayConfig
from gateway.log_config import logger
from gateway.metrics import record_auth_failure
from gateway.rate_limit import RateLimiter
from gateway.services.dashboard_session_service import (
    SESSION_COOKIE_NAME,
    apply_session_cookie,
    clear_session_cookie,
    create_dashboard_session,
    request_is_https,
    revoke_dashboard_session,
)

router = APIRouter(prefix="/v1/auth/session", tags=["auth"])


class CreateSessionRequest(BaseModel):
    """Sign in to the dashboard by proving possession of the master key."""

    master_key: str = Field(description="The gateway master key; verified once and never stored by the browser.")


class SessionResponse(BaseModel):
    """A freshly minted dashboard session (the token travels only in the cookie)."""

    expires_at: datetime = Field(description="When the session cookie stops being accepted.")


def _check_login_rate_limit(request: Request) -> None:
    """Throttle repeated failed sign-in attempts per client IP.

    Only called on a *failed* verification, so a correct master key is never
    throttled, even from an IP that has already used up its failure quota:
    the issue this implements explicitly requires that a legitimate operator
    is never locked out. That requirement is why this can't run *before*
    verification (see the note on create_session for why that was tried and
    reverted). Separate limiter from the general per-user ``rate_limit_rpm``
    (that one is keyed to authenticated users and never sees this pre-auth
    path). Raises 429 with Retry-After via RateLimiter.check when the calling
    IP is already over the configured limit.

    Client IP comes from ``request.client.host``, not a hand-parsed
    X-Forwarded-For: uvicorn's ProxyHeadersMiddleware already rewrites it from
    that header, but only when the immediate peer is in ``forwarded_allow_ips``
    (loopback by default, the same trust boundary ``request_is_https`` relies
    on for X-Forwarded-Proto). Parsing the header directly here, instead of
    through that trust boundary, would let anyone bypass the throttle by
    sending their own X-Forwarded-For.
    """
    login_rate_limiter: RateLimiter | None = getattr(request.app.state, "login_rate_limiter", None)
    if login_rate_limiter is None:
        return
    client_ip = request.client.host if request.client else None
    if client_ip is None:
        return
    try:
        login_rate_limiter.check(client_ip)
    except HTTPException:
        logger.warning("Dashboard sign-in rate limit exceeded for %s", client_ip)
        raise


@router.post("")
async def create_session(
    body: CreateSessionRequest,
    request: Request,
    response: Response,
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> SessionResponse:
    """Verify the master key and set the HttpOnly session cookie.

    The rate-limit check deliberately runs only after a failed verification,
    not before it: a pre-verification gate can't know whether *this* attempt
    would have succeeded, so once an IP has used up its failure quota it
    would end up blocking that IP's legitimate owner too, not just further
    attackers. The issue this implements explicitly rules that out. The
    DB/hash lookup this exposes to repeated attempts only runs when no fixed
    master_key is configured (the auto-generated bootstrap-key path); with a
    configured master_key, verification is a constant-time string compare,
    not a DB round trip.
    """
    if not await is_valid_master_key(body.master_key, config, db):
        record_auth_failure("invalid_key")
        _check_login_rate_limit(request)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid master key")
    try:
        token, expires_at = await create_dashboard_session(db, config.dashboard_session_ttl_hours)
        await db.commit()
    except SQLAlchemyError:
        await db.rollback()
        # Generic error to the client; the raw failure is only logged here.
        logger.warning("Failed to persist a dashboard session on sign-in", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error",
        ) from None
    apply_session_cookie(response, token, expires_at, secure=request_is_https(request))
    return SessionResponse(expires_at=expires_at)


@router.delete("", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(
    request: Request,
    response: Response,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    """Sign out: revoke the cookie's session server-side and expire the cookie.

    Deliberately unauthenticated and idempotent: it only ever revokes the
    session named by the caller's own cookie, and the dashboard calls it on the
    401-bounce path where no valid credential exists anymore. Unlike the read
    path in ``deps.py`` it applies no Sec-Fetch-Site check: ``SameSite=Strict``
    already keeps cross-site requests from carrying the cookie, and the worst a
    forged call could do is sign the operator out.
    """
    token = request.cookies.get(SESSION_COOKIE_NAME)
    if token:
        try:
            await revoke_dashboard_session(db, token)
            await db.commit()
        except SQLAlchemyError:
            await db.rollback()
            # Raising here would skip the cookie clear below (FastAPI discards
            # the injected response on an exception), leaving the browser with
            # a live cookie the operator believes is gone. Clear it and return
            # 204 anyway; the unrevoked row dies on its TTL.
            logger.warning("Failed to revoke the dashboard session on sign-out", exc_info=True)
    clear_session_cookie(response)
