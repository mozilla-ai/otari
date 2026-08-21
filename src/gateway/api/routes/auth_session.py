"""Dashboard sign-in sessions (standalone mode only).

``POST /v1/auth/session`` exchanges a credential for a server-issued session
held in an HttpOnly cookie, so the dashboard never persists that credential in
the browser and a sign-in survives tab closes and restarts. ``DELETE`` is
sign-out. The cookie is honored by the master-key auth dependencies in
``gateway.api.deps`` when a request carries no header credentials, and it names
the identity it was minted for, so those dependencies resolve a caller from it.

**Two credentials, one at a time, and which one depends on the deployment.**
mozilla-ai/otari-ai#1716 settled that the master key bootstraps a standalone
deployment and then retires as its dashboard login, staying the deployment-wide
API credential. So:

- Until any identity has a password, the master key signs in here, provisioning
  the default organization and its workspace and binding the session to the
  bootstrap operator. This is first boot, and it is unchanged.
- Once an identity has a password (an operator claimed the deployment through
  ``PUT /v1/auth/password``), the master key is refused *for sign-in*, and email
  and password is the login. It still authenticates ``/v1/keys``, ``/v1/users``
  and the rest of the management surface through the header, which is what every
  self-hoster's automation and the OSS smoke gate use.

``GET /v1/bootstrap`` publishes which of the two a deployment is currently
accepting, so the login page asks for the credential that will work rather than
discovering it from a 403.

#651 and #652 add further credentials (OAuth, WebAuthn). Both are redirect or
ceremony flows with more than one round trip, so they get their own endpoints
rather than another field here; what they share with this one is the session
those flows end by minting, not the request that starts them.
"""

import uuid
from datetime import datetime
from typing import Annotated, Self

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, Field, model_validator
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import get_config, get_db, is_valid_master_key
from gateway.core.config import GatewayConfig
from gateway.log_config import logger
from gateway.metrics import record_auth_failure
from gateway.models.tenancy import User as TenancyUser
from gateway.rate_limit import RateLimiter
from gateway.services.dashboard_session_service import (
    SESSION_COOKIE_NAME,
    apply_session_cookie,
    clear_session_cookie,
    create_dashboard_session,
    request_is_https,
    revoke_dashboard_session,
)
from gateway.services.password_service import MAX_PASSWORD_BYTES
from gateway.services.tenancy.email_address import MAX_EMAIL_LENGTH
from gateway.services.tenancy.errors import EmailNotVerifiedError, InvalidCredentialsError
from gateway.services.tenancy.provisioning_service import ensure_bootstrap_identity
from gateway.services.tenancy.user_service import authenticate, has_password_identity

router = APIRouter(prefix="/v1/auth/session", tags=["auth"])

MASTER_KEY_SIGN_IN_RETIRED = (
    "Master-key sign-in is retired on this deployment: an identity here has a password. "
    "Sign in with your email and password. The master key still authenticates the management API."
)


class CreateSessionRequest(BaseModel):
    """Sign in to the dashboard with exactly one credential.

    A flat body with an optional field per credential, rather than a tagged
    union: it is one extra key on the wire, it generates a client type a
    hand-written form can fill in, and the validator below makes the two forms
    exclusive anyway.

    The example carries one credential, because a generated example is a body
    somebody will post: the schema alone would produce every field at once,
    which is the one shape the validator below refuses.
    """

    model_config = {
        "json_schema_extra": {
            "example": {"email": "operator@example.com", "password": "a-real-password"}
        }
    }

    # Every field is bounded, because this endpoint is unauthenticated and the
    # validator below settles which credential arrived rather than how large it
    # is. Each bound is the widest value that could ever succeed, so nothing
    # legitimate is refused, and an oversized credential is answered before it
    # costs a bcrypt verification or a lookup.
    #
    # It does not bound what the process allocates: ASGI has already read the
    # whole body by the time a field is validated, so an 8MB sign-in body is
    # still buffered in full and only the answer changes. Capping the request
    # itself belongs to the proxy in front of the gateway.
    master_key: str | None = Field(
        default=None,
        # A generated key is ~52 characters; an operator-set one is arbitrary, so
        # this is a sanity ceiling rather than a format.
        max_length=512,
        description=(
            "The gateway master key; verified once and never stored by the browser. Accepted only "
            "while no identity on this deployment has a password (see GET /v1/bootstrap)."
        ),
    )
    email: str | None = Field(
        default=None,
        max_length=MAX_EMAIL_LENGTH,
        description="The identity's sign-in address.",
    )
    password: str | None = Field(
        default=None,
        # A stored password is at most MAX_PASSWORD_BYTES *bytes*, so it can
        # never be more than that many characters. Anything longer cannot match
        # any hash this gateway wrote, which makes refusing it early free.
        max_length=MAX_PASSWORD_BYTES,
        description="The identity's password.",
    )

    @model_validator(mode="after")
    def _exactly_one_credential(self) -> Self:
        """Refuse a body that presents neither credential, or both.

        Both is refused rather than resolved by precedence: a caller that sent
        two credentials does not know which one it is signing in with, and
        silently picking one would decide that for them.
        """
        by_master_key = self.master_key is not None
        by_password = self.email is not None or self.password is not None
        if by_master_key and by_password:
            msg = "Send either master_key or email and password, not both"
            raise ValueError(msg)
        if by_password and not (self.email and self.password):
            msg = "Both email and password are required to sign in with a password"
            raise ValueError(msg)
        if not by_master_key and not by_password:
            msg = "Send either master_key or email and password"
            raise ValueError(msg)
        return self


class SessionResponse(BaseModel):
    """A freshly minted dashboard session (the token travels only in the cookie)."""

    expires_at: datetime = Field(description="When the session cookie stops being accepted.")
    user_id: uuid.UUID = Field(description="The identity this session speaks for.")
    active_organization_id: uuid.UUID = Field(
        description="The organization that identity is acting in, which scopes every tenancy surface."
    )


def _check_login_rate_limit(request: Request) -> None:
    """Throttle repeated failed sign-in attempts per client IP.

    Only called on a *failed* verification, so a correct credential is never
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


async def _sign_in_with_master_key(
    master_key: str, request: Request, db: AsyncSession, config: GatewayConfig
) -> TenancyUser:
    """Bootstrap sign-in: verify the master key and resolve the operator identity.

    Refused once any identity holds a password, which is what retires this as a
    login. The refusal is a 403 and not a 401, and it comes after verification:
    the key is a valid credential, it is this *use* of it that is over, and
    saying so is what lets a stale client show the right message instead of
    prompting for the key again. It leaks nothing that ``GET /v1/bootstrap``
    does not already publish unauthenticated, by design, so that the login page
    can render the right form.
    """
    if not await is_valid_master_key(master_key, config, db):
        record_auth_failure("invalid_key")
        _check_login_rate_limit(request)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid master key")
    if await has_password_identity(db):
        record_auth_failure("master_key_sign_in_retired")
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=MASTER_KEY_SIGN_IN_RETIRED)
    # Provisions the tenancy root on a first-ever sign-in, and resolves the same
    # operator every time after that. It commits its own work, which is why it
    # runs before the session row is staged rather than beside it.
    return await ensure_bootstrap_identity(db)


async def _sign_in_with_password(email: str, password: str, request: Request, db: AsyncSession) -> TenancyUser:
    """Steady-state sign-in: verify an identity's own email and password.

    Both failures ``authenticate`` raises are converted here rather than left
    to the tenancy error handler, because a failed sign-in has to be counted
    and throttled, and the handler knows about neither. ``EmailNotVerifiedError``
    is still a failed sign-in attempt by every measure that matters here: it
    costs a bcrypt verification the same as a wrong password, and a caller who
    already knows a valid (email, password) pair for an unverified account
    would otherwise be able to hammer this endpoint with it, uncounted and
    unthrottled, unlike every other way this call can fail. Its own status and
    message survive the conversion, unlike ``InvalidCredentialsError``'s.
    """
    try:
        return await authenticate(db, email=email, password=password)
    except InvalidCredentialsError as exc:
        record_auth_failure("invalid_password")
        _check_login_rate_limit(request)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=exc.message) from None
    except EmailNotVerifiedError as exc:
        record_auth_failure("email_not_verified")
        _check_login_rate_limit(request)
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=exc.message) from None


@router.post("")
async def create_session(
    body: CreateSessionRequest,
    request: Request,
    response: Response,
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> SessionResponse:
    """Verify a sign-in credential and set the HttpOnly session cookie.

    The session is bound to the identity that authenticated, so every request it
    later authenticates resolves a user and that user's active organization
    rather than only "a credential was presented once". The response names both,
    so a client knows who it is signed in as without a second call.

    The rate-limit check deliberately runs only after a failed verification,
    not before it: a pre-verification gate can't know whether *this* attempt
    would have succeeded, so once an IP has used up its failure quota it
    would end up blocking that IP's legitimate owner too, not just further
    attackers. Running after verification also means the throttle bounds how
    many verdicts an IP gets, not how much work it can cause: a password attempt
    pays for a bcrypt verification (cost 12, on the order of 200ms of CPU, and
    one is burned against a stand-in hash even for an address nobody holds)
    before the limit is consulted, so a 429 costs the same as a 401. A gateway
    exposed to the internet should rate-limit this path at the proxy as well.
    """
    if body.master_key is not None:
        identity = await _sign_in_with_master_key(body.master_key, request, db, config)
    else:
        assert body.email is not None and body.password is not None  # guaranteed by the model validator
        identity = await _sign_in_with_password(body.email, body.password, request, db)
    try:
        token, expires_at = await create_dashboard_session(
            db, config.dashboard_session_ttl_hours, user_id=identity.id
        )
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
    return SessionResponse(
        expires_at=expires_at,
        user_id=identity.id,
        active_organization_id=identity.active_organization_id,
    )


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
