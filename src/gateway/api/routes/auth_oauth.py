"""Google and GitHub sign-in for the dashboard (standalone mode only).

Two calls per provider, and the split from ``auth_session.py`` is the one that
file's docstring already anticipated: an OAuth sign-in is a redirect through a
third party, so the credential cannot be another field on the sign-in body.
What it shares with that endpoint is the end and not the beginning. A completed
exchange mints the same HttpOnly session cookie a password does, through the
same ``gateway.services.dashboard_session_service``, so everything downstream of
a sign-in is unchanged.

**The whole surface is public**, because it is how somebody who is not signed in
signs in, and both routes are throttled per client IP through
``throttle_public_auth`` like the signup and reset routes.

**Where the browser's part begins and ends.** ``/authorize`` mints a CSRF
``state`` and hands back the consent-screen URL; the dashboard stores that state
in ``sessionStorage`` and sends the person to the provider. The provider returns
them to ``/auth/{provider}/callback``, an ordinary path (a redirect URI may not
carry a fragment, so it cannot be the hash route directly) which
``gateway.main`` redirects into the dashboard's own callback page. That page
compares the returned state against the stored one and, only then, posts the
code and the state here, where the state is checked again against the row
``/authorize`` wrote. Two checks that fail in different directions: the
browser's binds a callback to the tab that started the flow, and this one binds
it to a flow this deployment started.

**And to the browser that started it.** ``/authorize`` also sets an HttpOnly
flow cookie whose digest the row keeps, and the callback refuses without it.
The code and the state share one redirect URL that the access log and browser
history both record; the cookie is the half of the flow that neither does. See
``gateway.services.oauth_service``.

**What this route decides, and what it does not.** It proves the person holds
the provider account. Who that makes them *here* is behind
``IdentityProviderPort``: this build resolves the identity against its roster
and refuses one it does not recognize, and an overlay binds a different policy
without editing this file.
"""

import uuid
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Cookie, Depends, HTTPException, Path, Request, Response, status
from pydantic import BaseModel, Field
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import IdentityProviderPortDep, get_config, get_db
from gateway.api.routes._public_auth import throttle_public_auth

# The same refusal the password and passkey sign-ins carry, imported rather than
# restated: the freeze is one deployment-wide state, and three sign-in routes
# wording it differently would tell a person the doors closed for three reasons.
from gateway.api.routes.auth_session import MAINTENANCE_MODE_REFUSAL
from gateway.core.config import OAUTH_PROVIDERS, GatewayConfig
from gateway.log_config import logger
from gateway.metrics import record_auth_failure
from gateway.services.dashboard_session_service import (
    apply_session_cookie,
    create_dashboard_session,
    request_is_https,
)
from gateway.services.maintenance_mode_service import is_maintenance_mode
from gateway.services.oauth_service import (
    FLOW_COOKIE_NAME,
    OAUTH_ROUTE_PREFIX,
    apply_flow_cookie,
    authorization_url,
    exchange_code,
    flow_secret_for,
    provider_label,
    require_configured,
)
from gateway.services.tenancy.errors import OAuthNotConfiguredError, TenancyError
from gateway.services.tenancy.organization_domain_service import OrganizationDomainService

router = APIRouter(prefix=OAUTH_ROUTE_PREFIX, tags=["auth"])

# A code is a provider-issued opaque string, a few hundred characters at most;
# this is a sanity ceiling on an unauthenticated request body rather than a
# format, matching the bounds ``auth_session.CreateSessionRequest`` sets.
_MAX_SUBMITTED_CODE = 2048
# A state this deployment issued is 43 characters (``secrets.token_urlsafe(32)``).
# The ceiling is loose rather than exact because the value is looked up by hash
# and a wrong length is simply a state that matches nothing; what it bounds is
# how much an unauthenticated caller can make this process hash.
_MAX_SUBMITTED_STATE = 512

# Only a provider this deployment could ever configure is a path this router
# answers at all, so an unknown segment is the framework's own 422 rather than a
# handler deciding what to do with it. Spelled from the config vocabulary so the
# two cannot drift.
# The browser's flow cookie, when it sent one. Bounded for the same reason the
# state is: it is hashed before anything looks at it.
FlowCookie = Annotated[
    str | None, Cookie(alias=FLOW_COOKIE_NAME, max_length=_MAX_SUBMITTED_STATE, include_in_schema=False)
]

ProviderPath = Annotated[
    str,
    Path(
        description="Which OAuth provider to sign in with.",
        # ``pattern`` rather than an enum type, because the value is an open
        # string everywhere else it travels (see ``core.config.OAUTH_PROVIDERS``)
        # and a closed enum here would be the one place that could not carry a
        # connection an overlay contributes.
        pattern=f"^({'|'.join(OAUTH_PROVIDERS)})$",
    ),
]


class AuthorizeResponse(BaseModel):
    """Where to send the browser, and the state to check when it comes back."""

    authorization_url: str = Field(description="The provider consent screen to navigate to.")
    state: str = Field(
        description=(
            "An opaque CSRF value to keep for the length of the redirect, compare against the "
            "'state' the provider returns, and send back with the authorization code. A callback "
            "whose state does not match the one held by the browser that started the flow should "
            "be abandoned by the client rather than sent here; one that does is checked again "
            "against this deployment's own record of it. The response also sets an HttpOnly "
            "cookie that the callback requires, so the exchange can only be completed from the "
            "browser this call was made from."
        )
    )


class OAuthCallbackRequest(BaseModel):
    """The authorization code a provider handed the browser.

    No ``redirect_uri``: this deployment derives its own from ``public_base_url``
    so the URI used to build the authorization request and the one sent with the
    exchange are the same string by construction, and a browser cannot choose
    what this server sends to a provider.

    ``state`` is required, and is what binds this callback to an authorization
    request this deployment actually made: it is claimed from
    ``oauth_pending_state`` before the code is sent anywhere, and the row it
    claims is what carries the PKCE verifier the exchange needs. The flow
    cookie ``/authorize`` set travels alongside and binds it to the browser.
    """

    code: str = Field(
        max_length=_MAX_SUBMITTED_CODE,
        description="The authorization code from the provider's redirect.",
    )
    state: str = Field(
        max_length=_MAX_SUBMITTED_STATE,
        description="The 'state' from the provider's redirect, as issued by /authorize.",
    )


class OAuthSessionResponse(BaseModel):
    """A dashboard session minted by an OAuth sign-in (the token travels only in the cookie).

    The same three fields ``POST /api/v1/auth/session`` answers, deliberately: the
    dashboard's sign-in path does not care which credential got it here.
    """

    expires_at: datetime = Field(description="When the session cookie stops being accepted.")
    user_id: uuid.UUID = Field(description="The identity this session speaks for.")
    active_organization_id: uuid.UUID = Field(
        description="The organization that identity is acting in, which scopes every tenancy surface."
    )


def require_oauth_provider(
    provider: ProviderPath,
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> None:
    """Refuse a provider this deployment did not configure, and say which settings.

    A dependency rather than a check inside each handler, and ahead of both, for
    two reasons. It answers before the throttle and before the maintenance-mode
    read, so a request naming a provider that could never work costs nothing and
    never reaches an authorization code; and it makes "is this provider on
    offer" one decision rather than a property of whichever call happened to
    look first, which is what let a stubbed exchange hide it.

    Rendered here rather than left to the tenancy error handler, for the reason
    ``auth_webauthn.require_passkey_support`` gives: that handler blanks the
    message of every error carrying a status of 500 or above, which is right for
    an operator problem the caller cannot act on and wrong for this one, where
    the missing settings are exactly what the operator needs to read.
    """
    try:
        require_configured(config, provider)
    except OAuthNotConfiguredError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from None


@router.get(
    "/{provider}/authorize",
    response_model=AuthorizeResponse,
    dependencies=[Depends(require_oauth_provider)],
)
async def authorize(
    provider: ProviderPath,
    request: Request,
    response: Response,
    config: Annotated[GatewayConfig, Depends(get_config)],
    db: Annotated[AsyncSession, Depends(get_db)],
    flow_cookie: FlowCookie = None,
) -> AuthorizeResponse:
    """Start an OAuth sign-in: where to send the browser, and the state to keep.

    A GET that writes, which is the one thing to know about it. It records the
    authorization it is about to start (the state's hash, the PKCE verifier the
    exchange will need, and the digest of a flow secret it sets as an HttpOnly
    cookie) so the callback has something to check against, and that record is
    the whole reason the callback can refuse a code this deployment never asked
    for, or one presented from a browser other than the one that asked.

    Still safe to repeat: each call mints its own state, and only the one the
    browser kept is the one it sends back. The rows the others leave expire on
    their own and are swept by the next call. The cookie is reused when the
    browser already holds one, so a second tab does not break the first.
    """
    throttle_public_auth(request)
    flow_secret = flow_secret_for(flow_cookie)
    try:
        url, state = await authorization_url(config, provider, db=db, flow_secret=flow_secret)
        await db.commit()
    except SQLAlchemyError:
        await db.rollback()
        logger.warning("Failed to record a pending %s sign-in", provider, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error",
        ) from None
    apply_flow_cookie(response, flow_secret, secure=request_is_https(request))
    return AuthorizeResponse(authorization_url=url, state=state)


@router.post(
    "/{provider}/callback",
    response_model=OAuthSessionResponse,
    dependencies=[Depends(require_oauth_provider)],
)
async def callback(
    provider: ProviderPath,
    body: OAuthCallbackRequest,
    request: Request,
    response: Response,
    identity_provider: IdentityProviderPortDep,
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    flow_cookie: FlowCookie = None,
) -> OAuthSessionResponse:
    """Exchange an authorization code and set the HttpOnly session cookie.

    The session is bound to the identity the provider's account resolves to,
    exactly as a password sign-in binds one to the identity that authenticated,
    so every request it later authenticates resolves the same caller.

    A refusal is counted like the other sign-in failures
    (``record_auth_failure``) and rendered by the tenancy error handler. Like
    the passkey route there is no separate post-failure throttle: this route is
    throttled unconditionally on the way in, because there is no legitimate
    caller here whose correct credential must never be blocked. An authorization
    code is single-use and minted by a redirect, not something a person retries
    by hand.

    **Maintenance mode freezes this the way it freezes the other two sign-ins.**
    The freeze is on starting a session, not on a credential, so an OAuth sign-in
    has to answer to it or the switch is bypassable by anybody holding a Google
    account. Refused before the exchange, so a frozen deployment makes no
    outbound call, spends nobody's authorization code, and counts no auth
    failure: nobody failed to authenticate, the gateway declined to try.
    """
    throttle_public_auth(request)
    if await is_maintenance_mode(db):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=MAINTENANCE_MODE_REFUSAL,
        )
    try:
        external = await exchange_code(
            config, provider, code=body.code, state=body.state, flow_secret=flow_cookie, db=db
        )
        identity = await identity_provider.resolve(
            provider=external.provider,
            email=external.email,
            full_name=external.full_name,
            email_verified=external.email_verified,
        )
    except TenancyError:
        record_auth_failure("invalid_oauth")
        raise

    try:
        # Staging the session row is inside this block, not just the commit:
        # ``create_dashboard_session`` prunes expired rows with a DELETE before
        # it stages anything, so it is a statement that can fail on its own. Left
        # outside, that failure would skip the rollback and the log line below
        # and surface as a bare 500 from the generic handler.
        #
        # One commit for the whole sign-in: the adapter's link and verification
        # stamp are on this same session, so they land with the session row or
        # with neither.
        # Before the session row and inside the same transaction, so a new
        # membership and the sign-in that earned it land together or not at all.
        # Not guarded: the only expected failure is two concurrent sign-ins
        # racing, which the service settles on its own, and a database that
        # cannot stage this cannot stage the session row either.
        await OrganizationDomainService(db).auto_join_for_user(identity)
        token, expires_at = await create_dashboard_session(db, config.dashboard_session_ttl_hours, user_id=identity.id)
        await db.commit()
    except SQLAlchemyError:
        await db.rollback()
        logger.warning("Failed to persist a dashboard session on a %s sign-in", provider, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error",
        ) from None
    logger.info("Signed in %s with %s", identity.id, provider_label(provider))
    apply_session_cookie(response, token, expires_at, secure=request_is_https(request))
    return OAuthSessionResponse(
        expires_at=expires_at,
        user_id=identity.id,
        active_organization_id=identity.active_organization_id,
    )


__all__ = ["router"]
