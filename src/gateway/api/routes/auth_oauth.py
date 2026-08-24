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
authorization code here. So the state round-trips through the browser that
minted it and this deployment stores nothing between the two requests, which is
the same reason PKCE stays off; see ``gateway.services.oauth_service``.

**What this route decides, and what it does not.** It proves the person holds
the provider account. Who that makes them *here* is behind
``IdentityProviderPort``: this build resolves the identity against its roster
and refuses one it does not recognize, and an overlay binds a different policy
without editing this file.
"""

import uuid
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, Request, Response, status
from pydantic import BaseModel, Field
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import IdentityProviderPortDep, get_config, get_db_if_needed
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
    authorization_url,
    exchange_code,
    new_state,
    provider_label,
    require_configured,
)
from gateway.services.tenancy.errors import OAuthNotConfiguredError, TenancyError

router = APIRouter(prefix="/v1/auth/oauth", tags=["auth"])

# A code is a provider-issued opaque string, a few hundred characters at most;
# this is a sanity ceiling on an unauthenticated request body rather than a
# format, matching the bounds ``auth_session.CreateSessionRequest`` sets.
_MAX_SUBMITTED_CODE = 2048

# Only a provider this deployment could ever configure is a path this router
# answers at all, so an unknown segment is the framework's own 422 rather than a
# handler deciding what to do with it. Spelled from the config vocabulary so the
# two cannot drift.
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
            "An opaque CSRF value to keep for the length of the redirect and compare against the "
            "'state' the provider returns. It is not stored on this deployment, so a callback "
            "whose state does not match the one held by the browser that started the flow must be "
            "abandoned by the client rather than sent here."
        )
    )


class OAuthCallbackRequest(BaseModel):
    """The authorization code a provider handed the browser.

    No ``redirect_uri``: this deployment derives its own from ``public_base_url``
    so the URI used to build the authorization request and the one sent with the
    exchange are the same string by construction, and a browser cannot choose
    what this server sends to a provider.

    No ``state`` either, and that is not an omission. The state is checked in the
    browser, against the value that browser stored when it started the flow;
    sending it here would let this deployment compare a value to itself, which
    proves nothing without somewhere to have kept the original.
    """

    code: str = Field(
        max_length=_MAX_SUBMITTED_CODE,
        description="The authorization code from the provider's redirect.",
    )


class OAuthSessionResponse(BaseModel):
    """A dashboard session minted by an OAuth sign-in (the token travels only in the cookie).

    The same three fields ``POST /v1/auth/session`` answers, deliberately: the
    dashboard's sign-in path does not care which credential got it here.
    """

    expires_at: datetime = Field(description="When the session cookie stops being accepted.")
    user_id: uuid.UUID = Field(description="The identity this session speaks for.")
    active_organization_id: uuid.UUID = Field(
        description="The organization that identity is acting in, which scopes every tenancy surface."
    )


def _session(db: AsyncSession | None) -> AsyncSession:
    """Narrow the port-shaped optional session this router never actually sees.

    ``get_db_if_needed`` yields ``None`` only in hybrid mode, and this router is
    standalone-only (``api.main.register_routers``). It is taken in that shape
    anyway so the handler and ``IdentityProviderPort`` share one session and one
    transaction; see ``deps.get_identity_provider_port``.
    """
    assert db is not None, "the OAuth sign-in routes are mounted in standalone mode only"
    return db


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
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> AuthorizeResponse:
    """Start an OAuth sign-in: where to send the browser, and the state to keep.

    A GET, and safe: it reads configuration and mints a random value, writing
    nothing. Repeating it simply produces another state, and only the one the
    browser kept is the one it will compare against.
    """
    throttle_public_auth(request)
    state = new_state()
    return AuthorizeResponse(
        authorization_url=authorization_url(config, provider, state=state), state=state
    )


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
    db: Annotated[AsyncSession | None, Depends(get_db_if_needed)],
    config: Annotated[GatewayConfig, Depends(get_config)],
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
    session = _session(db)
    if await is_maintenance_mode(session):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=MAINTENANCE_MODE_REFUSAL,
        )
    try:
        external = await exchange_code(config, provider, code=body.code)
        identity = await identity_provider.resolve(
            provider=external.provider,
            email=external.email,
            full_name=external.full_name,
            email_verified=external.email_verified,
        )
    except TenancyError:
        record_auth_failure("invalid_oauth")
        raise

    token, expires_at = await create_dashboard_session(
        session, config.dashboard_session_ttl_hours, user_id=identity.id
    )
    try:
        # One commit for the whole sign-in: the adapter's link and verification
        # stamp are on this same session, so they land with the session row or
        # with neither.
        await session.commit()
    except SQLAlchemyError:
        await session.rollback()
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
