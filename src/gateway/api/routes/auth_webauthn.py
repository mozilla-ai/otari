"""Passkey ceremonies and the credentials a person manages (standalone only).

Two ceremonies, each two calls, plus a small CRUD surface over what they
produce. The split from ``auth_session.py`` is the one that file's docstring
already anticipated: a passkey needs a server-issued challenge before the
browser can answer, so it cannot be another field on the sign-in body. What it
shares with that endpoint is the end, not the beginning: a passkey sign-in mints
the same HttpOnly session cookie a password does, through the same
`gateway.services.dashboard_session_service`, so everything downstream of a
sign-in is unchanged.

**Which half is public.** ``/authenticate/options`` and ``/authenticate`` are
unauthenticated by necessity: they are how somebody who is not signed in signs
in. Both are throttled per client IP through ``throttle_public_auth``, like the
signup and reset routes. Everything else here (registering a passkey, listing,
renaming and deleting them) is behind the ordinary management credential, so a
passkey is always added by somebody who is already inside.

**Why registration is not a signup.** There is no way to arrive here without an
identity: registration attaches a credential to the caller's own session
identity. A deployment is still claimed with the master key and a password, and
a passkey joins an identity rather than creating one. See
`gateway.services.tenancy.webauthn_service`.

The options a ceremony issues, and the answer a browser sends back, are carried
as free-form objects rather than modeled field by field; see ``CeremonyOptions``
for why.
"""

import uuid
from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, Field
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import CurrentIdentity, get_config, get_db
from gateway.api.routes._public_auth import throttle_public_auth
from gateway.core.config import GatewayConfig
from gateway.log_config import logger
from gateway.metrics import record_auth_failure
from gateway.models.tenancy import (
    MAX_WEBAUTHN_CREDENTIAL_NAME,
    WebAuthnCredentialPublic,
    WebAuthnCredentialsPublic,
    WebAuthnCredentialUpdate,
)
from gateway.services.dashboard_session_service import (
    apply_session_cookie,
    create_dashboard_session,
    request_is_https,
)
from gateway.services.tenancy import webauthn_service
from gateway.services.tenancy.errors import PasskeysNotConfiguredError, TenancyError

router = APIRouter(prefix="/v1/auth/webauthn", tags=["auth"])


class CeremonyOptions(BaseModel):
    """The `PublicKeyCredentialCreationOptions`/`RequestOptions` a browser needs.

    Passed through as an opaque object rather than modeled field by field. The
    shape is the W3C's, the browser is the only consumer, and it is what
    ``navigator.credentials`` is handed verbatim after the two base64url fields
    are decoded. Restating it here would produce a second, slightly wrong copy
    of a spec this deployment does not own, and every field the library adds
    later would have to be added again to keep the client from dropping it.
    """

    model_config = {"extra": "allow"}


class RegisterPasskeyRequest(BaseModel):
    """A completed registration ceremony, with the label to file it under."""

    credential: dict[str, Any] = Field(description="The browser's PublicKeyCredential, serialized.")
    name: str | None = Field(
        default=None,
        max_length=MAX_WEBAUTHN_CREDENTIAL_NAME,
        description=(
            "What to call this passkey in the credential list. Optional: an unnamed one is "
            "numbered rather than refused, so a browser that offers no prompt still works."
        ),
    )


class AuthenticatePasskeyRequest(BaseModel):
    """A completed sign-in ceremony."""

    credential: dict[str, Any] = Field(description="The browser's PublicKeyCredential assertion, serialized.")


class PasskeySessionResponse(BaseModel):
    """A dashboard session minted by a passkey (the token travels only in the cookie).

    The same three fields ``POST /v1/auth/session`` answers, deliberately: the
    dashboard's sign-in path does not care which credential got it here.
    """

    expires_at: datetime = Field(description="When the session cookie stops being accepted.")
    user_id: uuid.UUID = Field(description="The identity this session speaks for.")
    active_organization_id: uuid.UUID = Field(
        description="The organization that identity is acting in, which scopes every tenancy surface."
    )


def require_passkey_support(config: Annotated[GatewayConfig, Depends(get_config)]) -> None:
    """Refuse up front on a deployment with no relying party, and say which setting.

    A dependency rather than a check inside each handler, and rendered here
    rather than left to the tenancy error handler, for the reason
    ``_public_auth.mail_unavailable`` gives: that handler blanks the message of
    every error carrying a status of 500 or above, which is right for the errors
    that describe a deployment failure the caller cannot act on and wrong for
    this one, where the missing setting is exactly what the operator needs to
    read.

    Not applied to renaming or deleting a passkey. Those need no ceremony, and a
    deployment whose relying-party ID was changed or lost is precisely when
    somebody needs to be able to clear out the rows it orphaned.
    """
    try:
        webauthn_service.require_relying_party(config)
    except PasskeysNotConfiguredError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from None


async def _commit(db: AsyncSession, what: str) -> None:
    """Commit, or turn a database failure into a 500 that leaks nothing.

    Shared by all five writers here so the rollback and the log line are not
    written five times; the message names the operation, because the exception
    itself never reaches the caller.
    """
    try:
        await db.commit()
    except SQLAlchemyError:
        await db.rollback()
        logger.warning("Failed to %s", what, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error",
        ) from None


@router.post("/register/options", response_model=CeremonyOptions, dependencies=[Depends(require_passkey_support)])
async def registration_options(
    identity: CurrentIdentity,
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> Any:
    """Start registering a passkey for the signed-in identity.

    A POST rather than a GET even though it reads like one: it issues a
    server-side challenge and writes it, so it is not safe to repeat, cache, or
    prefetch.
    """
    options = await webauthn_service.begin_registration(db, config, identity)
    await _commit(db, "issue a passkey registration challenge")
    return options


@router.post(
    "/register",
    response_model=WebAuthnCredentialPublic,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_passkey_support)],
)
async def register_passkey(
    body: RegisterPasskeyRequest,
    identity: CurrentIdentity,
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> Any:
    """Verify a registration ceremony and store the passkey it produced."""
    credential = await webauthn_service.finish_registration(db, config, identity, body.credential, body.name)
    await _commit(db, "store a registered passkey")
    return webauthn_service.to_public(credential)


@router.post(
    "/authenticate/options", response_model=CeremonyOptions, dependencies=[Depends(require_passkey_support)]
)
async def authentication_options(
    request: Request,
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> Any:
    """Start a passkey sign-in. Public, throttled, and names no credentials.

    The options carry no ``allowCredentials``, so this publishes nothing about
    who holds a passkey here; see ``webauthn_service.begin_authentication``.
    """
    throttle_public_auth(request)
    options = await webauthn_service.begin_authentication(db, config)
    await _commit(db, "issue a passkey sign-in challenge")
    return options


@router.post(
    "/authenticate", response_model=PasskeySessionResponse, dependencies=[Depends(require_passkey_support)]
)
async def authenticate_passkey(
    body: AuthenticatePasskeyRequest,
    request: Request,
    response: Response,
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> Any:
    """Verify an assertion and set the HttpOnly session cookie.

    The session is bound to the identity whose passkey signed, exactly as a
    password sign-in binds one to the identity that authenticated, so every
    request it later authenticates resolves the same caller.

    A refusal is counted like the other sign-in failures
    (``record_auth_failure``) and answered as a 401 by the tenancy error
    handler. Unlike the password path there is no separate post-failure
    throttle: this route is throttled unconditionally on the way in, because
    unlike a password there is no legitimate caller here whose correct
    credential must never be blocked (a passkey ceremony is one round trip a
    browser drives, not something a person retries by hand).
    """
    throttle_public_auth(request)
    try:
        identity = await webauthn_service.finish_authentication(db, config, body.credential)
    except TenancyError:
        record_auth_failure("invalid_passkey")
        raise
    token, expires_at = await create_dashboard_session(db, config.dashboard_session_ttl_hours, user_id=identity.id)
    await _commit(db, "persist a dashboard session on a passkey sign-in")
    apply_session_cookie(response, token, expires_at, secure=request_is_https(request))
    return PasskeySessionResponse(
        expires_at=expires_at,
        user_id=identity.id,
        active_organization_id=identity.active_organization_id,
    )


@router.get(
    "/credentials", response_model=WebAuthnCredentialsPublic, dependencies=[Depends(require_passkey_support)]
)
async def list_passkeys(
    identity: CurrentIdentity,
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> Any:
    """The caller's own passkeys. Never anybody else's, and never key material."""
    credentials = await webauthn_service.list_credentials(db, config, identity.id)
    data = [webauthn_service.to_public(credential) for credential in credentials]
    return WebAuthnCredentialsPublic(data=data, count=len(data))


@router.patch("/credentials/{credential_id}", response_model=WebAuthnCredentialPublic)
async def rename_passkey(
    credential_id: uuid.UUID,
    body: WebAuthnCredentialUpdate,
    identity: CurrentIdentity,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> Any:
    """Relabel one of the caller's passkeys, which is all that is editable."""
    credential = await webauthn_service.rename_credential(db, identity.id, credential_id, body.name)
    await _commit(db, "rename a passkey")
    return webauthn_service.to_public(credential)


@router.delete("/credentials/{credential_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_passkey(
    credential_id: uuid.UUID,
    identity: CurrentIdentity,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    """Remove one of the caller's passkeys.

    Removing the last one is allowed: an email and password is still this
    deployment's login, so this is not a lockout, and refusing would strand
    whoever lost the authenticator.
    """
    await webauthn_service.delete_credential(db, identity.id, credential_id)
    await _commit(db, "delete a passkey")
