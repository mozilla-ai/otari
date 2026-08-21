"""Signup, email verification, and resending a verification link (standalone mode only).

Deliberately public, the same reasoning ``invitations.py`` gives its two
routes: the person completing signup or opening a verification link holds no
master key and no session yet, and the token or address in the request is
their whole proof of anything here.

Signup only ever claims an identity ``organization_service`` already put on
the roster (an admin added or invited the address). It never creates one from
nothing; see ``user_service.create_user_for_signup``'s own docstring for why.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import get_config, get_db
from gateway.core.config import GatewayConfig
from gateway.services.mail import MailNotConfiguredError
from gateway.services.tenancy.email_address import MAX_EMAIL_LENGTH
from gateway.services.tenancy.user_service import (
    create_user_for_signup,
    resend_verification_email,
    verify_email,
)

router = APIRouter(prefix="/v1/auth", tags=["auth"])

# A generous sanity ceiling on the submitted password, not the policy: the
# policy (length, bcrypt's 72-byte ceiling) is enforced in the service so its
# message survives, matching auth_password.py's own ``_MAX_SUBMITTED_PASSWORD``.
_MAX_SUBMITTED_PASSWORD = 1024
# A generated token is ~43 characters (``secrets.token_urlsafe(32)``); this is a
# sanity ceiling on the request body, not a format, matching the master key
# bound in ``auth_session.CreateSessionRequest``.
_MAX_SUBMITTED_TOKEN = 512


class SignupRequest(BaseModel):
    """Claim an identity already on the roster by setting its password."""

    email: str = Field(max_length=MAX_EMAIL_LENGTH, description="The address an admin added or invited.")
    password: str = Field(
        min_length=8,
        max_length=_MAX_SUBMITTED_PASSWORD,
        description="The password to sign in with once verified. At least 8 characters, at most 72 bytes.",
    )
    full_name: str | None = Field(default=None, max_length=255, description="Filled in only if not already set.")
    terms_accepted: bool = Field(default=False, description="Whether the caller accepted this deployment's terms.")


class SignupResponse(BaseModel):
    """What happens next: a verification message, not a session."""

    email: str = Field(description="The address a verification link was sent to.")
    message: str = Field(description="What the caller should do next.")


class VerifyEmailRequest(BaseModel):
    token: str = Field(max_length=_MAX_SUBMITTED_TOKEN, description="The token from the verification link.")


class VerifyEmailResponse(BaseModel):
    email: str = Field(description="The address that is now verified.")


class ResendVerificationRequest(BaseModel):
    email: str = Field(max_length=MAX_EMAIL_LENGTH, description="The address to resend a verification link to.")


class ResendVerificationResponse(BaseModel):
    message: str = Field(description="The same message whether or not the address has anything to verify.")


_RESEND_MESSAGE = "If this address is registered and unverified, a verification email is on its way."


def _throttle(request: Request) -> None:
    """Throttle calls to these routes per client IP.

    Unconditional, not just on failure, the same reasoning ``invitations._throttle``
    gives its own two routes: there is no legitimate caller here at a rate worth
    exempting, only a client with an address it can retry from. Reuses the
    sign-in route's limiter/budget rather than a separate one.
    """
    limiter = getattr(request.app.state, "login_rate_limiter", None)
    if limiter is None:
        return
    client_ip = request.client.host if request.client else None
    if client_ip is None:
        return
    limiter.check(client_ip)


def _as_503(exc: MailNotConfiguredError) -> HTTPException:
    """Render a mail-gated refusal the way ``mail.py``'s own does.

    Not a ``TenancyError``: the central handler in ``gateway.main`` renders
    every status of 500 or above with a generic "Internal server error" body,
    on purpose, for the errors that already live in that family (an operator
    problem the caller cannot act on). This one is the opposite: the missing
    settings are exactly what the caller (or the dashboard reading them) needs
    to see, the same reason ``send_test_mail`` raises ``HTTPException``
    directly instead of wrapping this in a tenancy error.
    """
    return HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))


@router.post("/signup")
async def signup(
    body: SignupRequest,
    request: Request,
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> SignupResponse:
    """Claim a roster identity: set its password and send a verification link.

    No session is minted. The identity is hard-blocked from signing in until it
    verifies, so there is nothing yet to sign it into.
    """
    _throttle(request)
    try:
        identity = await create_user_for_signup(
            db,
            config,
            email=body.email,
            password=body.password,
            full_name=body.full_name,
            terms_accepted=body.terms_accepted,
        )
    except MailNotConfiguredError as exc:
        raise _as_503(exc) from None
    assert identity.email is not None  # guaranteed: signup only claims an address-holding row
    return SignupResponse(
        email=identity.email,
        message="Check your email to verify your address, then sign in.",
    )


@router.post("/verify-email")
async def verify_email_route(
    body: VerifyEmailRequest,
    request: Request,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> VerifyEmailResponse:
    """Confirm an address from its verification link, lifting the sign-in gate."""
    _throttle(request)
    identity = await verify_email(db, token=body.token)
    assert identity.email is not None  # guaranteed: only a claimed identity has a verification token
    return VerifyEmailResponse(email=identity.email)


@router.post("/resend-verification")
async def resend_verification(
    body: ResendVerificationRequest,
    request: Request,
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> ResendVerificationResponse:
    """Mail a fresh verification link, or do nothing: the response never says which."""
    _throttle(request)
    try:
        await resend_verification_email(db, config, email=body.email)
    except MailNotConfiguredError as exc:
        raise _as_503(exc) from None
    return ResendVerificationResponse(message=_RESEND_MESSAGE)


__all__ = ["router"]
