"""Signup, email verification, and resending a verification link (standalone mode only).

Deliberately public, the same reasoning ``invitations.py`` gives its two
routes: the person completing signup or opening a verification link holds no
master key and no session yet, and the token or address in the request is
their whole proof of anything here.

Signup only ever claims an identity ``organization_service`` already put on
the roster (an admin added or invited the address). It never creates one from
nothing; see ``user_service.create_user_for_signup``'s own docstring for why.
It is also enumeration-safe the same way resend and reset-request are: the
response never says whether the address was unknown, already claimed, or
genuinely just claimed.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import get_config, get_db
from gateway.api.routes._public_auth import mail_unavailable, throttle_public_auth
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
    """The same message whether or not the address had anything to claim."""

    message: str = Field(description="What the caller should do next.")


class VerifyEmailRequest(BaseModel):
    token: str = Field(max_length=_MAX_SUBMITTED_TOKEN, description="The token from the verification link.")


class VerifyEmailResponse(BaseModel):
    email: str = Field(description="The address that is now verified.")


class ResendVerificationRequest(BaseModel):
    email: str = Field(max_length=MAX_EMAIL_LENGTH, description="The address to resend a verification link to.")


class ResendVerificationResponse(BaseModel):
    message: str = Field(description="The same message whether or not the address has anything to verify.")


_SIGNUP_MESSAGE = "If this address is on our roster and unclaimed, check your email to verify it, then sign in."
_RESEND_MESSAGE = "If this address is registered and unverified, a verification email is on its way."


@router.post("/signup")
async def signup(
    body: SignupRequest,
    request: Request,
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> SignupResponse:
    """Claim a roster identity, or do nothing: the response never says which.

    No session is minted. A newly claimed identity is hard-blocked from
    signing in until it verifies, so there is nothing yet to sign it into.
    """
    throttle_public_auth(request)
    try:
        await create_user_for_signup(
            db,
            config,
            email=body.email,
            password=body.password,
            full_name=body.full_name,
            terms_accepted=body.terms_accepted,
        )
    except MailNotConfiguredError as exc:
        raise mail_unavailable(exc) from None
    return SignupResponse(message=_SIGNUP_MESSAGE)


@router.post("/verify-email")
async def verify_email_route(
    body: VerifyEmailRequest,
    request: Request,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> VerifyEmailResponse:
    """Confirm an address from its verification link, lifting the sign-in gate."""
    throttle_public_auth(request)
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
    throttle_public_auth(request)
    try:
        await resend_verification_email(db, config, email=body.email)
    except MailNotConfiguredError as exc:
        raise mail_unavailable(exc) from None
    return ResendVerificationResponse(message=_RESEND_MESSAGE)


__all__ = ["router"]
