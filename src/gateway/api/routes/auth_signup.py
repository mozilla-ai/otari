"""Signup, email verification, and resending a verification link (standalone mode only).

Deliberately public, the same reasoning ``invitations.py`` gives its two
routes: the person completing signup or opening a verification link holds no
master key and no session yet, and the token or address in the request is
their whole proof of anything here.

What signup may do depends on ``open_signup``. Off, the default, it only ever
claims an identity ``organization_service`` already put on the roster (an admin
added or invited the address) and creates nothing from nothing. On, an address
nobody has added is registered instead, with an organization of its own. See
``user_service.create_user_for_signup``'s own docstring for both.

Either way it is enumeration-safe the same way resend and reset-request are: the
response never says whether the address was unknown, already claimed, or
genuinely just claimed.
"""

from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, Request
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import (
    GrowthSignalPortDep,
    OrganizationServiceDep,
    UnitOfWorkDep,
    get_config,
    get_db,
)
from gateway.api.routes._public_auth import (
    count_failed_verification,
    mail_unavailable,
    refuse_exhausted_verification_attempts,
    throttle_public_auth,
)
from gateway.api.routes.organizations import OrganizationServiceDep as MembershipOrganizationServiceDep
from gateway.core.config import GatewayConfig
from gateway.exceptions.identity_exceptions import VerificationTokenInvalidError
from gateway.models.tenancy import MAX_FULL_NAME_LENGTH
from gateway.services.mail import MailNotConfiguredError
from gateway.services.tenancy import report_new_account
from gateway.services.tenancy.email_address import MAX_EMAIL_LENGTH
from gateway.services.tenancy.user_service import (
    create_user_for_signup,
    resend_verification_email,
    verify_email,
)

router = APIRouter(prefix="/auth", tags=["auth"])

# A generous sanity ceiling on the submitted password, not the policy: the
# policy (length, bcrypt's 72-byte ceiling) is enforced in the service so its
# message survives, matching auth_password.py's own ``_MAX_SUBMITTED_PASSWORD``.
_MAX_SUBMITTED_PASSWORD = 1024
# A generated token is ~43 characters (``secrets.token_urlsafe(32)``); this is a
# sanity ceiling on the request body, not a format, matching the master key
# bound in ``auth_session.CreateSessionRequest``.
_MAX_SUBMITTED_TOKEN = 512


class SignupRequest(BaseModel):
    """Set a password for an address, claiming or registering it."""

    email: str = Field(
        max_length=MAX_EMAIL_LENGTH,
        description=(
            "The address to sign in with. An address an admin added or invited where this "
            "deployment keeps signup closed; any address where the bootstrap reports open_signup."
        ),
    )
    password: str = Field(
        min_length=8,
        max_length=_MAX_SUBMITTED_PASSWORD,
        description="The password to sign in with once verified. At least 8 characters, at most 72 bytes.",
    )
    full_name: str | None = Field(
        default=None, max_length=MAX_FULL_NAME_LENGTH, description="Filled in only if not already set."
    )
    terms_accepted: bool = Field(
        default=False,
        description=(
            "Whether the caller accepted this deployment's terms. Required where the bootstrap reports a terms_url."
        ),
    )


class SignupResponse(BaseModel):
    """The same message whether or not the address had anything to claim."""

    message: str = Field(description="What the caller should do next.")


class VerifyEmailRequest(BaseModel):
    token: str = Field(max_length=_MAX_SUBMITTED_TOKEN, description="The token from the verification link.")
    password: str = Field(
        max_length=_MAX_SUBMITTED_PASSWORD,
        description=(
            "The password chosen at signup. The link verifies the address only together with it, "
            "so a link that reaches someone who did not sign up verifies nothing."
        ),
    )


class VerifyEmailResponse(BaseModel):
    email: str = Field(description="The address that is now verified.")


class ResendVerificationRequest(BaseModel):
    email: str = Field(max_length=MAX_EMAIL_LENGTH, description="The address to resend a verification link to.")


class ResendVerificationResponse(BaseModel):
    message: str = Field(description="The same message whether or not the address has anything to verify.")


# One message per posture, chosen by the deployment's setting and never by the
# address: that is what keeps it enumeration-safe. The closed wording would
# misdescribe an open deployment (where there is no roster to be on) and the
# open wording would promise a closed one an account it will not create.
_SIGNUP_MESSAGE = "If this address is on our roster and unclaimed, check your email to verify it, then sign in."
_OPEN_SIGNUP_MESSAGE = "If this address can be signed up, check your email to verify it, then sign in."
_RESEND_MESSAGE = "If this address is registered and unverified, a verification email is on its way."


@router.post("/signup")
async def signup(
    body: SignupRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    growth: GrowthSignalPortDep,
    uow: UnitOfWorkDep,
    organizations: MembershipOrganizationServiceDep,
) -> SignupResponse:
    """Claim a roster identity, register a new one, or do nothing: the response never says which.

    Which of the three this deployment will do is ``open_signup``, published in
    the bootstrap so the page can say so before anyone types an address.

    No session is minted. A newly claimed or registered identity is hard-blocked
    from signing in until it verifies, so there is nothing yet to sign it into.

    Refused with a 400 where the deployment publishes terms and ``terms_accepted``
    is not set, whatever the address.
    """
    throttle_public_auth(request)
    try:
        claimed = await create_user_for_signup(
            db,
            config,
            background_tasks=background_tasks,
            email=body.email,
            password=body.password,
            uow=uow,
            organizations=organizations,
            full_name=body.full_name,
            terms_accepted=body.terms_accepted,
        )
    except MailNotConfiguredError as exc:
        raise mail_unavailable(exc) from None
    await report_new_account(growth, claimed, background_tasks=background_tasks)
    return SignupResponse(message=_OPEN_SIGNUP_MESSAGE if config.open_signup else _SIGNUP_MESSAGE)


@router.post("/verify-email")
async def verify_email_route(
    body: VerifyEmailRequest,
    request: Request,
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    organizations: MembershipOrganizationServiceDep,
) -> VerifyEmailResponse:
    """Confirm an address from its verification link and the password chosen at signup, lifting the sign-in gate.

    One refusal for a link that is unknown, expired or spent and for a password
    that does not match, and a refusal consumes nothing. Refusals are counted
    per client IP and answer 429 once over the limit, so a held link cannot be
    used to guess its password; a verification that succeeds is never counted.
    """
    refuse_exhausted_verification_attempts(request)
    try:
        identity = await verify_email(
            db, organizations=organizations, token=body.token, password=body.password, open_signup=config.open_signup
        )
    except VerificationTokenInvalidError:
        count_failed_verification(request)
        raise
    assert identity.email is not None  # guaranteed: only a claimed identity has a verification token
    return VerifyEmailResponse(email=identity.email)


@router.post("/resend-verification")
async def resend_verification(
    body: ResendVerificationRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    organizations: OrganizationServiceDep,
) -> ResendVerificationResponse:
    """Mail a fresh verification link, or do nothing: the response never says which."""
    throttle_public_auth(request)
    try:
        await resend_verification_email(
            db, config, organizations=organizations, background_tasks=background_tasks, email=body.email
        )
    except MailNotConfiguredError as exc:
        raise mail_unavailable(exc) from None
    return ResendVerificationResponse(message=_RESEND_MESSAGE)


__all__ = ["router"]
