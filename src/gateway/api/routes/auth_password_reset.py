"""Password recovery: request a reset link, then complete it (standalone mode only).

A separate router from ``auth_password.py`` sharing its URL prefix on purpose:
that router carries a router-wide ``Depends(verify_master_key)``, and both
routes here are for a caller who holds neither the master key nor a session,
the same "the request's own token is the whole proof" shape as
``invitations.py`` and ``auth_signup.py``.
"""

from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, Request, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import (
    GrowthSignalPortDep,
    OrganizationServiceDep,
    get_config,
    get_db,
)
from gateway.api.routes._public_auth import mail_unavailable, throttle_public_auth
from gateway.api.routes.organizations import OrganizationServiceDep as MembershipOrganizationServiceDep
from gateway.core.config import GatewayConfig
from gateway.services.mail import MailNotConfiguredError
from gateway.services.tenancy import report_new_account
from gateway.services.tenancy.email_address import MAX_EMAIL_LENGTH
from gateway.services.tenancy.user_service import request_password_reset, reset_password

router = APIRouter(prefix="/auth/password", tags=["auth"])

# Same reasoning as ``auth_signup``'s bounds: a sanity ceiling on the request
# body, not the policy, which is enforced (and whose message is preserved) in
# the service.
_MAX_SUBMITTED_PASSWORD = 1024
_MAX_SUBMITTED_TOKEN = 512

# One wording for both cases a link is mailed for, a forgotten password and a
# first one, so the response says nothing about which applies to the address.
_RESET_REQUEST_MESSAGE = (
    "If this address has an account here, an email with a link to set or reset its password is on its way."
)


class RequestPasswordResetRequest(BaseModel):
    email: str = Field(max_length=MAX_EMAIL_LENGTH, description="The address to send a reset link to.")


class RequestPasswordResetResponse(BaseModel):
    message: str = Field(description="The same message whether or not a link was sent.")


class ResetPasswordRequest(BaseModel):
    token: str = Field(max_length=_MAX_SUBMITTED_TOKEN, description="The token from the reset link.")
    new_password: str = Field(
        min_length=8,
        max_length=_MAX_SUBMITTED_PASSWORD,
        description="The password to sign in with from now on. At least 8 characters, at most 72 bytes.",
    )
    terms_accepted: bool = Field(
        default=False,
        description=(
            "Whether the caller accepted this deployment's terms. Required where the bootstrap reports a "
            "terms_url and the account has not accepted them yet."
        ),
    )


@router.post("/reset")
async def request_reset(
    body: RequestPasswordResetRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    organizations: OrganizationServiceDep,
) -> RequestPasswordResetResponse:
    """Mail a password-reset link, or do nothing: the response never says which."""
    throttle_public_auth(request)
    try:
        await request_password_reset(
            db, config, organizations=organizations, background_tasks=background_tasks, email=body.email
        )
    except MailNotConfiguredError as exc:
        raise mail_unavailable(exc) from None
    return RequestPasswordResetResponse(message=_RESET_REQUEST_MESSAGE)


@router.post("/reset/confirm", status_code=status.HTTP_204_NO_CONTENT)
async def confirm_reset(
    body: ResetPasswordRequest,
    background_tasks: BackgroundTasks,
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    growth: GrowthSignalPortDep,
    organizations: MembershipOrganizationServiceDep,
) -> None:
    """Complete a password reset, or choose a first password. Single-use: the token stops working after this.

    Not throttled with the address-bearing routes, for the reason the
    verification route gives: the token is the proof.
    """
    claimed = await reset_password(
        db,
        config,
        organizations=organizations,
        token=body.token,
        new_password=body.new_password,
        terms_accepted=body.terms_accepted,
    )
    await report_new_account(growth, claimed, background_tasks=background_tasks)


__all__ = ["router"]
