"""Password recovery: request a reset link, then complete it (standalone mode only).

A separate router from ``auth_password.py`` sharing its URL prefix on purpose:
that router carries a router-wide ``Depends(verify_master_key)``, and both
routes here are for a caller who holds neither the master key nor a session,
the same "the request's own token is the whole proof" shape as
``invitations.py`` and ``auth_signup.py``.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, Request, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import get_config, get_db
from gateway.api.routes._public_auth import mail_unavailable, throttle_public_auth
from gateway.core.config import GatewayConfig
from gateway.services.mail import MailNotConfiguredError
from gateway.services.tenancy.email_address import MAX_EMAIL_LENGTH
from gateway.services.tenancy.user_service import request_password_reset, reset_password

router = APIRouter(prefix="/v1/auth/password", tags=["auth"])

# Same reasoning as ``auth_signup``'s bounds: a sanity ceiling on the request
# body, not the policy, which is enforced (and whose message is preserved) in
# the service.
_MAX_SUBMITTED_PASSWORD = 1024
_MAX_SUBMITTED_TOKEN = 512

_RESET_REQUEST_MESSAGE = "If this address has a password, a reset email is on its way."


class RequestPasswordResetRequest(BaseModel):
    email: str = Field(max_length=MAX_EMAIL_LENGTH, description="The address to send a reset link to.")


class RequestPasswordResetResponse(BaseModel):
    message: str = Field(description="The same message whether or not the address has a password to reset.")


class ResetPasswordRequest(BaseModel):
    token: str = Field(max_length=_MAX_SUBMITTED_TOKEN, description="The token from the reset link.")
    new_password: str = Field(
        min_length=8,
        max_length=_MAX_SUBMITTED_PASSWORD,
        description="The password to sign in with from now on. At least 8 characters, at most 72 bytes.",
    )


@router.post("/reset")
async def request_reset(
    body: RequestPasswordResetRequest,
    request: Request,
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> RequestPasswordResetResponse:
    """Mail a password-reset link, or do nothing: the response never says which."""
    throttle_public_auth(request)
    try:
        await request_password_reset(db, config, email=body.email)
    except MailNotConfiguredError as exc:
        raise mail_unavailable(exc) from None
    return RequestPasswordResetResponse(message=_RESET_REQUEST_MESSAGE)


@router.post("/reset/confirm", status_code=status.HTTP_204_NO_CONTENT)
async def confirm_reset(
    body: ResetPasswordRequest,
    request: Request,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    """Complete a password reset. Single-use: the token stops working after this."""
    throttle_public_auth(request)
    await reset_password(db, token=body.token, new_password=body.new_password)


__all__ = ["router"]
