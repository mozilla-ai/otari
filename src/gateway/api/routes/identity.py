"""Checks a page runs on a mailed link before it asks for a password (standalone mode only).

Public for the reason ``auth_signup.py`` gives: the token in the link is the
caller's whole proof. A check reads the token and never spends it, so a page
can say a link is expired or used on arrival rather than after a password has
been typed. It discloses no more than redeeming the link would.
"""

from fastapi import APIRouter, Request

from gateway.api.deps import IdentityServiceDep
from gateway.api.routes._public_auth import throttle_public_auth
from gateway.schemas.identity import MailedLinkRequest, MailedLinkStatusResponse

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/password/reset/validate")
async def validate_reset_link(
    body: MailedLinkRequest, request: Request, identity: IdentityServiceDep
) -> MailedLinkStatusResponse:
    """Say whether a password reset link can still be used, without using it."""
    throttle_public_auth(request)
    return MailedLinkStatusResponse(status=await identity.reset_link_status(body.token))


@router.post("/verify-email/validate")
async def validate_verification_link(
    body: MailedLinkRequest, request: Request, identity: IdentityServiceDep
) -> MailedLinkStatusResponse:
    """Say whether an email verification link can still be used, without using it.

    Verifying still needs the password chosen at signup, which this does not check.
    """
    throttle_public_auth(request)
    return MailedLinkStatusResponse(status=await identity.verification_link_status(body.token))


__all__ = ["router"]
