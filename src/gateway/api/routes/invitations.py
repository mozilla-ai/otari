"""Accepting an organization invitation (standalone mode only).

Deliberately public: the person following an emailed link holds no master
key and no session, and the token in the link is their whole proof of
anything here. Both routes therefore take no ``CurrentIdentity`` and are
scoped to exactly the one invitation the token names.

No session is minted on accept: accepting resolves the membership to
``active``, the same place ``POST /me/members`` already lands a member added
directly. An invitee who has never signed in can set a password in the same
call, which is the only way in on a deployment that sends no mail; without one,
the identity stays password-less until ``POST /api/v1/auth/signup`` or a
provider sign-in claims it.
"""

from fastapi import APIRouter, BackgroundTasks, Request

from gateway.api.deps import ClaimingOrganizationServiceDep, GrowthSignalPortDep, OrganizationServiceDep
from gateway.api.routes._public_auth import throttle_public_auth
from gateway.models.tenancy import (
    AcceptInvitationRequest,
    AcceptInvitationResultPublic,
    InvitationPreviewPublic,
    ValidateInvitationRequest,
)

router = APIRouter(prefix="/invitations", tags=["invitations"])


@router.post("/validate")
async def validate_invitation(
    request: Request,
    service: OrganizationServiceDep,
    body: ValidateInvitationRequest,
) -> InvitationPreviewPublic:
    """Look up a pending invitation by its token, for the accept page to render before committing.

    A ``POST`` with the token in the body, not a ``GET`` with it in the URL:
    the token is a bearer credential, and a URL path is what an access log or
    an intermediate proxy routinely retains.
    """
    throttle_public_auth(request)
    return await service.get_invitation_preview(body.token)


@router.post("/accept")
async def accept_invitation(
    request: Request,
    background_tasks: BackgroundTasks,
    service: ClaimingOrganizationServiceDep,
    growth: GrowthSignalPortDep,
    body: AcceptInvitationRequest,
) -> AcceptInvitationResultPublic:
    """Accept a pending invitation, resolving it to an active membership and optionally setting a first password.

    Setting a first password makes the account usable for the first time, so it
    is reported to ``GrowthSignalPort`` as a signup.
    """
    throttle_public_auth(request)
    acceptance = await service.accept_invitation(
        body.token,
        password=body.password,
        full_name=body.full_name,
        terms_accepted=body.terms_accepted,
    )
    claimed = acceptance.claimed
    if claimed is not None and claimed.email is not None:
        await growth.record_signup(
            background_tasks=background_tasks,
            user_id=claimed.id,
            email=claimed.email,
            full_name=claimed.full_name,
            created_at=claimed.created_at,
        )
    return acceptance.result
