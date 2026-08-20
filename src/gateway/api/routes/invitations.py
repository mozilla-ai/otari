"""Accepting an organization invitation (standalone mode only).

Deliberately public, unlike every other route under ``/v1``: the person
following an emailed link holds no master key and no session, and the token
in the link is their whole proof of anything here. Both routes therefore take
no ``CurrentIdentity`` and are scoped to exactly the one invitation the token
names.

No session is minted on accept. Otari has no per-user sign-in yet (every
authenticated request resolves to the one bootstrap operator identity), so
there is nothing to sign this visitor into; accepting only resolves their
membership to ``active``, the same place ``POST /me/members`` already lands a
member added directly. They see the sign-in screen next, same as anyone else
added to an organization today.
"""

from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import get_db
from gateway.models.tenancy import (
    AcceptInvitationRequest,
    AcceptInvitationResultPublic,
    InvitationPreviewPublic,
    ValidateInvitationRequest,
)
from gateway.services.tenancy import OrganizationService

router = APIRouter(prefix="/v1/invitations", tags=["invitations"])


def get_organization_service(db: Annotated[AsyncSession, Depends(get_db)]) -> OrganizationService:
    """Build the organization service on the request's session.

    A separate copy of the same tiny factory `organizations.py` declares: the
    two route modules deliberately don't import from each other (one is
    public, one is master-key gated), and importing the dependency alone
    across that boundary is not worth it for one function.
    """
    return OrganizationService(db)


OrganizationServiceDep = Annotated[OrganizationService, Depends(get_organization_service)]


@router.post("/validate")
async def validate_invitation(
    service: OrganizationServiceDep,
    body: ValidateInvitationRequest,
) -> InvitationPreviewPublic:
    """Look up a pending invitation by its token, for the accept page to render before committing.

    A ``POST`` with the token in the body, not a ``GET`` with it in the URL:
    the token is a bearer credential, and a URL path is what an access log or
    an intermediate proxy routinely retains.
    """
    return await service.get_invitation_preview(body.token)


@router.post("/accept")
async def accept_invitation(
    service: OrganizationServiceDep,
    body: AcceptInvitationRequest,
) -> AcceptInvitationResultPublic:
    """Accept a pending invitation, resolving it to an active membership."""
    return await service.accept_invitation(body.token)
