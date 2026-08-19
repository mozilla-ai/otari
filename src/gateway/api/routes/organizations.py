"""Organization context and membership (standalone mode only).

Thin composition over `gateway.services.tenancy.organization_service`: resolve
the caller's identity, call the service, return its typed result. The response
models come from `gateway.models.tenancy` and are the contracts the dashboard's
generated client is built from, so they keep the platform's shapes.

Every path is scoped to ``/me``, the organization the caller's identity is
pointed at, and no request can name an organization at all. That is the tenant
boundary, and in this edition it is also the whole story: a standalone
deployment has exactly one organization, provisioned at first boot.

**Why there is no create, switch, delete, or membership list here.** Those are
what make a deployment host more than one tenant, and a self-hosted gateway is
one tenant with several people in it: many identities, fixed roles, and
workspaces as the unit teams are isolated by. The model stays tenancy-shaped
because the hosted edition needs it and the schema is edition-invariant, so
those surfaces are the overlay's to contribute, gated on an entitlement the way
`ARCHITECTURE.md` describes for every capability line. Nothing here has to
change to allow them; they are simply not mounted.
"""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import CurrentIdentity, get_config, get_db, verify_master_key
from gateway.core.config import GatewayConfig
from gateway.models.tenancy import (
    ActiveOrganizationMemberCreateRequest,
    ActiveOrganizationMemberCreateResultPublic,
    ActiveOrganizationMemberPublic,
    ActiveOrganizationMembersPublic,
    ActiveOrganizationMemberUpdateRequest,
    ActiveOrganizationUpdateRequest,
    InviteOrganizationMemberRequest,
    InviteOrganizationMemberResultPublic,
    OrganizationMembershipContextPublic,
)
from gateway.services.tenancy import OrganizationService

# Auth is declared on the router, not left to arrive through `CurrentIdentity`:
# every handler here happens to take one today, and a future handler that did
# not would be unauthenticated with nothing to notice.
router = APIRouter(
    prefix="/v1/organizations",
    tags=["organizations"],
    dependencies=[Depends(verify_master_key)],
)


class Message(BaseModel):
    """A human-readable acknowledgment for an operation with nothing to return."""

    message: str = Field(description="What happened.")


def get_organization_service(db: Annotated[AsyncSession, Depends(get_db)]) -> OrganizationService:
    """Build the organization service on the request's session."""
    return OrganizationService(db)


OrganizationServiceDep = Annotated[OrganizationService, Depends(get_organization_service)]


@router.get("/me")
async def get_active_organization_context(
    service: OrganizationServiceDep,
    current_identity: CurrentIdentity,
) -> OrganizationMembershipContextPublic:
    """Get the caller's active organization and their standing in it."""
    return await service.get_active_membership_context_for_user(current_identity)


@router.patch("/me")
async def update_active_organization(
    service: OrganizationServiceDep,
    current_identity: CurrentIdentity,
    body: ActiveOrganizationUpdateRequest,
) -> OrganizationMembershipContextPublic:
    """Rename the caller's active organization."""
    return await service.update_active_organization_for_user(
        user=current_identity,
        organization_name=body.name,
    )


@router.get("/me/members")
async def list_active_organization_members(
    service: OrganizationServiceDep,
    current_identity: CurrentIdentity,
    skip: Annotated[int, Query(ge=0, description="Number of records to skip")] = 0,
    limit: Annotated[int, Query(ge=1, le=1000, description="Maximum number of records to return")] = 100,
) -> ActiveOrganizationMembersPublic:
    """List the members of the caller's active organization."""
    return await service.list_active_organization_members_for_user(
        user=current_identity,
        skip=skip,
        limit=limit,
    )


@router.post("/me/members", status_code=status.HTTP_201_CREATED)
async def create_active_organization_member(
    service: OrganizationServiceDep,
    current_identity: CurrentIdentity,
    body: ActiveOrganizationMemberCreateRequest,
) -> ActiveOrganizationMemberCreateResultPublic:
    """Add a member to the caller's active organization, by email address.

    Organization owners and admins only. The member is active immediately and
    the response says so: this edition has no invitation to send and no way to
    accept one, so it answers on the ``active`` arm of the result rather than
    the ``invited`` one the platform uses. An address that belongs to no
    identity yet creates one, which carries the address as the handle a future
    sign-in flow will claim it by, and can do nothing until then.
    """
    return await service.create_active_organization_member_for_user(user=current_identity, request=body)


@router.patch("/me/members/{organization_member_id}")
async def update_active_organization_member(
    service: OrganizationServiceDep,
    current_identity: CurrentIdentity,
    organization_member_id: uuid.UUID,
    body: ActiveOrganizationMemberUpdateRequest,
) -> ActiveOrganizationMemberPublic:
    """Change a member's role or status. Organization owners and admins only."""
    return await service.update_active_organization_member_for_user(
        user=current_identity,
        organization_member_id=organization_member_id,
        request=body,
    )


@router.delete("/me/members/{organization_member_id}")
async def remove_active_organization_member(
    service: OrganizationServiceDep,
    current_identity: CurrentIdentity,
    organization_member_id: uuid.UUID,
) -> Message:
    """Remove a member by suspending their membership, keeping their history resolvable."""
    await service.remove_active_organization_member_for_user(
        user=current_identity,
        organization_member_id=organization_member_id,
    )
    return Message(message="Organization member removed")


@router.post("/me/member-invitations", status_code=status.HTTP_201_CREATED)
async def invite_active_organization_member(
    service: OrganizationServiceDep,
    current_identity: CurrentIdentity,
    config: Annotated[GatewayConfig, Depends(get_config)],
    body: InviteOrganizationMemberRequest,
) -> InviteOrganizationMemberResultPublic:
    """Invite an address to the caller's active organization by email.

    Organization owners and admins only. Unlike ``POST /me/members``, the
    membership lands ``invited`` rather than ``active``: it becomes active
    once the recipient accepts (``POST /v1/invitations/accept``). The response
    always carries the accept link, whether or not it was actually emailed
    (``mail_sent``), so an operator can share it themselves when mail is not
    configured or the send fails.
    """
    return await service.invite_active_organization_member_for_user(
        user=current_identity,
        request=body,
        config=config,
    )


@router.delete("/me/member-invitations/{invitation_id}")
async def revoke_active_organization_member_invitation(
    service: OrganizationServiceDep,
    current_identity: CurrentIdentity,
    invitation_id: uuid.UUID,
) -> Message:
    """Revoke an unaccepted invitation. Organization owners and admins only.

    Suspends the paired membership rather than deleting it, the same as
    removing an active member: re-inviting the same address later revives it.
    """
    await service.revoke_organization_member_invitation_for_user(
        user=current_identity,
        invitation_id=invitation_id,
    )
    return Message(message="Invitation revoked")
