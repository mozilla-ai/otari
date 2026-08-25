"""Organization context, membership, and switching (standalone mode only).

Thin composition over `gateway.services.tenancy.organization_service`: resolve
the caller's identity, call the service, return its typed result. The response
models come from `gateway.models.tenancy` and are the contracts the dashboard's
generated client is built from, so they keep the platform's shapes.

Nearly every path is scoped to ``/me``, the organization the caller's identity
is pointed at, and cannot name an organization at all. ``POST /me/switch`` is
the one that can, because moving that pointer is the one operation that has to
be told where to; it answers 404 for an organization the caller holds no active
membership in, so the boundary holds there too.

**Why create, switch, and the membership list are here.** A standalone
deployment boots one organization and almost always keeps exactly that, but a
second one is already reachable: invite an address that belongs to an
organization elsewhere on this deployment, they accept, and they hold two
memberships with nothing to switch between them. The tables are this
repository's, and so are the invariants a second organization needs (who
becomes its owner, what its slug is, that it has a workspace to work in), so an
overlay contributing no tables could only fork them. See mozilla-ai/otari#715.

Deleting an organization is still not here, and that is a different question
rather than an oversight: every historical attribution resolves through rows
that hang off it.
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
    CallerOrganizationMembershipsPublic,
    InviteOrganizationMemberRequest,
    InviteOrganizationMemberResultPublic,
    OrganizationCreateRequest,
    OrganizationMembershipContextPublic,
    OrganizationPublic,
    SwitchActiveOrganizationRequest,
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


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_organization(
    service: OrganizationServiceDep,
    current_identity: CurrentIdentity,
    body: OrganizationCreateRequest,
) -> OrganizationPublic:
    """Create an organization with the caller as its owner.

    Takes a name; the slug is derived from it with a random suffix, so two
    organizations may share a name and a later rename does not move the slug.
    A default workspace is provisioned alongside, because an organization
    without one has nowhere to hold a key, a budget or a usage row.

    The caller is **not** moved into it. Switching is a separate call
    (``POST /me/switch``), so creating an organization does not change what the
    rest of the caller's session is looking at.
    """
    return await service.create_organization_for_user(user=current_identity, request=body)


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


@router.get("/me/memberships")
async def list_caller_organization_memberships(
    service: OrganizationServiceDep,
    current_identity: CurrentIdentity,
    skip: Annotated[int, Query(ge=0, description="Number of records to skip")] = 0,
    limit: Annotated[int, Query(ge=1, le=1000, description="Maximum number of records to return")] = 100,
) -> CallerOrganizationMembershipsPublic:
    """List the organizations the caller belongs to, and their role in each.

    The caller's own active memberships, not a directory of the deployment's
    organizations: this is what an organization switcher renders, and one row
    carries ``is_active_organization`` so it can mark the current one. Not to be
    confused with ``GET /me/members``, which is the active organization's
    roster.
    """
    return await service.list_organization_memberships_for_user(
        user=current_identity,
        skip=skip,
        limit=limit,
    )


@router.post("/me/switch")
async def switch_active_organization(
    service: OrganizationServiceDep,
    current_identity: CurrentIdentity,
    body: SwitchActiveOrganizationRequest,
) -> OrganizationMembershipContextPublic:
    """Point the caller's identity at another organization they belong to.

    Distinct from ``PATCH /me``, which renames the organization already active.
    This changes which organization every later request is scoped to, so
    workspaces, keys, budgets and usage all follow it. Answers 404 for an
    organization the caller holds no active membership in, whether or not it
    exists.
    """
    return await service.switch_active_organization_for_user(
        user=current_identity,
        organization_id=body.organization_id,
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
