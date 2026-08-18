"""Organization context and membership (standalone mode only).

Thin composition over `gateway.services.tenancy.organization_service`: resolve
the caller's identity, call the service, return its typed result. The response
models come from `gateway.models.tenancy` and are the contracts the dashboard's
generated client is built from, so they keep the platform's shapes.

Every path is scoped to ``/me``, the organization the caller's identity is
pointed at. That is the tenant boundary: apart from ``/me/switch``, which checks
membership before it moves the pointer, no request can name an organization at
all.
"""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import CurrentIdentity, get_db
from gateway.models.tenancy import (
    ActiveOrganizationMemberCreateRequest,
    ActiveOrganizationMemberCreateResultPublic,
    ActiveOrganizationMemberPublic,
    ActiveOrganizationMembersPublic,
    ActiveOrganizationMemberUpdateRequest,
    ActiveOrganizationUpdateRequest,
    OrganizationCreateRequest,
    OrganizationMembershipContextPublic,
    OrganizationMembershipContextsPublic,
    OrganizationSwitchRequest,
)
from gateway.services.tenancy import OrganizationService

router = APIRouter(prefix="/v1/organizations", tags=["organizations"])


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


@router.get("/me/memberships")
async def list_organization_memberships(
    service: OrganizationServiceDep,
    current_identity: CurrentIdentity,
) -> OrganizationMembershipContextsPublic:
    """List every organization the caller is an active member of."""
    return await service.list_membership_contexts_for_user(current_identity)


@router.post("/me", status_code=status.HTTP_201_CREATED)
async def create_organization(
    service: OrganizationServiceDep,
    current_identity: CurrentIdentity,
    body: OrganizationCreateRequest,
) -> OrganizationMembershipContextPublic:
    """Create an organization owned by the caller, and switch them into it."""
    return await service.create_organization_for_user(
        user=current_identity,
        organization_name=body.name,
    )


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


@router.delete("/me")
async def delete_active_organization(
    service: OrganizationServiceDep,
    current_identity: CurrentIdentity,
) -> Message:
    """Delete the caller's active organization. Owners only."""
    await service.delete_active_organization(current_user=current_identity)
    return Message(message="Organization deleted")


@router.post("/me/switch")
async def switch_active_organization(
    service: OrganizationServiceDep,
    current_identity: CurrentIdentity,
    body: OrganizationSwitchRequest,
) -> OrganizationMembershipContextPublic:
    """Point the caller at another organization they are a member of."""
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
