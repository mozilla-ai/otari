"""Workspaces and workspace membership (standalone mode only).

Thin composition over `gateway.services.tenancy.workspace_service`. Every path
resolves its workspace inside the caller's active organization, so a workspace id
from another tenant answers 404 rather than reaching the service's logic.
"""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import CurrentIdentity, get_db
from gateway.api.routes.organizations import Message
from gateway.models.tenancy import (
    WorkspaceCreate,
    WorkspaceMemberPublic,
    WorkspaceMembersPublic,
    WorkspacePublic,
    WorkspacesPublic,
    WorkspaceUpdate,
)
from gateway.services.tenancy import WorkspaceService

router = APIRouter(prefix="/v1/workspaces", tags=["workspaces"])

WORKSPACE_ROLE_DESCRIPTION = "Role to assign: owner, admin, member, or viewer."


def get_workspace_service(db: Annotated[AsyncSession, Depends(get_db)]) -> WorkspaceService:
    """Build the workspace service on the request's session."""
    return WorkspaceService(db)


WorkspaceServiceDep = Annotated[WorkspaceService, Depends(get_workspace_service)]


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_workspace(
    service: WorkspaceServiceDep,
    current_identity: CurrentIdentity,
    body: WorkspaceCreate,
) -> WorkspacePublic:
    """Create a workspace in the caller's organization. Owners and admins only."""
    return await service.create_workspace(user=current_identity, workspace_create=body)


@router.get("")
async def list_workspaces(
    service: WorkspaceServiceDep,
    current_identity: CurrentIdentity,
    skip: Annotated[int, Query(ge=0, description="Number of records to skip")] = 0,
    limit: Annotated[int, Query(ge=1, le=1000, description="Maximum number of records to return")] = 100,
) -> WorkspacesPublic:
    """List the workspaces the caller can see in their organization."""
    return await service.list_workspaces(user=current_identity, skip=skip, limit=limit)


@router.get("/{workspace_id}")
async def get_workspace(
    service: WorkspaceServiceDep,
    current_identity: CurrentIdentity,
    workspace_id: uuid.UUID,
) -> WorkspacePublic:
    """Get one workspace."""
    return await service.get_workspace(user=current_identity, workspace_id=workspace_id)


@router.patch("/{workspace_id}")
async def update_workspace(
    service: WorkspaceServiceDep,
    current_identity: CurrentIdentity,
    workspace_id: uuid.UUID,
    body: WorkspaceUpdate,
) -> WorkspacePublic:
    """Rename a workspace or change its description."""
    return await service.update_workspace(
        user=current_identity,
        workspace_id=workspace_id,
        workspace_update=body,
    )


@router.delete("/{workspace_id}")
async def delete_workspace(
    service: WorkspaceServiceDep,
    current_identity: CurrentIdentity,
    workspace_id: uuid.UUID,
) -> Message:
    """Delete a workspace and its memberships. Organization owners and admins only."""
    await service.delete_workspace(user=current_identity, workspace_id=workspace_id)
    return Message(message="Workspace deleted")


@router.get("/{workspace_id}/members")
async def list_workspace_members(
    service: WorkspaceServiceDep,
    current_identity: CurrentIdentity,
    workspace_id: uuid.UUID,
    skip: Annotated[int, Query(ge=0, description="Number of records to skip")] = 0,
    limit: Annotated[int, Query(ge=1, le=1000, description="Maximum number of records to return")] = 100,
) -> WorkspaceMembersPublic:
    """List a workspace's members."""
    return await service.list_members(
        user=current_identity,
        workspace_id=workspace_id,
        skip=skip,
        limit=limit,
    )


@router.post("/{workspace_id}/members/{user_id}", status_code=status.HTTP_201_CREATED)
async def add_workspace_member(
    service: WorkspaceServiceDep,
    current_identity: CurrentIdentity,
    workspace_id: uuid.UUID,
    user_id: uuid.UUID,
    role: Annotated[str, Query(description=WORKSPACE_ROLE_DESCRIPTION)] = "member",
) -> WorkspaceMemberPublic:
    """Add an existing organization member to a workspace."""
    return await service.add_member(
        user=current_identity,
        workspace_id=workspace_id,
        user_id=user_id,
        role=role,
    )


@router.patch("/{workspace_id}/members/{user_id}")
async def update_workspace_member_role(
    service: WorkspaceServiceDep,
    current_identity: CurrentIdentity,
    workspace_id: uuid.UUID,
    user_id: uuid.UUID,
    role: Annotated[str, Query(description=WORKSPACE_ROLE_DESCRIPTION)],
) -> WorkspaceMemberPublic:
    """Change a workspace member's role."""
    return await service.update_member_role(
        user=current_identity,
        workspace_id=workspace_id,
        user_id=user_id,
        role=role,
    )


@router.delete("/{workspace_id}/members/{user_id}")
async def remove_workspace_member(
    service: WorkspaceServiceDep,
    current_identity: CurrentIdentity,
    workspace_id: uuid.UUID,
    user_id: uuid.UUID,
) -> Message:
    """Remove a member from a workspace. Idempotent."""
    await service.remove_member(user=current_identity, workspace_id=workspace_id, user_id=user_id)
    return Message(message="Workspace member removed")
