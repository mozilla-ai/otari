"""Organization-scoped provider keys (standalone mode only).

Thin composition over `gateway.services.tenancy.org_provider_key_service`.
Two routers, because the surface has two scopes: an organization's own keys
(created, archived, defaulted) under ``/v1/organizations/me/provider-keys`,
matching `organizations.py`'s ``/me`` convention, and one workspace's view of
those keys (override, model-restrict) under
``/v1/workspaces/{workspace_id}/provider-keys``, matching `workspaces.py`'s
path-scoped convention. A caller manages several workspaces in one
organization, so the workspace surface takes ``workspace_id`` as a path
parameter the way `workspaces.py` does, unlike the organization surface's
``/me``.
"""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import CurrentIdentity, get_db, verify_master_key
from gateway.api.routes.organizations import Message
from gateway.models.provider_keys import (
    OrgProviderKeyCreateRequest,
    OrgProviderKeyPublic,
    OrgProviderKeysPublic,
    OrgProviderKeyUpdateRequest,
    WorkspaceProviderKeyOverridePublic,
    WorkspaceProviderKeyOverrideRequest,
    WorkspaceProviderKeyOverridesPublic,
    WorkspaceProviderModelRestrictionRequest,
    WorkspaceProviderModelRestrictionsPublic,
)
from gateway.services.tenancy import OrgProviderKeyService

# Auth is declared on the router, not left to arrive through `CurrentIdentity`:
# see organizations.py/workspaces.py for the same note.
org_router = APIRouter(
    prefix="/v1/organizations/me/provider-keys",
    tags=["provider-keys"],
    dependencies=[Depends(verify_master_key)],
)

workspace_router = APIRouter(
    prefix="/v1/workspaces/{workspace_id}/provider-keys",
    tags=["provider-keys"],
    dependencies=[Depends(verify_master_key)],
)


def get_org_provider_key_service(db: Annotated[AsyncSession, Depends(get_db)]) -> OrgProviderKeyService:
    """Build the org provider key service on the request's session."""
    return OrgProviderKeyService(db)


OrgProviderKeyServiceDep = Annotated[OrgProviderKeyService, Depends(get_org_provider_key_service)]


# ==============================================================================
# Organization-scoped keys
# ==============================================================================


@org_router.get("")
async def list_org_provider_keys(
    service: OrgProviderKeyServiceDep,
    current_identity: CurrentIdentity,
    include_archived: Annotated[bool, Query(description="Include archived keys.")] = False,
    skip: Annotated[int, Query(ge=0, description="Number of records to skip")] = 0,
    limit: Annotated[int, Query(ge=1, le=1000, description="Maximum number of records to return")] = 100,
) -> OrgProviderKeysPublic:
    """List the caller's organization's provider keys. Any active member may read it."""
    return await service.list_keys_for_user(
        user=current_identity, include_archived=include_archived, skip=skip, limit=limit
    )


@org_router.post("", status_code=status.HTTP_201_CREATED)
async def create_org_provider_key(
    service: OrgProviderKeyServiceDep,
    current_identity: CurrentIdentity,
    body: OrgProviderKeyCreateRequest,
) -> OrgProviderKeyPublic:
    """Create a provider key in the caller's organization. Organization owners and admins only."""
    return await service.create_key_for_user(user=current_identity, request=body)


@org_router.patch("/{key_id}")
async def update_org_provider_key(
    service: OrgProviderKeyServiceDep,
    current_identity: CurrentIdentity,
    key_id: uuid.UUID,
    body: OrgProviderKeyUpdateRequest,
) -> OrgProviderKeyPublic:
    """Change a key's name, credential, base URL, or client args. Organization owners and admins only."""
    return await service.update_key_for_user(user=current_identity, key_id=key_id, request=body)


@org_router.post("/{key_id}/archive")
async def archive_org_provider_key(
    service: OrgProviderKeyServiceDep,
    current_identity: CurrentIdentity,
    key_id: uuid.UUID,
) -> OrgProviderKeyPublic:
    """Archive a key. Organization owners and admins only."""
    return await service.archive_key_for_user(user=current_identity, key_id=key_id)


@org_router.post("/{key_id}/restore")
async def restore_org_provider_key(
    service: OrgProviderKeyServiceDep,
    current_identity: CurrentIdentity,
    key_id: uuid.UUID,
) -> OrgProviderKeyPublic:
    """Restore an archived key. Organization owners and admins only."""
    return await service.restore_key_for_user(user=current_identity, key_id=key_id)


@org_router.delete("/{key_id}")
async def delete_org_provider_key(
    service: OrgProviderKeyServiceDep,
    current_identity: CurrentIdentity,
    key_id: uuid.UUID,
) -> Message:
    """Permanently delete an archived key. Organization owners and admins only."""
    await service.delete_key_for_user(user=current_identity, key_id=key_id)
    return Message(message="Provider key deleted")


@org_router.post("/{key_id}/default")
async def set_org_provider_key_default(
    service: OrgProviderKeyServiceDep,
    current_identity: CurrentIdentity,
    key_id: uuid.UUID,
) -> OrgProviderKeyPublic:
    """Make a key the organization's default for its provider. Organization owners and admins only."""
    return await service.set_org_default_for_user(user=current_identity, key_id=key_id)


# ==============================================================================
# Workspace overrides and model restrictions
# ==============================================================================


@workspace_router.get("")
async def list_workspace_provider_keys(
    service: OrgProviderKeyServiceDep,
    current_identity: CurrentIdentity,
    workspace_id: uuid.UUID,
) -> WorkspaceProviderKeyOverridesPublic:
    """The effective view of every key visible to this workspace. Any member of the workspace may read it."""
    return await service.list_effective_keys_for_workspace(user=current_identity, workspace_id=workspace_id)


@workspace_router.patch("/{key_id}")
async def set_workspace_provider_key_override(
    service: OrgProviderKeyServiceDep,
    current_identity: CurrentIdentity,
    workspace_id: uuid.UUID,
    key_id: uuid.UUID,
    body: WorkspaceProviderKeyOverrideRequest,
) -> WorkspaceProviderKeyOverridePublic:
    """Pin or disable a key for this workspace. Organization owners/admins or this workspace's owners/admins."""
    return await service.set_workspace_override_for_user(
        user=current_identity,
        workspace_id=workspace_id,
        key_id=key_id,
        request=body,
    )


@workspace_router.delete("/{key_id}")
async def reset_workspace_provider_key_override(
    service: OrgProviderKeyServiceDep,
    current_identity: CurrentIdentity,
    workspace_id: uuid.UUID,
    key_id: uuid.UUID,
) -> Message:
    """Remove this workspace's override, reverting to full inheritance. Idempotent."""
    await service.reset_workspace_override_for_user(user=current_identity, workspace_id=workspace_id, key_id=key_id)
    return Message(message="Provider key override reset")


@workspace_router.get("/{key_id}/models")
async def list_workspace_provider_key_model_restrictions(
    service: OrgProviderKeyServiceDep,
    current_identity: CurrentIdentity,
    workspace_id: uuid.UUID,
    key_id: uuid.UUID,
) -> WorkspaceProviderModelRestrictionsPublic:
    """List this workspace's model allow-list for a key. Empty means every model is allowed."""
    return await service.list_model_restrictions_for_user(
        user=current_identity,
        workspace_id=workspace_id,
        key_id=key_id,
    )


@workspace_router.post("/{key_id}/models", status_code=status.HTTP_201_CREATED)
async def add_workspace_provider_key_model_restriction(
    service: OrgProviderKeyServiceDep,
    current_identity: CurrentIdentity,
    workspace_id: uuid.UUID,
    key_id: uuid.UUID,
    body: WorkspaceProviderModelRestrictionRequest,
) -> Message:
    """Narrow this workspace's allow-list for a key to include one more model. Idempotent."""
    await service.add_model_restriction_for_user(
        user=current_identity,
        workspace_id=workspace_id,
        key_id=key_id,
        model=body.model,
    )
    return Message(message="Model restriction added")


@workspace_router.delete("/{key_id}/models/{model:path}")
async def remove_workspace_provider_key_model_restriction(
    service: OrgProviderKeyServiceDep,
    current_identity: CurrentIdentity,
    workspace_id: uuid.UUID,
    key_id: uuid.UUID,
    model: str,
) -> Message:
    """Remove one model from this workspace's allow-list for a key. Idempotent."""
    await service.remove_model_restriction_for_user(
        user=current_identity,
        workspace_id=workspace_id,
        key_id=key_id,
        model=model,
    )
    return Message(message="Model restriction removed")
