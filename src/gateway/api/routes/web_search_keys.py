"""An organization's own web search and fetch keys, and each workspace's choice among them.

Thin over ``services/tools/_web_search_keys.py``. Two routers, as for provider keys:
the organization's keys under ``/organizations/me/web-search-keys``, owners and admins
only, and one workspace's view of them under ``/workspaces/{workspace_id}/web-search-keys``,
which any member reads and the workspace's owners and admins change.
"""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Query, status

from gateway.api.deps import CurrentIdentity, WebSearchKeyServiceDep, verify_master_key
from gateway.schemas.tools import (
    OrgWebSearchKeyCreateRequest,
    OrgWebSearchKeyPublic,
    OrgWebSearchKeysPublic,
    OrgWebSearchKeyUpdateRequest,
    WorkspaceWebSearchKeyOverrideRequest,
    WorkspaceWebSearchKeysPublic,
)

org_router = APIRouter(
    prefix="/organizations/me/web-search-keys",
    tags=["web-search-keys"],
    dependencies=[Depends(verify_master_key)],
)

workspace_router = APIRouter(
    prefix="/workspaces/{workspace_id}/web-search-keys",
    tags=["web-search-keys"],
    dependencies=[Depends(verify_master_key)],
)


@org_router.get("")
async def list_org_web_search_keys(
    service: WebSearchKeyServiceDep,
    current_identity: CurrentIdentity,
    include_archived: Annotated[bool, Query(description="Include archived keys.")] = False,
    skip: Annotated[int, Query(ge=0, description="Number of records to skip")] = 0,
    limit: Annotated[int, Query(ge=1, le=1000, description="Maximum number of records to return")] = 100,
) -> OrgWebSearchKeysPublic:
    """List the caller's organization's web search and fetch keys. Organization owners and admins only."""
    return await service.list_keys(user=current_identity, include_archived=include_archived, skip=skip, limit=limit)


@org_router.post("", status_code=status.HTTP_201_CREATED)
async def create_org_web_search_key(
    service: WebSearchKeyServiceDep, current_identity: CurrentIdentity, body: OrgWebSearchKeyCreateRequest
) -> OrgWebSearchKeyPublic:
    """Add a search or fetch key to the caller's organization. Organization owners and admins only.

    A workspace with a usable key of a kind uses it for that kind's calls rather than the deployment's instance.
    """
    return await service.create_key(user=current_identity, request=body)


@org_router.patch("/{key_id}")
async def update_org_web_search_key(
    service: WebSearchKeyServiceDep,
    current_identity: CurrentIdentity,
    key_id: uuid.UUID,
    body: OrgWebSearchKeyUpdateRequest,
) -> OrgWebSearchKeyPublic:
    """Rename a key, replace its secret, or change its api_base or options. Organization owners and admins only.

    Its kind and provider are fixed. A new api_base on a key that holds a secret
    needs that secret in the same request, `api_key` and any masked `options`
    entry, since the next call sends it to the new host. Clearing api_base needs
    neither.
    """
    return await service.update_key(user=current_identity, key_id=key_id, request=body)


@org_router.post("/{key_id}/archive")
async def archive_org_web_search_key(
    service: WebSearchKeyServiceDep, current_identity: CurrentIdentity, key_id: uuid.UUID
) -> OrgWebSearchKeyPublic:
    """Take a key out of use. Organization owners and admins only."""
    return await service.archive_key(user=current_identity, key_id=key_id)


@org_router.post("/{key_id}/restore")
async def restore_org_web_search_key(
    service: WebSearchKeyServiceDep, current_identity: CurrentIdentity, key_id: uuid.UUID
) -> OrgWebSearchKeyPublic:
    """Put an archived key back in use. Organization owners and admins only."""
    return await service.restore_key(user=current_identity, key_id=key_id)


@org_router.delete("/{key_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_org_web_search_key(
    service: WebSearchKeyServiceDep, current_identity: CurrentIdentity, key_id: uuid.UUID
) -> None:
    """Delete an archived key. Organization owners and admins only."""
    await service.delete_key(user=current_identity, key_id=key_id)


@org_router.post("/{key_id}/default")
async def set_org_default_web_search_key(
    service: WebSearchKeyServiceDep, current_identity: CurrentIdentity, key_id: uuid.UUID
) -> OrgWebSearchKeyPublic:
    """Make a key the organization's default of its kind, in place of the previous one.

    Organization owners and admins only.
    """
    return await service.set_default(user=current_identity, key_id=key_id)


@workspace_router.get("")
async def list_workspace_web_search_keys(
    service: WebSearchKeyServiceDep,
    current_identity: CurrentIdentity,
    workspace_id: uuid.UUID,
    skip: Annotated[int, Query(ge=0, description="Number of records to skip")] = 0,
    limit: Annotated[int, Query(ge=1, le=1000, description="Maximum number of records to return")] = 100,
) -> WorkspaceWebSearchKeysPublic:
    """The organization's keys as this workspace sees them, and which of each kind it uses.

    Any member of the workspace may read it. ``is_effective`` is decided across every key, not only this page,
    and ``search_provider`` and ``fetch_provider`` say what the workspace's searches and fetches run on.
    """
    return await service.list_workspace_keys(user=current_identity, workspace_id=workspace_id, skip=skip, limit=limit)


@workspace_router.patch("/{key_id}")
async def set_workspace_web_search_key_override(
    service: WebSearchKeyServiceDep,
    current_identity: CurrentIdentity,
    workspace_id: uuid.UUID,
    key_id: uuid.UUID,
    body: WorkspaceWebSearchKeyOverrideRequest,
) -> WorkspaceWebSearchKeysPublic:
    """Pin a key as this workspace's own for its kind, or turn it off for this workspace.

    Organization owners and admins, or this workspace's owners and admins. Answers with the list's first page.
    """
    return await service.set_workspace_override(
        user=current_identity, workspace_id=workspace_id, key_id=key_id, request=body
    )


@workspace_router.delete("/{key_id}")
async def reset_workspace_web_search_key_override(
    service: WebSearchKeyServiceDep, current_identity: CurrentIdentity, workspace_id: uuid.UUID, key_id: uuid.UUID
) -> WorkspaceWebSearchKeysPublic:
    """Return this workspace to inheriting the key. Idempotent. Answers with the list's first page."""
    return await service.reset_workspace_override(user=current_identity, workspace_id=workspace_id, key_id=key_id)
