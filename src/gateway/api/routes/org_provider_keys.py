"""Organization-scoped provider keys (standalone mode only).

Thin composition over `gateway.services.tenancy.org_provider_key_service`.
Two routers, because the surface has two scopes: an organization's own keys
(created, archived, defaulted) under ``/api/v1/organizations/me/provider-keys`,
matching `organizations.py`'s ``/me`` convention, and one workspace's view of
those keys (override, model-restrict) under
``/api/v1/workspaces/{workspace_id}/provider-keys``, matching `workspaces.py`'s
path-scoped convention. A caller manages several workspaces in one
organization, so the workspace surface takes ``workspace_id`` as a path
parameter the way `workspaces.py` does, unlike the organization surface's
``/me``.
"""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import (
    CurrentIdentity,
    OrgProviderModelServiceDep,
    get_db,
    verify_master_key,
)
from gateway.api.routes.organizations import Message
from gateway.core.surface import Surface
from gateway.schemas.providers import (
    OrgProviderAvailableModelsPublic,
    OrgProviderKeyCreateRequest,
    OrgProviderKeyModelCreateRequest,
    OrgProviderKeyModelPublic,
    OrgProviderKeyModelsPublic,
    OrgProviderKeyModelUpdateRequest,
    OrgProviderKeyPublic,
    OrgProviderKeysPublic,
    OrgProviderKeyUpdateRequest,
    OrgProviderModelsRefreshPublic,
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
    prefix="/organizations/me/provider-keys",
    tags=["provider-keys"],
    dependencies=[Depends(verify_master_key)],
)

# Published by both topologies, because the page behind it is where an
# organization's models are offered, priced and switched, and that is a tenant's
# question whether or not the deployment has more than one tenant. ``providers``
# stays standalone-only beside it, and the two are disjoint mechanisms (see
# ``models/provider_keys.py``), so neither stands in for the other. Not named
# after its prefix, since ``organizations`` is already a surface.
SURFACE = Surface("organization_providers")

workspace_router = APIRouter(
    prefix="/workspaces/{workspace_id}/provider-keys",
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
    """List the caller's organization's provider keys. Organization owners and admins only."""
    return await service.list_keys_for_user(
        user=current_identity, include_archived=include_archived, skip=skip, limit=limit
    )


@org_router.post("", status_code=status.HTTP_201_CREATED)
async def create_org_provider_key(
    models: OrgProviderModelServiceDep,
    current_identity: CurrentIdentity,
    body: OrgProviderKeyCreateRequest,
) -> OrgProviderKeyPublic:
    """Create a provider key in the caller's organization. Organization owners and admins only.

    Everything the provider lists on the new credential is offered at once, so a
    key starts with its real catalog rather than an empty list an admin retypes
    by hand. A provider that will not say (no listing endpoint, unreachable,
    credential refused) yields a key with no models rather than a failed create:
    the credential may still be right for dispatch, and models can be added by
    name. The response is the key either way; the models are read back through
    ``GET /{key_id}/models``.
    """
    return await models.add_provider_key(user=current_identity, request=body)


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
# Offered models
# ==============================================================================
#
# The model half of a provider key: what the credential reaches, what each model
# costs this organization, and whether the runtime serves it.
#
# Every failure a *provider* can hand back is reported in the body rather than as
# a status: an unreachable upstream, a credential the provider refused, a backend
# with no model-listing endpoint, and a stored key this deployment can no longer
# decrypt are all facts about the provider or the deployment rather than about
# the request, and the page renders each beside a list that is still standing.
# A 4xx here means the *request* was wrong: no such key, no such model, a name
# already offered.


@org_router.get("/{key_id}/models")
async def list_org_provider_key_models(
    service: OrgProviderModelServiceDep,
    current_identity: CurrentIdentity,
    key_id: uuid.UUID,
    skip: Annotated[int, Query(ge=0, description="Number of records to skip")] = 0,
    limit: Annotated[int, Query(ge=1, le=1000, description="Maximum number of records to return")] = 500,
) -> OrgProviderKeyModelsPublic:
    """List the models offered on one key, each with the rate it currently serves at.

    Organization owners and admins only. ``count`` is the total rather than the
    page length, so a client knows whether another page is owed.
    """
    return await service.list_models(user=current_identity, key_id=key_id, skip=skip, limit=limit)


@org_router.post("/{key_id}/models", status_code=status.HTTP_201_CREATED)
async def add_org_provider_key_model(
    service: OrgProviderModelServiceDep,
    current_identity: CurrentIdentity,
    key_id: uuid.UUID,
    body: OrgProviderKeyModelCreateRequest,
) -> OrgProviderKeyModelPublic:
    """Offer one model by name, for a backend whose models cannot be listed.

    Carries no rate: an organization's rates are written through
    ``/api/v1/organizations/me/pricing``, so a price set here and a price set
    there could not disagree about what a request costs. The offer seeds the
    community default like any other, and a model nothing prices arrives
    disabled. Organization owners and admins only.
    """
    return await service.add_model(user=current_identity, key_id=key_id, model=body.model)


@org_router.patch("/{key_id}/models/{model_id}")
async def set_org_provider_key_model_enabled(
    service: OrgProviderModelServiceDep,
    current_identity: CurrentIdentity,
    key_id: uuid.UUID,
    model_id: uuid.UUID,
    body: OrgProviderKeyModelUpdateRequest,
) -> OrgProviderKeyModelPublic:
    """Turn one offered model's serving switch on or off. Organization owners and admins only."""
    return await service.set_model_enabled(
        user=current_identity, key_id=key_id, model_id=model_id, enabled=body.enabled
    )


@org_router.delete("/{key_id}/models/{model_id}")
async def remove_org_provider_key_model(
    service: OrgProviderModelServiceDep,
    current_identity: CurrentIdentity,
    key_id: uuid.UUID,
    model_id: uuid.UUID,
) -> Message:
    """Stop offering one model. Its rate and its history stay. Organization owners and admins only."""
    await service.remove_model(user=current_identity, key_id=key_id, model_id=model_id)
    return Message(message="Model no longer offered")


@org_router.post("/{key_id}/models/refresh")
async def refresh_org_provider_key_models(
    service: OrgProviderModelServiceDep,
    current_identity: CurrentIdentity,
    key_id: uuid.UUID,
) -> OrgProviderModelsRefreshPublic:
    """Ask the provider again and offer whatever is newly listed.

    Additive only: nothing already offered is removed or switched off, because
    delisting a model is a decision the serving switch owns and an upstream
    hiccup must not empty a catalog. New models follow the offer rule, seeded
    with the community default rate and disabled when nothing prices them. A
    rate this surface seeded and nobody has changed moves to today's default.
    Organization owners and admins only.
    """
    return await service.refresh_models(
        user=current_identity, key_id=key_id
    )


@org_router.post("/{key_id}/pricing/refresh")
async def refresh_org_provider_key_model_pricing(
    service: OrgProviderModelServiceDep,
    current_identity: CurrentIdentity,
    key_id: uuid.UUID,
) -> OrgProviderModelsRefreshPublic:
    """Move every rate this surface seeded onto today's community default.

    The other half of the refresh above, without the dial: re-reading community
    rates is cheap and asking a provider for its whole catalog is not, so an
    admin who only wants the price move does not wait on an upstream. A rate an
    admin has since set is left alone, and a model that arrived unpriced is
    offered a rate and switched on if one has appeared. Organization owners and
    admins only.
    """
    return await service.refresh_pricing(user=current_identity, key_id=key_id)


@org_router.get("/{key_id}/available-models")
async def list_org_provider_key_available_models(
    service: OrgProviderModelServiceDep,
    current_identity: CurrentIdentity,
    key_id: uuid.UUID,
) -> OrgProviderAvailableModelsPublic:
    """Ask the provider what it serves on this key's stored credential.

    Dials the upstream on every call rather than caching: the caller is a model
    picker, opened rarely and entitled to a current answer. The credential never
    leaves the process; only model names come back. Organization owners and
    admins only.
    """
    return await service.available_models(
        user=current_identity, key_id=key_id
    )


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
