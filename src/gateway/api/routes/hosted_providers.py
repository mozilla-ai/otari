"""The hosted-providers operator API.

The deployment's own upstream credentials: which providers it serves hosted
inference on, the keys it serves them with, and the models it offers on each
with the rate the deployment charges for them.

Two gates on the router, and they answer different questions.
``verify_master_key`` asks whether the caller authenticated; the operator gate
asks whether that caller may act deployment-wide, and refuses everyone else
with the 404 ``/api/v1/admin`` answers, because a surface holding the
deployment's own credentials is one worth not confirming to a signed-in member.
Declared on the router so a route added later inherits both.

**This surface never returns a credential.** Not on create, not on read, not
once. A hosted provider is named by its provider and the last four characters
of its key, and that is all a client ever learns about it. The key is read in
one place, the ``ModelProviderPort`` adapter, which hands it to the upstream
SDK inside this process and never over the wire. There is deliberately no
resolve endpoint here.

Every route addresses a provider by its name rather than a row id, because a
hosted provider may also be declared in configuration with no row of its own,
and its roster still has to be reachable.
"""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Query, status

from gateway.api.deps import HostedProviderServiceDep, require_deployment_operator_or_absent, verify_master_key
from gateway.api.routes.organizations import Message
from gateway.core.surface import Surface
from gateway.schemas.providers import (
    HostedAvailableModelsPublic,
    HostedCatalogRefreshPublic,
    HostedModelCreateRequest,
    HostedModelPublic,
    HostedModelsPublic,
    HostedModelsRefreshPublic,
    HostedModelUpdateRequest,
    HostedProviderCreateRequest,
    HostedProviderPublic,
    HostedProvidersPublic,
    HostedProviderUpdateRequest,
)

router = APIRouter(
    prefix="/hosted-providers",
    tags=["hosted-providers"],
    dependencies=[Depends(verify_master_key), Depends(require_deployment_operator_or_absent)],
)

# Published by both topologies: a standalone deployment serves its own hosted
# inference on these, and a hosted control plane holds them for the data plane
# that resolves against it. The page is operator-only on either.
SURFACE = Surface("hosted_providers")


@router.get("")
async def list_hosted_providers(
    service: HostedProviderServiceDep,
    skip: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
) -> HostedProvidersPublic:
    """List the providers this deployment serves hosted inference on."""
    return await service.list_providers(skip=skip, limit=limit)


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_hosted_provider(
    service: HostedProviderServiceDep, body: HostedProviderCreateRequest
) -> HostedProviderPublic:
    """Configure a provider with the key this deployment will serve it on.

    Everything the provider lists on that key is offered at once; a provider
    that will not say yields a provider with no models, not an error.
    """
    return await service.create_provider(body)


@router.get("/refresh/preview")
async def preview_catalog_refresh(service: HostedProviderServiceDep) -> HostedCatalogRefreshPublic:
    """What a catalog sweep would remove, without removing it.

    A GET, and a truthful one: it writes nothing. It does dial every enabled
    provider, as ``available-models`` does, because a model the provider still
    lists would survive the sweep and previewing it as doomed would be a lie.
    """
    return await service.refresh_catalog(apply=False)


@router.post("/refresh")
async def refresh_catalog(service: HostedProviderServiceDep) -> HostedCatalogRefreshPublic:
    """Re-ask every enabled provider what it lists, then take the rest off the catalog.

    The whole deployment rather than one provider: a stored price is what puts a
    model on a tenant's Models page, and a price under a provider nothing here
    serves cannot be reached by a per-provider refresh. Declared before the
    ``/{provider}`` paths to read in the order the surface is used; no method
    collides in any case.
    """
    return await service.refresh_catalog(apply=True)


@router.patch("/{provider}")
async def update_hosted_provider(
    service: HostedProviderServiceDep, provider: str, body: HostedProviderUpdateRequest
) -> HostedProviderPublic:
    """Rotate the key, repoint the base, or turn a provider off."""
    return await service.update_provider(provider, body)


@router.delete("/{provider}")
async def delete_hosted_provider(service: HostedProviderServiceDep, provider: str) -> Message:
    """Remove a provider and its roster, leaving the runtime with no hosted path for it."""
    await service.delete_provider(provider)
    return Message(message="Hosted provider deleted")


@router.get("/{provider}/models")
async def list_hosted_models(
    service: HostedProviderServiceDep,
    provider: str,
    skip: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=1000)] = 500,
) -> HostedModelsPublic:
    """List the models offered on one provider, with the price each currently serves at."""
    return await service.list_models(provider, skip=skip, limit=limit)


@router.post("/{provider}/models", status_code=status.HTTP_201_CREATED)
async def add_hosted_model(
    service: HostedProviderServiceDep, provider: str, body: HostedModelCreateRequest
) -> HostedModelPublic:
    """Offer a model on a provider, optionally at a custom price."""
    return await service.add_model(provider, body)


@router.post("/{provider}/models/refresh")
async def refresh_hosted_models(service: HostedProviderServiceDep, provider: str) -> HostedModelsRefreshPublic:
    """Ask the provider again and offer whatever is newly listed.

    Additive only: nothing already offered is removed or toggled. New models
    follow the offer rule, seeded with the community default price and off
    when nothing prices them. A seeded price nobody changed moves with the
    default.
    """
    return await service.refresh_models(provider)


@router.get("/{provider}/available-models")
async def list_available_models(service: HostedProviderServiceDep, provider: str) -> HostedAvailableModelsPublic:
    """Ask the provider what it serves on the stored credential.

    Dials the upstream on every call rather than caching: the caller is the
    admin form's model picker, opened rarely and entitled to a current answer.
    Failure comes back in the body, so the picker can fall back to a text box.
    """
    return await service.available_models(provider)


@router.patch("/{provider}/models/{model_id}")
async def update_hosted_model(
    service: HostedProviderServiceDep, provider: str, model_id: uuid.UUID, body: HostedModelUpdateRequest
) -> HostedModelPublic:
    """Reprice one offered model, toggle whether it is served, or both."""
    return await service.update_model(provider, model_id, body)


@router.delete("/{provider}/models/{model_id}")
async def remove_hosted_model(service: HostedProviderServiceDep, provider: str, model_id: uuid.UUID) -> Message:
    """Stop offering a model on a provider. Its pricing history stays."""
    await service.remove_model(provider, model_id)
    return Message(message="Hosted model removed")
