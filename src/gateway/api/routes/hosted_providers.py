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
once. A hosted provider is known by its name, its implementation and the last
four characters of its key, and that is all a client ever learns about it. The key is read in
one place, the ``ModelProviderPort`` adapter, which hands it to the upstream
SDK inside this process and never over the wire. There is deliberately no
resolve endpoint here.

Every route addresses a provider by its name rather than a row id, because a
hosted provider may also be declared in configuration with no row of its own,
and its roster still has to be reachable.
"""

import uuid
from typing import Annotated

from fastapi import APIRouter, Body, Depends, Query, status

from gateway.api.deps import HostedProviderServiceDep, require_deployment_operator_or_absent, verify_master_key
from gateway.api.routes.organizations import Message
from gateway.core.surface import Surface
from gateway.schemas.providers import (
    DeploymentProviderMigrationPublic,
    DeploymentProviderMigrationRequest,
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
    HostedProvidersReencryptPublic,
    HostedProviderTestPublic,
    HostedProviderTestRequest,
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

_EVERY_ROW = DeploymentProviderMigrationRequest()


@router.get("")
async def list_hosted_providers(
    service: HostedProviderServiceDep,
    skip: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
) -> HostedProvidersPublic:
    """List the providers this deployment serves hosted inference on.

    Stored providers and those defined in the configuration's
    ``hosted_providers:`` block alike, each marked with its ``source``.
    """
    return await service.list_providers(skip=skip, limit=limit)


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_hosted_provider(
    service: HostedProviderServiceDep, body: HostedProviderCreateRequest
) -> HostedProviderPublic:
    """Configure a provider with the key this deployment will serve it on.

    Everything the provider lists on that key is offered at once; a provider
    that will not say yields a provider with no models, not an error. A
    provider the configuration defines may be created too, and the stored one
    then serves instead.
    """
    return await service.create_provider(body)


@router.post("/test")
async def test_hosted_provider(
    service: HostedProviderServiceDep, body: HostedProviderTestRequest
) -> HostedProviderTestPublic:
    """Ask a provider what it lists on the given credentials, storing nothing.

    Backs the form's "Test connection". With ``name`` naming a stored provider,
    a field left out is read from it, so an edit can be tested without typing
    the key again; pointing the stored key at another origin still needs it.
    Failure comes back in the body.
    """
    return await service.test_credentials(body)


@router.post("/reencrypt")
async def reencrypt_hosted_providers(service: HostedProviderServiceDep) -> HostedProvidersReencryptPublic:
    """Re-encrypt every stored key and set of client arguments with the primary ``OTARI_SECRET_KEY``.

    Run after rotating the key, alongside ``POST /api/v1/provider-credentials/reencrypt``.
    A provider no configured key reads is left as it is, to be re-entered.
    """
    return await service.reencrypt()


@router.get("/migration")
async def preview_deployment_provider_migration(
    service: HostedProviderServiceDep,
) -> DeploymentProviderMigrationPublic:
    """What moving the deployment providers stored through the dashboard onto hosted providers would do.

    Writes nothing and dials nothing.
    """
    return await service.migrate_deployment_providers(apply=False)


@router.post("/migration")
async def apply_deployment_provider_migration(
    service: HostedProviderServiceDep,
    # Defaulted rather than optional: a POST with no body moves every row that can move.
    body: Annotated[DeploymentProviderMigrationRequest, Body()] = _EVERY_ROW,
) -> DeploymentProviderMigrationPublic:
    """Move the deployment providers stored through the dashboard onto hosted providers of the same name.

    Each is created as a hosted provider, offering what it lists as a create
    does, and the deployment provider is removed once the hosted one offers a
    model. One that cannot move is reported and left in place. ``rows``
    limits the move to the rows of the preview the caller reviewed, at the
    versions it was shown them.
    """
    reviewed = None if body.rows is None else {row.instance: row.updated_at for row in body.rows}
    return await service.migrate_deployment_providers(apply=True, reviewed=reviewed)


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
    serves cannot be reached by a per-provider refresh. A model list whose
    provider nothing defines any more, such as a removed ``config.yml`` entry,
    goes too, and is named in ``orphaned_rosters``. Where no provider offers a
    model, only those lists go and every price stays; with none of them either,
    the sweep answers 409. Declared before the
    ``/{provider}`` paths to read in the order the surface is used; no method
    collides in any case.
    """
    return await service.refresh_catalog(apply=True)


@router.patch("/{provider}")
async def update_hosted_provider(
    service: HostedProviderServiceDep, provider: str, body: HostedProviderUpdateRequest
) -> HostedProviderPublic:
    """Rotate the key, repoint the base, or turn a provider off.

    A provider defined in configuration answers 409: it is changed there.
    """
    return await service.update_provider(provider, body)


@router.delete("/{provider}")
async def delete_hosted_provider(service: HostedProviderServiceDep, provider: str) -> Message:
    """Remove a provider and its roster, leaving the runtime with no hosted path for it.

    A provider defined in configuration answers 409. A stored provider that
    overrides a configured one keeps the roster, which the configured one
    serves again.
    """
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

    Nothing already offered is removed, even when the provider no longer lists
    it. New models follow the offer rule, seeded with the community default
    price and off when nothing prices them. A seeded price nobody changed moves
    with the default. An offered model whose stored price has since been
    deleted is seeded again, or switched off where the default does not price
    it, even one an operator switched on: a refresh never switches a model on.
    """
    return await service.refresh_models(provider)


@router.get("/{provider}/available-models")
async def list_available_models(service: HostedProviderServiceDep, provider: str) -> HostedAvailableModelsPublic:
    """Ask the provider what it serves on its credential.

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
