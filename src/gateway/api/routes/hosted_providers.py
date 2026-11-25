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
SDK inside this process and never over the wire.

Every route addresses a provider by its name rather than a row id, because a
hosted provider may also be declared in configuration with no row of its own,
and its roster still has to be reachable.
"""

import uuid
from typing import Annotated

from fastapi import APIRouter, Body, Depends, Query, status

from gateway.api.deps import (
    HostedCatalogUpdateDep,
    HostedProviderServiceDep,
    SelectorReindexDep,
    require_deployment_operator_or_absent,
    verify_master_key,
)
from gateway.api.routes.organizations import Message
from gateway.core.surface import Surface
from gateway.schemas.providers import (
    DeploymentProviderMigrationPublic,
    DeploymentProviderMigrationRequest,
    HostedAvailableModelsPublic,
    HostedCatalogUpdatePreviewPublic,
    HostedCatalogUpdatePreviewRequest,
    HostedCatalogUpdatePublic,
    HostedCatalogUpdateRequest,
    HostedModelCreateRequest,
    HostedModelPublic,
    HostedModelsPublic,
    HostedModelsRefreshPublic,
    HostedModelUpdateRequest,
    HostedPriceResetPublic,
    HostedPriceResetRequest,
    HostedProviderCreateRequest,
    HostedProviderPublic,
    HostedProviderRegionCreateRequest,
    HostedProviderRegionUpdateRequest,
    HostedProvidersPublic,
    HostedProvidersReencryptPublic,
    HostedProviderTestPublic,
    HostedProviderTestRequest,
    HostedProviderUpdateRequest,
    HostedVendorModelsSwitchPublic,
    HostedVendorModelsSwitchRequest,
    HostedVendorPublic,
    HostedVendorUpdateRequest,
)
from gateway.types.model_type import OfferedModelType

router = APIRouter(
    prefix="/hosted-providers",
    tags=["hosted-providers"],
    dependencies=[Depends(verify_master_key), Depends(require_deployment_operator_or_absent)],
)

# The deployment serves its own hosted inference on these. The page is operator-only.
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
    service: HostedProviderServiceDep, reindex: SelectorReindexDep, body: HostedProviderCreateRequest
) -> HostedProviderPublic:
    """Configure a provider with the key this deployment will serve it on.

    Everything the provider lists on that key is offered at once; a provider
    that will not say yields a provider with no models, not an error. A
    provider the configuration defines may be created too, and the stored one
    then serves instead.
    """
    created = await service.create_provider(body)
    await reindex()
    return created


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

    Part of the ``OTARI_SECRET_KEY`` rotation procedure; run it alongside
    ``POST /api/v1/search-tools/reencrypt`` and
    ``POST /api/v1/organization-provider-keys/reencrypt``.
    A provider no configured key reads is left as it is, to be re-entered.
    """
    return await service.reencrypt()


@router.get("/migration")
async def preview_deployment_provider_migration(
    service: HostedProviderServiceDep,
) -> DeploymentProviderMigrationPublic:
    """What moving the deployment providers the retired dashboard page stored onto hosted providers would do.

    The gateway moves them at startup, so this lists the ones it could not
    move, each with the reason. Writes nothing and dials nothing.
    """
    return await service.migrate_deployment_providers(apply=False)


@router.post("/migration")
async def apply_deployment_provider_migration(
    service: HostedProviderServiceDep,
    reindex: SelectorReindexDep,
    # Defaulted rather than optional: a POST with no body moves every row that can move.
    body: Annotated[DeploymentProviderMigrationRequest, Body()] = _EVERY_ROW,
) -> DeploymentProviderMigrationPublic:
    """Move the deployment providers the retired dashboard page stored onto hosted providers of the same name.

    Each is created as a hosted provider that serves models not on its roster,
    offering what it lists as a create does, and the deployment provider is
    removed. One that cannot move is reported and left in place. ``rows``
    limits the move to the rows of the preview the caller reviewed, at the
    versions it was shown them.
    """
    reviewed = None if body.rows is None else {row.instance: row.updated_at for row in body.rows}
    # A move that fails leaves the moves before it committed, so they are indexed either way.
    try:
        return await service.migrate_deployment_providers(apply=True, reviewed=reviewed)
    finally:
        await reindex()


@router.delete("/migration/{instance}", status_code=status.HTTP_204_NO_CONTENT)
async def discard_deployment_provider(instance: str, service: HostedProviderServiceDep) -> None:
    """Remove a deployment provider the move could not take, which serves nothing."""
    await service.discard_deployment_provider(instance)


@router.post("/catalog-update/preview")
async def preview_catalog_update(
    catalog: HostedCatalogUpdateDep, body: HostedCatalogUpdatePreviewRequest
) -> HostedCatalogUpdatePreviewPublic:
    """What updating the catalog would do: the models.dev price changes, then what each provider newly lists.

    A POST, because it holds the fetched prices as pending for the apply to
    name by their ``digest``; nothing else is written. It dials every enabled
    provider. A price fetch that fails is reported in ``price_error`` and the
    model lists are still previewed. With ``cleanup``, it also previews which
    prices of models no provider serves would be removed.
    """
    return await catalog.preview(cleanup=body.cleanup)


@router.post("/catalog-update")
async def update_catalog(
    catalog: HostedCatalogUpdateDep, reindex: SelectorReindexDep, body: HostedCatalogUpdateRequest
) -> HostedCatalogUpdatePublic:
    """Apply the reviewed prices, then refresh every provider's models, then the cleanup if asked.

    The price step comes first, so the model step prices new models at the
    prices it applied, and a model held for want of a price is served once it
    has one. Without a ``digest`` no price changes. A ``digest`` that no longer
    names the pending prices answers 409 and changes nothing. ``cleanup``
    removes the prices of models no provider serves, with their history.
    """
    # The approved prices commit before the sweep, so they are indexed even when the cleanup fails.
    try:
        return await catalog.apply(digest=body.digest, cleanup=body.cleanup)
    finally:
        await reindex()


@router.patch("/{provider}")
async def update_hosted_provider(
    service: HostedProviderServiceDep, reindex: SelectorReindexDep, provider: str, body: HostedProviderUpdateRequest
) -> HostedProviderPublic:
    """Rotate the key, repoint the base, or turn a provider off.

    A provider defined in configuration answers 409: it is changed there.
    """
    updated = await service.update_provider(provider, body)
    await reindex()
    return updated


@router.delete("/{provider}")
async def delete_hosted_provider(
    service: HostedProviderServiceDep, reindex: SelectorReindexDep, provider: str
) -> Message:
    """Remove a provider and its roster, leaving the runtime with no hosted path for it.

    A provider defined in configuration answers 409. A stored provider that
    overrides a configured one keeps the roster, which the configured one
    serves again.
    """
    await service.delete_provider(provider)
    await reindex()
    return Message(message="Hosted provider deleted")


@router.post("/{provider}/regions", status_code=status.HTTP_201_CREATED)
async def add_hosted_provider_region(
    service: HostedProviderServiceDep, provider: str, body: HostedProviderRegionCreateRequest
) -> HostedProviderPublic:
    """Add a region the provider serves from, after its others, and answer the provider.

    The first region added keeps the provider's own connection as a region
    ahead of it, so what was served keeps being served. A request to the
    provider is served by the first enabled region whose countries admit it,
    and ``<provider>@<slug>:<model>`` pins one. A provider defined in
    configuration answers 409.
    """
    return await service.add_region(provider, body)


@router.patch("/{provider}/regions/{slug}")
async def update_hosted_provider_region(
    service: HostedProviderServiceDep, provider: str, slug: str, body: HostedProviderRegionUpdateRequest
) -> HostedProviderPublic:
    """Change, switch off or move one region, and answer the provider."""
    return await service.update_region(provider, slug, body)


@router.delete("/{provider}/regions/{slug}")
async def remove_hosted_provider_region(
    service: HostedProviderServiceDep, provider: str, slug: str
) -> HostedProviderPublic:
    """Remove one region, and answer the provider.

    Removing the last region leaves the provider serving from its own connection.
    """
    return await service.remove_region(provider, slug)


@router.get("/{provider}/models")
async def list_hosted_models(
    service: HostedProviderServiceDep,
    provider: str,
    skip: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=1000)] = 500,
    model_type: Annotated[
        OfferedModelType | None,
        Query(alias="type", description="Only models of this kind; `unknown` for those no source names a kind for"),
    ] = None,
    search: Annotated[
        str | None,
        Query(max_length=200, description="Only models whose name contains this text, case-insensitively"),
    ] = None,
    enabled: Annotated[
        bool | None,
        Query(
            description="Only models served (`true`) or only models withheld (`false`), "
            "a model of a vendor switched off counting as withheld"
        ),
    ] = None,
    vendor: Annotated[
        str | None,
        Query(max_length=128, description="Only models made by this vendor, `Other` for those nothing names one for"),
    ] = None,
) -> HostedModelsPublic:
    """List the models offered on one provider, with the price each currently serves at.

    ``search`` matches a model's name or its vendor. ``vendors`` counts each vendor's models
    across the whole provider, whatever the list was narrowed to, with the vendor's own switch.
    """
    return await service.list_models(
        provider, skip=skip, limit=limit, model_type=model_type, search=search, enabled=enabled, vendor=vendor
    )


@router.patch("/{provider}/vendors")
async def update_hosted_vendor(
    service: HostedProviderServiceDep, reindex: SelectorReindexDep, provider: str, body: HostedVendorUpdateRequest
) -> HostedVendorPublic:
    """Switch one vendor on or off on a provider.

    Off withholds every model of the vendor, including models a later refresh offers, and keeps
    each model's own switch, which applies again once the vendor is switched back on.
    """
    updated = await service.set_vendor(provider, body)
    await reindex()
    return updated


@router.post("/{provider}/models/switch")
async def switch_hosted_vendor_models(
    service: HostedProviderServiceDep, reindex: SelectorReindexDep, provider: str, body: HostedVendorModelsSwitchRequest
) -> HostedVendorModelsSwitchPublic:
    """Switch every model of one vendor on or off at once.

    Switching on serves only the models that have a price; the rest stay off and are counted
    as skipped. ``only_unpriced`` with switching off withholds only the models served without a price.
    """
    switched = await service.switch_vendor_models(provider, body)
    await reindex()
    return switched


@router.post("/{provider}/models/reset-prices")
async def reset_hosted_model_prices(
    service: HostedProviderServiceDep, reindex: SelectorReindexDep, provider: str, body: HostedPriceResetRequest
) -> HostedPriceResetPublic:
    """Put models priced by an operator back on the default price, or preview it with ``dry_run``.

    The default is models.dev's price, or the one the provider's own listing names. A model the
    default does not price keeps its price. Requests already billed keep the cost they were billed at.
    """
    reset = await service.reset_prices(provider, body)
    if not body.dry_run:
        await reindex()
    return reset


@router.post("/{provider}/models", status_code=status.HTTP_201_CREATED)
async def add_hosted_model(
    service: HostedProviderServiceDep, reindex: SelectorReindexDep, provider: str, body: HostedModelCreateRequest
) -> HostedModelPublic:
    """Offer a model on a provider, optionally at a custom price."""
    added = await service.add_model(provider, body)
    await reindex()
    return added


@router.post("/{provider}/models/refresh")
async def refresh_hosted_models(
    service: HostedProviderServiceDep, reindex: SelectorReindexDep, provider: str
) -> HostedModelsRefreshPublic:
    """Ask the provider again and offer whatever is newly listed.

    Nothing already offered is removed, even when the provider no longer lists
    it. New models follow the offer rule, seeded with the community default
    price and off when nothing prices them. A seeded price nobody changed moves
    with the default. An offered model whose stored price has since been
    deleted is seeded again, or switched off where the default does not price
    it, even one an operator switched on: a refresh never switches a model on.
    """
    refreshed = await service.refresh_models(provider)
    await reindex()
    return refreshed


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
    service: HostedProviderServiceDep,
    reindex: SelectorReindexDep,
    provider: str,
    model_id: uuid.UUID,
    body: HostedModelUpdateRequest,
) -> HostedModelPublic:
    """Reprice one offered model, toggle whether it is served, or both."""
    updated = await service.update_model(provider, model_id, body)
    await reindex()
    return updated


@router.delete("/{provider}/models/{model_id}")
async def remove_hosted_model(
    service: HostedProviderServiceDep, reindex: SelectorReindexDep, provider: str, model_id: uuid.UUID
) -> Message:
    """Stop offering a model on a provider. Its pricing history stays."""
    await service.remove_model(provider, model_id)
    await reindex()
    return Message(message="Hosted model removed")
