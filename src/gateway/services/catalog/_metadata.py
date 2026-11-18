"""models.dev's description of a catalog model, and the catalog copy a listing reads."""

from gateway.core.config import GatewayConfig
from gateway.services.model_catalog_service import (
    ModelCatalogEntry,
    background_catalog_enabled,
    cached_models_dev_catalog,
    load_models_dev_catalog,
)
from gateway.types.models_dev import ModelsDevCatalog


async def load_catalog_metadata(config: GatewayConfig, *, cached_only: bool = False) -> ModelsDevCatalog | None:
    """The models.dev catalog a model listing joins, or ``None`` without one.

    Answers from the cache whenever the refresher keeps it warm, so a listing
    waits on models.dev only before the first fetch has finished. ``cached_only``
    never fetches, for a caller off the request path; see
    :func:`~gateway.services.model_catalog_service.cached_models_dev_catalog`.
    """
    if cached_only:
        return cached_models_dev_catalog()
    return await load_models_dev_catalog(config, serve_stale=background_catalog_enabled(config))


def metadata_entry(catalog: ModelsDevCatalog | None, provider_type: str, model_id: str) -> ModelCatalogEntry | None:
    """models.dev's entry for a model under a provider type, or None.

    Looked up directly rather than through ``build_metadata_map`` so an
    organization's key, which is not a ``providers:`` instance, is enriched too.
    """
    if catalog is None:
        return None
    model = catalog.model_for(provider_type, model_id)
    return None if model is None else ModelCatalogEntry.from_model(model)
