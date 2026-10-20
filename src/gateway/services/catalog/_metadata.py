"""models.dev's description of a catalog model, and the catalog copy a listing reads."""

from typing import Any

from gateway.core.config import GatewayConfig
from gateway.services.model_catalog_service import (
    ModelCatalogEntry,
    background_catalog_enabled,
    cached_models_dev_catalog,
    load_models_dev_catalog,
    models_dev_provider_id,
    parse_entry,
)

# Gemini's listing names a model ``models/<id>``, where models.dev keys the bare id.
_LISTING_PREFIX = "models/"


async def load_catalog_metadata(config: GatewayConfig, *, cached_only: bool = False) -> dict[str, Any] | None:
    """The models.dev catalog a model listing joins, or ``None`` without one.

    Answers from the cache whenever the refresher keeps it warm, so a listing
    waits on models.dev only before the first fetch has finished. ``cached_only``
    never fetches, for a caller off the request path; see
    :func:`~gateway.services.model_catalog_service.cached_models_dev_catalog`.
    """
    if cached_only:
        return cached_models_dev_catalog(config)
    return await load_models_dev_catalog(config, serve_stale=background_catalog_enabled(config))


def metadata_entry(catalog: dict[str, Any] | None, provider_type: str, model_id: str) -> ModelCatalogEntry | None:
    """models.dev's entry for a model under a provider type, or None.

    Looked up directly rather than through ``build_metadata_map`` so an
    organization's key, which is not a ``providers:`` instance, is enriched too.
    """
    if not catalog:
        return None
    provider = catalog.get(models_dev_provider_id(provider_type))
    if not isinstance(provider, dict):
        return None
    models = provider.get("models")
    if not isinstance(models, dict):
        return None
    model = models.get(model_id)
    if model is None and model_id.startswith(_LISTING_PREFIX):
        model = models.get(model_id.removeprefix(_LISTING_PREFIX))
    return parse_entry(model) if isinstance(model, dict) else None
