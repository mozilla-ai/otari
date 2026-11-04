"""The models.dev catalog packaged with the gateway: the offline price fallback, and the facts before a download."""

import json
from datetime import UTC, datetime
from functools import cache
from importlib import resources

from gateway.services.pricing.models_dev_index import ModelsDevPriceIndex, PriceGeneration
from gateway.types.models_dev import ModelsDevCatalog, read_catalog_document

BUNDLED_SNAPSHOT_NAME = "models_dev_catalog.json"
_META_KEY = "_meta"


@cache
def _load() -> tuple[datetime, ModelsDevCatalog]:
    document = json.loads(resources.files("gateway").joinpath("data", BUNDLED_SNAPSHOT_NAME).read_text("utf-8"))
    generated_at = datetime.fromisoformat(document[_META_KEY]["generated_at"]).replace(tzinfo=UTC)
    return generated_at, read_catalog_document(document)


@cache
def load_bundled_index() -> ModelsDevPriceIndex:
    """The packaged catalog's price index, built once per process."""
    return ModelsDevPriceIndex.from_catalog(_load()[1])


def bundled_catalog() -> ModelsDevCatalog:
    """The packaged catalog, read once per process."""
    return _load()[1]


def bundled_generation() -> PriceGeneration:
    """The packaged catalog's prices as the baseline generation, effective from the day it was generated."""
    return PriceGeneration(effective_at=_load()[0], index=load_bundled_index(), baseline=True)
