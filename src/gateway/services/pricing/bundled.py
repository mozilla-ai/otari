"""The models.dev snapshot packaged with the gateway, the offline price fallback."""

import json
from datetime import UTC, datetime
from functools import cache
from importlib import resources

from gateway.services.pricing.models_dev_index import ModelsDevPriceIndex, PriceGeneration

BUNDLED_SNAPSHOT_NAME = "models_dev_pricing.json"
_META_KEY = "_meta"


@cache
def _load() -> tuple[datetime, ModelsDevPriceIndex]:
    document = json.loads(resources.files("gateway").joinpath("data", BUNDLED_SNAPSHOT_NAME).read_text("utf-8"))
    meta = document.pop(_META_KEY, {})
    generated_at = datetime.fromisoformat(meta["generated_at"]).replace(tzinfo=UTC)
    return generated_at, ModelsDevPriceIndex.from_catalog(document)


def load_bundled_index() -> ModelsDevPriceIndex:
    """The packaged snapshot's index, built once per process."""
    return _load()[1]


def bundled_generation() -> PriceGeneration:
    """The packaged snapshot as the baseline generation, effective from the day it was generated."""
    generated_at, index = _load()
    return PriceGeneration(effective_at=generated_at, index=index, baseline=True)
