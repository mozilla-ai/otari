"""Who makes each model a hosted provider offers, and the per-vendor switch that withholds them.

A provider that routes to others (Together, OpenRouter, another Otari) serves
models from many vendors, so an operator decides about them a vendor at a time.
The vendor is read the way the catalog reads it: models.dev's canonical id
first, then the org segment of the id, then the model family. After those, the
org segment as the id spells it, then the vendor whose own provider this is. A
model none of those names is listed under :data:`OTHER_VENDOR`.

The switch is stored per provider and vendor, as a row naming a vendor that
provider does not serve. It sits above each model's own switch, so turning a
vendor back on restores every model to the setting it had.
"""

from collections.abc import Iterable

from gateway.services.model_catalog_service import cached_models_dev_catalog
from gateway.services.model_identity import identity_key, infer_vendor, vendor_publishing_on
from gateway.services.providers._hosted_offers import OTARI_IMPLEMENTATION

OTHER_VENDOR = "Other"
"""Where a model whose vendor nothing names is listed."""


def vendor_of(implementation: str, model: str) -> str:
    """The vendor of ``model`` as the any-llm ``implementation`` offers it, :data:`OTHER_VENDOR` when unknown.

    Another Otari names a model ``vendor/model`` under models.dev's own
    provider key for that vendor, so that is where its entry is read.
    """
    catalog = cached_models_dev_catalog()
    provider_type, model_id = implementation, model
    if implementation == OTARI_IMPLEMENTATION:
        vendor, separator, name = model.partition("/")
        if separator:
            provider_type, model_id = vendor, name
    entry = catalog.model_for(provider_type, model_id) if catalog is not None else None
    display = entry.name if entry is not None else None
    canonical = entry.canonical_model_id if entry is not None else None
    key = identity_key(provider_type, model_id, display)
    inferred = infer_vendor(model, key, canonical)
    if inferred:
        return inferred
    org, separator, _ = model.partition("/")
    if separator and org:
        return org
    return vendor_publishing_on(provider_type) or OTHER_VENDOR


def vendors_of(implementation: str, models: Iterable[str]) -> dict[str, str]:
    """The vendor of each model, keyed by model name."""
    return {model: vendor_of(implementation, model) for model in models}
