"""The models.dev catalog as the gateway reads it: one typed copy of an ``api.json`` document.

models.dev (https://models.dev, MIT licensed) publishes what each provider serves: a
model's modalities, flags, limits, prices and dates. The gateway reads it in three
places, the metadata a listing shows, the price index and the stored price snapshot,
and all three read it through this module, so a change upstream is one change here.

It lives here, not in the catalog or the pricing service package, because both domains
read it and neither imports the other. It is standard library only:
``scripts/update_models_dev_snapshot.py`` loads this file by path, so the bundled
snapshot is cut with the same rules as every snapshot the gateway stores.
"""

import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

__all__ = [
    "LISTING_PREFIX",
    "RATE_FIELDS",
    "ModelsDevCatalog",
    "ModelsDevCost",
    "ModelsDevModel",
    "ModelsDevProvider",
    "ModelsDevRates",
    "ModelsDevTier",
    "models_dev_provider_id",
    "parse_catalog",
    "snapshot_document",
]

# Gemini's listing names a model ``models/<id>``, where models.dev keys the bare id.
LISTING_PREFIX = "models/"

# The rate fields of a ``cost`` block and of each tier in it, USD per million tokens.
RATE_FIELDS = ("input", "output", "cache_read", "cache_write")

# models.dev spells a handful of providers differently from any-llm. Looked up
# by the any-llm implementation name; a provider absent here is spelled the same
# in both, or is one models.dev does not carry (a local server, a proxy).
_MODELS_DEV_PROVIDER_IDS: dict[str, str] = {
    "fireworks": "fireworks-ai",
    "gemini": "google",
    "vertexai": "google-vertex",
    "vertexaianthropic": "google-vertex-anthropic",
    "bedrock": "amazon-bedrock",
    "together": "togetherai",
    "moonshot": "moonshotai",
    "azureopenai": "azure",
    "github": "github-copilot",
    "gmi": "gmicloud",
    "qiniu": "qiniu-ai",
}


def models_dev_provider_id(provider_type: str) -> str:
    """The models.dev provider id for an any-llm implementation."""
    return _MODELS_DEV_PROVIDER_IDS.get(provider_type, provider_type)


@dataclass(frozen=True, slots=True)
class ModelsDevRates:
    """Rates as the document spells them, ``None`` where it gives none.

    A number is kept as written, believable or not: whether it is a price is the
    price index's call. A rate that is present but not a number is read as
    ``None`` and named in ``malformed``, so the index can refuse the document
    rather than price the model as if the rate were missing.
    """

    input: float | None = None
    output: float | None = None
    cache_read: float | None = None
    cache_write: float | None = None
    malformed: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ModelsDevTier:
    """One entry of ``cost.tiers``: the rates that apply from ``size`` of what ``kind`` measures."""

    kind: str | None
    size: int | None
    rates: ModelsDevRates


@dataclass(frozen=True, slots=True)
class ModelsDevCost:
    """A model's ``cost`` block: its base rates, its tiers and the older ``context_over_200k`` block."""

    rates: ModelsDevRates
    tiers: tuple[ModelsDevTier, ...] = ()
    context_over_200k: ModelsDevRates | None = None


@dataclass(frozen=True, slots=True)
class ModelsDevModel:
    """One model as a provider serves it. ``id`` is the document's, or its key where the document gives none."""

    id: str
    name: str | None = None
    description: str | None = None
    family: str | None = None
    canonical_model_id: str | None = None
    """models.dev's provider-independent id, ``vendor/model``: who made the model."""
    status: str | None = None
    input_modalities: tuple[str, ...] = ()
    output_modalities: tuple[str, ...] = ()
    reasoning: bool = False
    tool_call: bool = False
    structured_output: bool = False
    attachment: bool = False
    temperature: bool = False
    open_weights: bool = False
    context_window: int | None = None
    max_output_tokens: int | None = None
    knowledge: str | None = None
    release_date: str | None = None
    last_updated: str | None = None
    cost: ModelsDevCost | None = None


@dataclass(frozen=True, slots=True)
class ModelsDevProvider:
    """One provider and its models, keyed as the document keys them."""

    id: str
    name: str | None
    doc: str | None
    models: Mapping[str, ModelsDevModel]


@dataclass(frozen=True, slots=True)
class ModelsDevCatalog:
    """One ``api.json`` document: its providers, keyed as the document keys them."""

    providers: Mapping[str, ModelsDevProvider]

    def models(self) -> Iterator[tuple[ModelsDevProvider, ModelsDevModel]]:
        """Every model with its provider, in document order."""
        for provider in self.providers.values():
            yield from ((provider, model) for model in provider.models.values())

    def provider_for(self, provider_type: str) -> ModelsDevProvider | None:
        """The provider an any-llm implementation name is listed under."""
        return self.providers.get(models_dev_provider_id(provider_type))

    def model_for(self, provider_type: str, model_id: str) -> ModelsDevModel | None:
        """The model ``model_id`` names under an any-llm implementation, by its listing id or its bare one."""
        provider = self.provider_for(provider_type)
        if provider is None:
            return None
        model = provider.models.get(model_id)
        if model is None and model_id.startswith(LISTING_PREFIX):
            model = provider.models.get(model_id.removeprefix(LISTING_PREFIX))
        return model


def _str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _number(value: object) -> float | None:
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _limit(value: object) -> int | None:
    """A token limit, or ``None`` for anything but a positive integer.

    models.dev writes 0 for a limit it does not know, and a client would read
    0 as no tokens at all.
    """
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def _tier_size(value: object) -> int | None:
    number = _number(value)
    if number is None or not math.isfinite(number) or number <= 0:
        return None
    return int(number) if int(number) == number else None


def _str_tuple(value: object) -> tuple[str, ...]:
    return tuple(item for item in value if isinstance(item, str)) if isinstance(value, list) else ()


def _rates(raw: object) -> ModelsDevRates:
    if not isinstance(raw, dict):
        return ModelsDevRates()
    values = {name: raw.get(name) for name in RATE_FIELDS}
    return ModelsDevRates(
        **{name: _number(value) for name, value in values.items()},
        malformed=tuple(name for name, value in values.items() if value is not None and _number(value) is None),
    )


def _tier(raw: dict[str, Any]) -> ModelsDevTier:
    kind = raw.get("tier")
    kind = kind if isinstance(kind, dict) else {}
    return ModelsDevTier(kind=_str(kind.get("type")), size=_tier_size(kind.get("size")), rates=_rates(raw))


def _cost(raw: object) -> ModelsDevCost | None:
    if not isinstance(raw, dict):
        return None
    raw_tiers = raw.get("tiers")
    tiers = tuple(_tier(tier) for tier in (raw_tiers if isinstance(raw_tiers, list) else ()) if isinstance(tier, dict))
    over = raw.get("context_over_200k")
    return ModelsDevCost(
        rates=_rates(raw), tiers=tiers, context_over_200k=_rates(over) if isinstance(over, dict) else None
    )


def _model(key: str, raw: dict[str, Any]) -> ModelsDevModel:
    modalities = raw.get("modalities")
    modalities = modalities if isinstance(modalities, dict) else {}
    limit = raw.get("limit")
    limit = limit if isinstance(limit, dict) else {}
    return ModelsDevModel(
        id=_str(raw.get("id")) or key,
        name=_str(raw.get("name")),
        description=_str(raw.get("description")),
        family=_str(raw.get("family")),
        canonical_model_id=_str(raw.get("canonical_model_id")),
        status=_str(raw.get("status")),
        input_modalities=_str_tuple(modalities.get("input")),
        output_modalities=_str_tuple(modalities.get("output")),
        reasoning=bool(raw.get("reasoning")),
        tool_call=bool(raw.get("tool_call")),
        structured_output=bool(raw.get("structured_output")),
        attachment=bool(raw.get("attachment")),
        temperature=bool(raw.get("temperature")),
        open_weights=bool(raw.get("open_weights")),
        context_window=_limit(limit.get("context")),
        max_output_tokens=_limit(limit.get("output")),
        knowledge=_str(raw.get("knowledge")),
        release_date=_str(raw.get("release_date")),
        last_updated=_str(raw.get("last_updated")),
        cost=_cost(raw.get("cost")),
    )


def parse_catalog(raw: object) -> ModelsDevCatalog:
    """Read a parsed ``api.json``.

    A provider without a ``models`` object, and a model that is not an object,
    are left out: the document is fetched JSON, so a shape it was not expected
    to have describes nothing rather than raising. A document that is not an
    object raises ``TypeError``.
    """
    if not isinstance(raw, Mapping):
        raise TypeError("models.dev catalog is not a JSON object")
    providers: dict[str, ModelsDevProvider] = {}
    for key, provider in raw.items():
        if not isinstance(provider, dict) or not isinstance(provider.get("models"), dict):
            continue
        models = {
            str(model_key): _model(str(model_key), model)
            for model_key, model in provider["models"].items()
            if isinstance(model, dict)
        }
        providers[str(key)] = ModelsDevProvider(
            id=_str(provider.get("id")) or str(key),
            name=_str(provider.get("name")),
            doc=_str(provider.get("doc")),
            models=MappingProxyType(models),
        )
    return ModelsDevCatalog(providers=MappingProxyType(providers))


def _rates_document(rates: ModelsDevRates) -> dict[str, Any]:
    return {name: value for name in RATE_FIELDS if (value := getattr(rates, name)) is not None}


def _tier_document(tier: ModelsDevTier) -> dict[str, Any]:
    return {name: value for name, value in (("type", tier.kind), ("size", tier.size)) if value is not None}


def _cost_document(cost: ModelsDevCost) -> dict[str, Any]:
    document = _rates_document(cost.rates)
    if cost.tiers:
        document["tiers"] = [{**_rates_document(tier.rates), "tier": _tier_document(tier)} for tier in cost.tiers]
    if cost.context_over_200k is not None:
        document["context_over_200k"] = _rates_document(cost.context_over_200k)
    return document


def _model_document(key: str, model: ModelsDevModel) -> dict[str, Any]:
    document: dict[str, Any] = {}
    if model.id != key:
        document["id"] = model.id
    for name in ("name", "canonical_model_id", "status"):
        if (value := getattr(model, name)) is not None:
            document[name] = value
    if model.cost is not None and (cost := _cost_document(model.cost)):
        document["cost"] = cost
    if model.context_window is not None:
        document["limit"] = {"context": model.context_window}
    return document


def snapshot_document(catalog: ModelsDevCatalog) -> dict[str, Any]:
    """The catalog as a stored price snapshot: each provider's id, name and doc, and each model's price fields.

    The snapshot keeps the document's own shape, so :func:`parse_catalog` reads
    it back, and a stored or bundled snapshot is read like a live download.
    """
    document: dict[str, Any] = {}
    for key, provider in catalog.providers.items():
        entry: dict[str, Any] = {"id": provider.id}
        for name in ("name", "doc"):
            if (value := getattr(provider, name)) is not None:
                entry[name] = value
        entry["models"] = {model_key: _model_document(model_key, model) for model_key, model in provider.models.items()}
        document[key] = entry
    return document
