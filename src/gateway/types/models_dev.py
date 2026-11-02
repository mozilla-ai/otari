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
    "MAX_ISSUE_EXAMPLES",
    "RATE_FIELDS",
    "REFUSE_DAMAGED_SHARE",
    "ModelsDevCatalog",
    "ModelsDevCost",
    "ModelsDevIssues",
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

# How many issue paths a catalog keeps to name in a log line; the count is kept in full.
MAX_ISSUE_EXAMPLES = 5

# A document is refused when more than this share of its models are damaged. On
# 2026-11-02 the largest provider held about 7% of the catalog's models, so one
# broken provider stays under it, while a format change upstream damages most.
REFUSE_DAMAGED_SHARE = 0.25

# A key longer than this is cut and quoted in an issue path.
_MAX_SEGMENT = 64

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
class ModelsDevIssues:
    """What :func:`parse_catalog` could not read.

    ``count`` is every value of the wrong type or shape, and ``examples`` the
    first few as ``path: expected <type>``. A path names keys and types only,
    never a value, because the document is fetched data. ``models`` counts the
    model entries the document lists, and ``damaged_models`` those left out or
    read with a field missing.
    """

    count: int = 0
    examples: tuple[str, ...] = ()
    models: int = 0
    damaged_models: int = 0

    def describe(self) -> str:
        """The issues for a log line: how many, in how many models, and the first few paths."""
        return "; ".join(
            (f"{self.count} unreadable values in {self.damaged_models} of {self.models} models", *self.examples)
        )


@dataclass(frozen=True, slots=True)
class ModelsDevCatalog:
    """One ``api.json`` document: its providers, keyed as the document keys them, and what could not be read."""

    providers: Mapping[str, ModelsDevProvider]
    issues: ModelsDevIssues = ModelsDevIssues()

    def refusal(self) -> str | None:
        """Why the document is not fit to serve, or ``None`` when it is.

        A document that lists no models, or one with more than
        ``REFUSE_DAMAGED_SHARE`` of its models damaged, has a shape the gateway
        cannot read; serving it would drop most models or most facts.
        """
        issues = self.issues
        if issues.models == 0:
            reason = "the document lists no models"
        elif issues.damaged_models > issues.models * REFUSE_DAMAGED_SHARE:
            reason = f"{issues.damaged_models} of {issues.models} models are damaged"
        else:
            return None
        return "; ".join((reason, *issues.examples))

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


def _number(value: object) -> float | None:
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _tier_size(value: object) -> int | None:
    number = _number(value)
    if number is None or number <= 0:
        return None
    if isinstance(number, int):
        # Checked before ``math.isfinite``, which raises on an int past float range.
        return number
    return int(number) if math.isfinite(number) and int(number) == number else None


class _Reader:
    """Reads one document and notes each value it cannot read, by path and the type it expected."""

    def __init__(self) -> None:
        self.count = 0
        self.examples: list[str] = []

    def note(self, path: str, expected: str) -> None:
        self.count += 1
        if len(self.examples) < MAX_ISSUE_EXAMPLES:
            self.examples.append(f"{path}: expected {expected}")

    def mapping(self, raw: dict[str, Any], name: str, path: str) -> dict[str, Any] | None:
        value = raw.get(name)
        if value is None or isinstance(value, dict):
            return value
        self.note(f"{path}.{name}", "an object")
        return None

    def text(self, raw: dict[str, Any], name: str, path: str) -> str | None:
        value = raw.get(name)
        if value is None or isinstance(value, str):
            return value or None
        self.note(f"{path}.{name}", "a string")
        return None

    def flag(self, raw: dict[str, Any], name: str, path: str) -> bool:
        value = raw.get(name)
        if value is None or isinstance(value, bool):
            return value is True
        self.note(f"{path}.{name}", "a boolean")
        return False

    def texts(self, raw: dict[str, Any], name: str, path: str) -> tuple[str, ...]:
        value = raw.get(name)
        if value is None:
            return ()
        items = tuple(item for item in value if isinstance(item, str)) if isinstance(value, list) else ()
        if not isinstance(value, list) or len(items) != len(value):
            self.note(f"{path}.{name}", "a list of strings")
        return items

    def limit(self, raw: dict[str, Any], name: str, path: str) -> int | None:
        """A token limit; models.dev writes 0 for one it does not know, which a client would read as no tokens."""
        value = raw.get(name)
        if value is None:
            return None
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value or None
        self.note(f"{path}.{name}", "a non-negative integer")
        return None

    def rates(self, raw: dict[str, Any], path: str) -> ModelsDevRates:
        values = {name: raw.get(name) for name in RATE_FIELDS}
        malformed = tuple(name for name, value in values.items() if value is not None and _number(value) is None)
        for name in malformed:
            self.note(f"{path}.{name}", "a number")
        return ModelsDevRates(**{name: _number(value) for name, value in values.items()}, malformed=malformed)

    def tier(self, raw: object, path: str) -> ModelsDevTier | None:
        if not isinstance(raw, dict):
            self.note(path, "an object")
            return None
        kind = self.mapping(raw, "tier", path) or {}
        size = kind.get("size")
        if size is not None and _tier_size(size) is None:
            self.note(f"{path}.tier.size", "a positive integer")
        return ModelsDevTier(
            kind=self.text(kind, "type", f"{path}.tier"), size=_tier_size(size), rates=self.rates(raw, path)
        )

    def cost(self, raw: dict[str, Any], path: str) -> ModelsDevCost | None:
        cost = self.mapping(raw, "cost", path)
        if cost is None:
            return None
        path = f"{path}.cost"
        raw_tiers = cost.get("tiers")
        if raw_tiers is not None and not isinstance(raw_tiers, list):
            self.note(f"{path}.tiers", "a list")
            raw_tiers = None
        tiers = [self.tier(tier, f"{path}.tiers[{n}]") for n, tier in enumerate(raw_tiers or ())]
        over = self.mapping(cost, "context_over_200k", path)
        return ModelsDevCost(
            rates=self.rates(cost, path),
            tiers=tuple(tier for tier in tiers if tier is not None),
            context_over_200k=self.rates(over, f"{path}.context_over_200k") if over is not None else None,
        )

    def model(self, key: str, raw: dict[str, Any], path: str) -> ModelsDevModel:
        modalities = self.mapping(raw, "modalities", path) or {}
        limit = self.mapping(raw, "limit", path) or {}
        return ModelsDevModel(
            id=self.text(raw, "id", path) or key,
            name=self.text(raw, "name", path),
            description=self.text(raw, "description", path),
            family=self.text(raw, "family", path),
            canonical_model_id=self.text(raw, "canonical_model_id", path),
            status=self.text(raw, "status", path),
            input_modalities=self.texts(modalities, "input", f"{path}.modalities"),
            output_modalities=self.texts(modalities, "output", f"{path}.modalities"),
            reasoning=self.flag(raw, "reasoning", path),
            tool_call=self.flag(raw, "tool_call", path),
            structured_output=self.flag(raw, "structured_output", path),
            attachment=self.flag(raw, "attachment", path),
            temperature=self.flag(raw, "temperature", path),
            open_weights=self.flag(raw, "open_weights", path),
            context_window=self.limit(limit, "context", f"{path}.limit"),
            max_output_tokens=self.limit(limit, "output", f"{path}.limit"),
            knowledge=self.text(raw, "knowledge", path),
            release_date=self.text(raw, "release_date", path),
            last_updated=self.text(raw, "last_updated", path),
            cost=self.cost(raw, path),
        )


def _segment(key: object) -> str:
    """A document key as a path segment. Keys are fetched data and paths are logged, so an odd one is quoted."""
    text = str(key)
    return text if text.isprintable() and len(text) <= _MAX_SEGMENT else repr(text[:_MAX_SEGMENT])


def parse_catalog(raw: object) -> ModelsDevCatalog:
    """Read a parsed ``api.json``, noting in ``issues`` what it could not read.

    A provider without a ``models`` object, and a model that is not an object,
    are left out, and a field of the wrong type is read as absent, so one bad
    entry costs only itself. Whether the document as a whole is fit to serve is
    :meth:`ModelsDevCatalog.refusal`. A document that is not an object raises
    ``TypeError``.
    """
    if not isinstance(raw, Mapping):
        raise TypeError("models.dev catalog is not a JSON object")
    reader = _Reader()
    seen = damaged = 0
    providers: dict[str, ModelsDevProvider] = {}
    for key, provider in raw.items():
        path = _segment(key)
        if not isinstance(provider, dict):
            reader.note(path, "an object")
            continue
        if not isinstance(provider.get("models"), dict):
            reader.note(f"{path}.models", "an object")
            continue
        models: dict[str, ModelsDevModel] = {}
        for model_key, model in provider["models"].items():
            seen += 1
            before = reader.count
            model_path = f"{path}.models.{_segment(model_key)}"
            if isinstance(model, dict):
                models[str(model_key)] = reader.model(str(model_key), model, model_path)
            else:
                reader.note(model_path, "an object")
            damaged += reader.count > before
        providers[str(key)] = ModelsDevProvider(
            id=reader.text(provider, "id", path) or str(key),
            name=reader.text(provider, "name", path),
            doc=reader.text(provider, "doc", path),
            models=MappingProxyType(models),
        )
    issues = ModelsDevIssues(count=reader.count, examples=tuple(reader.examples), models=seen, damaged_models=damaged)
    return ModelsDevCatalog(providers=MappingProxyType(providers), issues=issues)


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
