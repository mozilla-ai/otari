"""The models.dev catalog as the gateway reads it: one typed copy of an ``api.json`` document.

models.dev (https://models.dev, MIT licensed) publishes what each provider serves: a
model's modalities, flags, limits, prices and dates. The gateway reads it in three
places, the metadata a listing shows, the price index and the stored price snapshot,
and all three read it through this module, so a change upstream is one change here.
The gateway also stores the whole typed catalog (:func:`catalog_document`), so a
restart or an offline deployment still has the facts.

models.dev publishes two documents. v2 is the full one, and v1 is a reduced copy
made from it. Both read into the same types: the v2 reader fills every field the
v1 reader fills, the way models.dev derives v1, and adds what only v2 carries.

It lives here, not in the catalog or the pricing service package, because both domains
read it and neither imports the other. It is standard library only:
``scripts/update_models_dev_snapshot.py`` loads this file by path, so the bundled
snapshot is cut with the same rules as every snapshot the gateway stores.
"""

import math
from collections.abc import Callable, Iterator, Mapping
from dataclasses import MISSING, dataclass, fields, is_dataclass
from decimal import Decimal
from types import MappingProxyType
from typing import Any

__all__ = [
    "CATALOG_FORMAT",
    "LISTING_PREFIX",
    "MAX_ISSUE_EXAMPLES",
    "MODELS_DEV_SOURCES",
    "MODELS_DEV_UNITS",
    "RATE_FIELDS",
    "REFUSE_DAMAGED_SHARE",
    "ModelsDevApi",
    "ModelsDevCachePrices",
    "ModelsDevCatalog",
    "ModelsDevCost",
    "ModelsDevIssues",
    "ModelsDevModalityCost",
    "ModelsDevModel",
    "ModelsDevMultiplier",
    "ModelsDevOverride",
    "ModelsDevPricing",
    "ModelsDevProvider",
    "ModelsDevRates",
    "ModelsDevSource",
    "ModelsDevTier",
    "ModelsDevUnitPrice",
    "catalog_document",
    "models_dev_provider_id",
    "models_dev_provider_ids",
    "parse_catalog",
    "parse_catalog_v2",
    "read_catalog_document",
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

# The version of the stored catalog format, in ``_meta.format``. A document
# without it is a price snapshot in the shape of a v1 ``api.json``.
CATALOG_FORMAT = 2
_META_KEY = "_meta"

# A key longer than this is cut and quoted in an issue path.
_MAX_SEGMENT = 64

# The rate fields of a ``cost`` block and of each tier in it, USD per million tokens.
RATE_FIELDS = ("input", "output", "cache_read", "cache_write")

# The units a v2 price can be in (``CostUnitV2Values`` in models.dev's
# ``packages/core/src/schema-v2.ts``). A price that names none is per ``token``.
MODELS_DEV_UNITS = ("token", "character", "image", "megapixel", "second", "minute")

# A price as a document gives it: a JSON number, read as ``float`` from models.dev and
# as ``Decimal`` from a stored rate, so money a row holds never rounds through a float.
type Number = float | Decimal

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


# A provider whose models models.dev moved under another one. v2 lists Vertex's
# Claude models under ``google-vertex``, where v1 and the snapshots cut from it
# keep ``google-vertex-anthropic``.
_MODELS_DEV_MOVED_PROVIDERS: dict[str, str] = {"google-vertex-anthropic": "google-vertex"}


def models_dev_provider_id(provider_type: str) -> str:
    """The models.dev provider id for an any-llm implementation."""
    return _MODELS_DEV_PROVIDER_IDS.get(provider_type, provider_type)


def models_dev_provider_ids(provider_type: str) -> tuple[str, ...]:
    """Every models.dev provider id an any-llm implementation's models may be under, in the order to try them."""
    first = models_dev_provider_id(provider_type)
    moved = _MODELS_DEV_MOVED_PROVIDERS.get(first)
    return (first,) if moved is None else (first, moved)


@dataclass(frozen=True, slots=True)
class ModelsDevRates:
    """Rates as the document spells them, ``None`` where it gives none.

    A number is kept as written, believable or not: whether it is a price is the
    price index's call. A rate that is present but not a number is read as
    ``None`` and named in ``malformed``, so the index can refuse the document
    rather than price the model as if the rate were missing.
    """

    input: Number | None = None
    output: Number | None = None
    cache_read: Number | None = None
    cache_write: Number | None = None
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
class ModelsDevCachePrices:
    """The cache prices of a v2 input price, in its unit.

    ``write`` is the one price v1 shows: the document's single price, or the
    first TTL it lists, which is the host's default. ``write_ttls`` keeps every
    TTL in document order.
    """

    read: Number | None = None
    write: Number | None = None
    write_ttls: tuple[tuple[str, Number], ...] = ()
    storage: Number | None = None


@dataclass(frozen=True, slots=True)
class ModelsDevOverride:
    """One entry of a v2 price's ``overrides``: the prices that replace the base ones ``when`` it holds.

    ``when`` is the condition tree as the document writes it, read only. The
    other fields are those of :class:`ModelsDevUnitPrice`.
    """

    when: Mapping[str, object]
    price: Number | None = None
    reasoning: Number | None = None
    cache: ModelsDevCachePrices | None = None
    explicit_cache: ModelsDevCachePrices | None = None
    malformed: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ModelsDevUnitPrice:
    """One v2 price: an input, an output or ``web_search``.

    ``unit`` is the document's, ``None`` where it gives none, which models.dev
    reads as ``token``. One not in ``MODELS_DEV_UNITS`` is kept and named in the
    catalog's issues, and ``gateway.types.price_unit`` says what one price buys.
    ``explicit_cache`` is the document's
    ``cache.explicit``: the prices of explicit caching where a host prices it
    apart from implicit caching. ``malformed`` names each field present but not
    a number, by its path (``price``, ``cache.read``, ``cache.explicit.write``).
    """

    price: Number | None
    unit: str | None = None
    reasoning: Number | None = None
    cache: ModelsDevCachePrices | None = None
    explicit_cache: ModelsDevCachePrices | None = None
    overrides: tuple[ModelsDevOverride, ...] = ()
    malformed: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ModelsDevModalityCost:
    """The input and output prices of one modality of a v2 ``cost`` block."""

    input: ModelsDevUnitPrice | None = None
    output: ModelsDevUnitPrice | None = None


@dataclass(frozen=True, slots=True)
class ModelsDevMultiplier:
    """One entry of a v2 ``cost.multipliers``.

    A number ``factor`` scales every price; an object scales only the prices it
    names, and is kept as the document writes it, read only.
    """

    when: Mapping[str, object]
    factor: Number | Mapping[str, object]


@dataclass(frozen=True, slots=True)
class ModelsDevPricing:
    """A model's full v2 ``cost`` block: each modality's prices, ``web_search`` and the multipliers."""

    modalities: Mapping[str, ModelsDevModalityCost]
    web_search: ModelsDevUnitPrice | None = None
    multipliers: tuple[ModelsDevMultiplier, ...] = ()


@dataclass(frozen=True, slots=True)
class ModelsDevApi:
    """One entry of a v2 model's ``api`` block: a protocol the host takes, and where.

    ``path`` is set only where the host does not use the protocol's usual path.
    """

    protocol: str
    base_url: str | None = None
    path: str | None = None


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
    """The per-token text prices, as v1 shows them."""
    type: str | None = "chat"
    """What the model does (``chat``, ``image``, ``embedding``...); a v2 model that gives none is ``chat``.

    ``None`` for a v1 model: v1 has no type, and calling every v1 model ``chat``
    would call its embedding models chat models too.
    """
    api: tuple[ModelsDevApi, ...] = ()
    """The protocols the host takes, in its order of preference. v2 only."""
    pricing: ModelsDevPricing | None = None
    """Every price the document gives, in its unit. v2 only."""


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
    read with a field missing. ``unknown_units`` names each v2 price unit the
    document uses that ``MODELS_DEV_UNITS`` does not, once, in document order.
    Such a price keeps its unit and prices nothing here, so it is not damage,
    and a unit added upstream never refuses the document.
    """

    count: int = 0
    examples: tuple[str, ...] = ()
    models: int = 0
    damaged_models: int = 0
    unknown_units: tuple[str, ...] = ()

    def describe(self) -> str:
        """The issues for a log line: how many, in how many models, the first few paths, and the units not known."""
        parts = [f"{self.count} unreadable values in {self.damaged_models} of {self.models} models", *self.examples]
        if self.unknown_units:
            # Fetched text, bounded and quoted the way a path segment is.
            parts.append(f"price units not known here: {', '.join(_segment(unit) for unit in self.unknown_units)}")
        return "; ".join(parts)


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
        ids = models_dev_provider_ids(provider_type)
        return next((self.providers[key] for key in ids if key in self.providers), None)

    def model_for(self, provider_type: str, model_id: str) -> ModelsDevModel | None:
        """The model ``model_id`` names under an any-llm implementation, by its listing id or its bare one."""
        names = (model_id, model_id.removeprefix(LISTING_PREFIX))
        for key in models_dev_provider_ids(provider_type):
            models = self.providers[key].models if key in self.providers else {}
            if (model := next((models[name] for name in names if name in models), None)) is not None:
                return model
        return None


def _number(value: object) -> Number | None:
    return value if isinstance(value, (int, float, Decimal)) and not isinstance(value, bool) else None


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

    def __init__(self, known_units: tuple[str, ...] = MODELS_DEV_UNITS) -> None:
        self.count = 0
        self.examples: list[str] = []
        self.known_units = known_units
        self.unknown_units: list[str] = []

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
            type=None,
        )

    def model_v2(self, key: str, raw: dict[str, Any], path: str) -> ModelsDevModel:
        modalities = self.mapping(raw, "modalities", path) or {}
        limit = self.mapping(raw, "limit", path) or {}
        capabilities = self.mapping(raw, "capabilities", path) or {}
        at = f"{path}.capabilities"
        inputs = self.modalities(modalities, "input", f"{path}.modalities")
        pricing = self.pricing(raw, path)
        return ModelsDevModel(
            id=self.text(raw, "id", path) or key,
            name=self.text(raw, "name", path),
            description=self.text(raw, "description", path),
            family=self.text(raw, "family", path),
            canonical_model_id=self.text(raw, "canonical_id", path),
            status=self.text(raw, "status", path),
            input_modalities=inputs,
            output_modalities=self.modalities(modalities, "output", f"{path}.modalities"),
            reasoning=self.flag(self.mapping(capabilities, "reasoning", at) or {}, "supported", f"{at}.reasoning"),
            tool_call=self.flag(self.mapping(capabilities, "tools", at) or {}, "supported", f"{at}.tools"),
            structured_output=self.flag(capabilities, "structured_output", at),
            attachment=any(modality != "text" for modality in inputs),
            temperature=self.flag(capabilities, "temperature", at),
            open_weights=self.flag(raw, "open_weights", path),
            context_window=self.limit(limit, "context", f"{path}.limit"),
            max_output_tokens=self.limit(limit, "output", f"{path}.limit"),
            knowledge=self.text(raw, "knowledge", path),
            release_date=self.text(raw, "release_date", path),
            last_updated=self.text(raw, "last_updated", path),
            cost=None if pricing is None else token_cost(pricing),
            type=self.text(raw, "type", path) or "chat",
            api=self.api(raw, path),
            pricing=pricing,
        )

    def modalities(self, raw: dict[str, Any], name: str, path: str) -> tuple[str, ...]:
        """v2 lists media types (``image/png``); the gateway reads the v1 word for each, once."""
        found: list[str] = []
        for media_type in self.texts(raw, name, path):
            modality = _v1_modality(media_type)
            if modality is None:
                self.note(f"{path}.{name}", "a known media type")
            elif modality not in found:
                found.append(modality)
        return tuple(found)

    def price(self, raw: dict[str, Any], name: str, path: str, malformed: list[str], field: str = "") -> Number | None:
        value = raw.get(name)
        if value is None:
            return None
        if (number := _number(value)) is None:
            self.note(f"{path}.{name}", "a number")
            malformed.append(field or name)
        return number

    def condition(self, raw: dict[str, Any], path: str) -> Mapping[str, object] | None:
        value = raw.get("when")
        if isinstance(value, dict):
            return _frozen_mapping(value)
        self.note(f"{path}.when", "an object")
        return None

    def objects(self, raw: dict[str, Any], name: str, path: str) -> Iterator[tuple[str, dict[str, Any]]]:
        """Each object of a list field, with its path."""
        value = raw.get(name)
        if value is None:
            return
        if not isinstance(value, list):
            self.note(f"{path}.{name}", "a list")
            return
        for n, item in enumerate(value):
            if isinstance(item, dict):
                yield f"{path}.{name}[{n}]", item
            else:
                self.note(f"{path}.{name}[{n}]", "an object")

    def cache(
        self, raw: dict[str, Any], path: str, malformed: list[str]
    ) -> tuple[ModelsDevCachePrices | None, ModelsDevCachePrices | None]:
        """A ``cache`` block and the ``explicit`` block in it; a bad field joins ``malformed`` by its path."""
        cache = self.mapping(raw, "cache", path)
        if cache is None:
            return None, None
        path = f"{path}.cache"
        explicit = self.mapping(cache, "explicit", path)
        return (
            self.cache_prices(cache, path, malformed, "cache"),
            None if explicit is None else self.cache_prices(explicit, f"{path}.explicit", malformed, "cache.explicit"),
        )

    def cache_prices(self, cache: dict[str, Any], path: str, malformed: list[str], field: str) -> ModelsDevCachePrices:
        write = cache.get("write")
        first: Number | None = None
        ttls: list[tuple[str, Number]] = []
        if isinstance(write, dict):
            for n, (ttl, price) in enumerate(write.items()):
                if (number := _number(price)) is None:
                    self.note(f"{path}.write.{_segment(ttl)}", "a number")
                    malformed.append(f"{field}.write")
                else:
                    ttls.append((str(ttl), number))
                first = number if n == 0 else first
        else:
            first = self.price(cache, "write", path, malformed, f"{field}.write")
        return ModelsDevCachePrices(
            read=self.price(cache, "read", path, malformed, f"{field}.read"),
            write=first,
            write_ttls=tuple(ttls),
            storage=self.price(cache, "storage", path, malformed, f"{field}.storage"),
        )

    def override(self, raw: dict[str, Any], path: str) -> ModelsDevOverride | None:
        when = self.condition(raw, path)
        malformed: list[str] = []
        price = self.price(raw, "price", path, malformed)
        reasoning = self.price(raw, "reasoning", path, malformed)
        cache, explicit = self.cache(raw, path, malformed)
        if when is None:
            return None
        return ModelsDevOverride(
            when=when,
            price=price,
            reasoning=reasoning,
            cache=cache,
            explicit_cache=explicit,
            malformed=tuple(malformed),
        )

    def unit_price(self, raw: dict[str, Any], name: str, path: str) -> ModelsDevUnitPrice | None:
        entry = self.mapping(raw, name, path)
        if entry is None:
            return None
        path = f"{path}.{name}"
        malformed: list[str] = []
        price = self.price(entry, "price", path, malformed)
        reasoning = self.price(entry, "reasoning", path, malformed)
        cache, explicit = self.cache(entry, path, malformed)
        overrides = [self.override(item, at) for at, item in self.objects(entry, "overrides", path)]
        unit = self.text(entry, "unit", path)
        if unit is not None and unit not in self.known_units and unit not in self.unknown_units:
            self.unknown_units.append(unit)
        return ModelsDevUnitPrice(
            price=price,
            unit=unit,
            reasoning=reasoning,
            cache=cache,
            explicit_cache=explicit,
            overrides=tuple(override for override in overrides if override is not None),
            malformed=tuple(malformed),
        )

    def multiplier(self, raw: dict[str, Any], path: str) -> ModelsDevMultiplier | None:
        when = self.condition(raw, path)
        value = raw.get("factor")
        factor: Number | Mapping[str, object] | None = (
            _frozen_mapping(value) if isinstance(value, dict) else _number(value)
        )
        if factor is None:
            self.note(f"{path}.factor", "a number or an object")
        return None if when is None or factor is None else ModelsDevMultiplier(when=when, factor=factor)

    def pricing(self, raw: dict[str, Any], path: str) -> ModelsDevPricing | None:
        cost = self.mapping(raw, "cost", path)
        return None if cost is None else self.cost_v2(cost, f"{path}.cost")

    def cost_v2(self, cost: dict[str, Any], path: str) -> ModelsDevPricing:
        """A ``cost`` block: each key but ``web_search`` and ``multipliers`` is a modality."""
        modalities: dict[str, ModelsDevModalityCost] = {}
        for key, value in cost.items():
            if key in ("web_search", "multipliers"):
                continue
            at = f"{path}.{_segment(key)}"
            if isinstance(value, dict):
                modalities[str(key)] = ModelsDevModalityCost(
                    input=self.unit_price(value, "input", at), output=self.unit_price(value, "output", at)
                )
            else:
                self.note(at, "an object")
        multipliers = [self.multiplier(item, at) for at, item in self.objects(cost, "multipliers", path)]
        return ModelsDevPricing(
            modalities=MappingProxyType(modalities),
            web_search=self.unit_price(cost, "web_search", path),
            multipliers=tuple(multiplier for multiplier in multipliers if multiplier is not None),
        )

    def api(self, raw: dict[str, Any], path: str) -> tuple[ModelsDevApi, ...]:
        api = self.mapping(raw, "api", path)
        if api is None:
            return ()
        entries: list[ModelsDevApi] = []
        for protocol, entry in api.items():
            at = f"{path}.api.{_segment(protocol)}"
            if isinstance(entry, dict):
                base_url, url_path = self.text(entry, "base_url", at), self.text(entry, "path", at)
                entries.append(ModelsDevApi(protocol=str(protocol), base_url=base_url, path=url_path))
            else:
                self.note(at, "an object")
        return tuple(entries)


def _frozen(value: object) -> object:
    if isinstance(value, dict):
        return _frozen_mapping(value)
    if isinstance(value, list):
        return tuple(_frozen(item) for item in value)
    return value


def _frozen_mapping(value: dict[Any, Any]) -> Mapping[str, object]:
    """A read-only copy of a JSON object, nested objects and lists included."""
    return MappingProxyType({str(key): _frozen(item) for key, item in value.items()})


def _v1_modality(media_type: str) -> str | None:
    """The v1 word for a v2 modality, as models.dev derives v1 (``toV1Modality``); ``None`` where v1 has none."""
    if media_type == "application/pdf":
        return "pdf"
    if media_type in ("decision", "embedding", "reranking"):
        return "text"
    base = media_type.partition("/")[0]
    return base if base in ("text", "image", "audio", "video") else None


def _context_size(when: Mapping[str, object]) -> int | None:
    """Where a v1 tier starts, for a condition on context alone; ``gt`` and ``gte`` both give its size."""
    context = when.get("context")
    if set(when) != {"context"} or not isinstance(context, Mapping):
        return None
    bound = context.get("gt")
    return _tier_size(context.get("gte") if bound is None else bound)


def _cache_write(price: ModelsDevUnitPrice | ModelsDevOverride) -> Number | None:
    """v1's one cache write price: the default TTL's, else the explicit cache's."""
    for cache in (price.cache, price.explicit_cache):
        if cache is not None and cache.write is not None:
            return cache.write
    return None


def _rates_malformed(given: tuple[str, ...], made: tuple[str, ...]) -> tuple[str, ...]:
    """The v1 rate names of the v2 price fields that are present but not numbers."""
    found = {
        "input": "price" in given,
        "output": "price" in made,
        "cache_read": "cache.read" in given,
        "cache_write": "cache.write" in given or "cache.explicit.write" in given,
    }
    return tuple(name for name, bad in found.items() if bad)


def token_cost(pricing: ModelsDevPricing) -> ModelsDevCost | None:
    """The per-token text prices v1 shows for a v2 ``cost`` block, the way models.dev derives them (``costV2ToV1``).

    v1 has a price only where text input and output are both per token. Its
    tiers are the overrides on context alone, each starting from the base
    input and output prices and taking what the override replaces.
    """
    text = pricing.modalities.get("text")
    if text is None or text.input is None or text.output is None:
        return None
    if text.input.unit not in (None, "token") or text.output.unit not in (None, "token"):
        return None
    return direction_cost(text.input, text.output)


def direction_cost(given: ModelsDevUnitPrice | None, made: ModelsDevUnitPrice | None) -> ModelsDevCost:
    """The v1 shape of one price pair in one unit: ``given`` as the input rates, ``made`` as the output rates.

    The tiers are the overrides on context alone, each starting from the base
    prices and taking what the override replaces. A side that is absent gives
    no price. The caller says what the prices are in; this reads the numbers.
    """
    tiers: dict[int, dict[str, Number | None]] = {}
    bad: dict[int, tuple[list[str], list[str]]] = {}
    given_price = None if given is None else given.price
    made_price = None if made is None else made.price

    def tier(override: ModelsDevOverride, *, output: bool) -> dict[str, Number | None] | None:
        size = _context_size(override.when)
        if size is None:
            return None
        bad.setdefault(size, ([], []))[output].extend(override.malformed)
        return tiers.setdefault(size, {"input": given_price, "output": made_price})

    for override in () if given is None else given.overrides:
        if (entry := tier(override, output=False)) is not None:
            replaced = (
                ("input", override.price),
                ("cache_read", None if override.cache is None else override.cache.read),
                ("cache_write", _cache_write(override)),
            )
            entry.update((name, value) for name, value in replaced if value is not None)
    for override in () if made is None else made.overrides:
        if (entry := tier(override, output=True)) is not None and override.price is not None:
            entry["output"] = override.price
    return ModelsDevCost(
        rates=ModelsDevRates(
            input=given_price,
            output=made_price,
            cache_read=None if given is None or given.cache is None else given.cache.read,
            cache_write=None if given is None else _cache_write(given),
            malformed=_rates_malformed(
                () if given is None else given.malformed, () if made is None else made.malformed
            ),
        ),
        tiers=tuple(
            ModelsDevTier(
                kind="context",
                size=size,
                rates=ModelsDevRates(**fields, malformed=_rates_malformed(*map(tuple, bad[size]))),
            )
            for size, fields in tiers.items()
        ),
    )


def parse_pricing(
    raw: object, *, units: tuple[str, ...] = MODELS_DEV_UNITS, path: str = "cost"
) -> tuple[ModelsDevPricing, ModelsDevIssues]:
    """Read one ``cost`` block on its own, with the issues the read noted.

    The same reader the catalog goes through, so a stored rate and the
    document it came from read alike: a key it does not know is ignored, and a
    value it cannot read is an issue, read as absent. ``units`` are the price
    units it knows; one outside them joins ``unknown_units``. Each issue is
    named under ``path``. A block that is not an object raises ``TypeError``.
    """
    if not isinstance(raw, Mapping):
        raise TypeError("a cost block is not a JSON object")
    reader = _Reader(units)
    pricing = reader.cost_v2(dict(raw), path)
    issues = ModelsDevIssues(
        count=reader.count,
        examples=tuple(reader.examples),
        models=1,
        damaged_models=int(reader.count > 0),
        unknown_units=tuple(reader.unknown_units),
    )
    return pricing, issues


def pricing_document(pricing: ModelsDevPricing) -> dict[str, Any]:
    """A ``cost`` block as models.dev writes it, from what :func:`parse_pricing` read, so the two round-trip."""
    document: dict[str, Any] = {}
    for modality, prices in pricing.modalities.items():
        entry: dict[str, Any] = {}
        if prices.input is not None:
            entry["input"] = _unit_price_document(prices.input)
        if prices.output is not None:
            entry["output"] = _unit_price_document(prices.output)
        document[modality] = entry
    if pricing.web_search is not None:
        document["web_search"] = _unit_price_document(pricing.web_search)
    if pricing.multipliers:
        document["multipliers"] = [
            {"when": _thawed(multiplier.when), "factor": _thawed(multiplier.factor)}
            for multiplier in pricing.multipliers
        ]
    return document


def _unit_price_document(price: ModelsDevUnitPrice | ModelsDevOverride) -> dict[str, Any]:
    document: dict[str, Any] = {}
    if isinstance(price, ModelsDevOverride):
        document["when"] = _thawed(price.when)
    if price.price is not None:
        document["price"] = price.price
    if isinstance(price, ModelsDevUnitPrice) and price.unit is not None:
        document["unit"] = price.unit
    if price.reasoning is not None:
        document["reasoning"] = price.reasoning
    cache = _cache_document(price.cache)
    if price.explicit_cache is not None:
        cache["explicit"] = _cache_document(price.explicit_cache)
    if cache:
        document["cache"] = cache
    if isinstance(price, ModelsDevUnitPrice) and price.overrides:
        document["overrides"] = [_unit_price_document(override) for override in price.overrides]
    return document


def _cache_document(cache: ModelsDevCachePrices | None) -> dict[str, Any]:
    if cache is None:
        return {}
    document: dict[str, Any] = {}
    if cache.read is not None:
        document["read"] = cache.read
    if cache.write_ttls:
        document["write"] = dict(cache.write_ttls)
    elif cache.write is not None:
        document["write"] = cache.write
    if cache.storage is not None:
        document["storage"] = cache.storage
    return document


def _thawed(value: object) -> object:
    """A read-only mapping or tuple back as the JSON object or list it was read from."""
    if isinstance(value, Mapping):
        return {key: _thawed(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thawed(item) for item in value]
    return value


def _segment(key: object) -> str:
    """A document key as a path segment. Keys are fetched data and paths are logged, so an odd one is quoted."""
    text = str(key)
    return text if text.isprintable() and len(text) <= _MAX_SEGMENT else repr(text[:_MAX_SEGMENT])


def parse_catalog(raw: object) -> ModelsDevCatalog:
    """Read a parsed v1 ``api.json``, noting in ``issues`` what it could not read.

    A provider without a ``models`` object, and a model that is not an object,
    are left out, and a field of the wrong type is read as absent, so one bad
    entry costs only itself. Whether the document as a whole is fit to serve is
    :meth:`ModelsDevCatalog.refusal`. A document that is not an object raises
    ``TypeError``.
    """
    return _parse(raw, _Reader.model)


def parse_catalog_v2(raw: object) -> ModelsDevCatalog:
    """Read a parsed v2 ``api.json`` into the same catalog, by the same rules as :func:`parse_catalog`."""
    return _parse(raw, _Reader.model_v2)


def _parse(raw: object, read_model: Callable[[_Reader, str, dict[str, Any], str], ModelsDevModel]) -> ModelsDevCatalog:
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
                models[str(model_key)] = read_model(reader, str(model_key), model, model_path)
            else:
                reader.note(model_path, "an object")
            damaged += reader.count > before
        providers[str(key)] = ModelsDevProvider(
            id=reader.text(provider, "id", path) or str(key),
            name=reader.text(provider, "name", path),
            doc=reader.text(provider, "doc", path),
            models=MappingProxyType(models),
        )
    issues = ModelsDevIssues(
        count=reader.count,
        examples=tuple(reader.examples),
        models=seen,
        damaged_models=damaged,
        unknown_units=tuple(reader.unknown_units),
    )
    return ModelsDevCatalog(providers=MappingProxyType(providers), issues=issues)


@dataclass(frozen=True, slots=True)
class ModelsDevSource:
    """One document models.dev publishes, where it lives and the reader for it."""

    name: str
    url: str
    parse: Callable[[object], ModelsDevCatalog]


# The documents in the order the gateway tries them: v2, then v1 when v2 fails.
# v2 still lives under ``/experimental/``; when it moves, its URL here is the one
# change. Without ``?type=all``, models.dev leaves out the decision models.
MODELS_DEV_SOURCES = (
    ModelsDevSource("v2", "https://models.dev/experimental/v2.0/api.json?type=all", parse_catalog_v2),
    ModelsDevSource("v1", "https://models.dev/api.json", parse_catalog),
)


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


def _encoded(value: object) -> object:
    """A catalog value as JSON: a dataclass without the fields left at their default, a tuple as a list."""
    if is_dataclass(value) and not isinstance(value, type):
        document: dict[str, object] = {}
        for field in fields(value):
            item = getattr(value, field.name)
            if field.default is MISSING or item != field.default:
                document[field.name] = _encoded(item)
        return document
    if isinstance(value, Mapping):
        return {key: _encoded(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_encoded(item) for item in value]
    return value


def catalog_document(catalog: ModelsDevCatalog, **meta: object) -> dict[str, Any]:
    """The catalog as JSON with every field it has, for :func:`read_catalog_document` to read back.

    ``meta`` joins the format marker under ``_meta``.
    """
    document = _encoded(catalog)
    assert isinstance(document, dict)
    return {_META_KEY: {"format": CATALOG_FORMAT, **meta}, **document}


def _load[T](cls: Callable[..., T], raw: object, **read: Callable[[Any], object]) -> T:
    """``cls`` from a stored object: ``read`` reads each field that holds catalog types, and the rest read frozen."""
    if not isinstance(raw, dict):
        raise TypeError("expected an object")
    return cls(**{name: read[name](value) if name in read else _frozen(value) for name, value in raw.items()})


def _each[T](read: Callable[[Any], T]) -> Callable[[Any], tuple[T, ...]]:
    return lambda raw: tuple(read(item) for item in raw)


def _keyed[T](read: Callable[[Any], T]) -> Callable[[Any], Mapping[str, T]]:
    return lambda raw: MappingProxyType({str(key): read(item) for key, item in raw.items()})


def _stored_rates(raw: object) -> ModelsDevRates:
    return _load(ModelsDevRates, raw)


def _stored_tier(raw: object) -> ModelsDevTier:
    return _load(ModelsDevTier, raw, rates=_stored_rates)


def _stored_cost(raw: object) -> ModelsDevCost:
    return _load(ModelsDevCost, raw, rates=_stored_rates, tiers=_each(_stored_tier), context_over_200k=_stored_rates)


def _stored_cache(raw: object) -> ModelsDevCachePrices:
    return _load(ModelsDevCachePrices, raw)


def _stored_override(raw: object) -> ModelsDevOverride:
    return _load(ModelsDevOverride, raw, cache=_stored_cache, explicit_cache=_stored_cache)


def _stored_unit_price(raw: object) -> ModelsDevUnitPrice:
    return _load(
        ModelsDevUnitPrice, raw, cache=_stored_cache, explicit_cache=_stored_cache, overrides=_each(_stored_override)
    )


def _stored_modality(raw: object) -> ModelsDevModalityCost:
    return _load(ModelsDevModalityCost, raw, input=_stored_unit_price, output=_stored_unit_price)


def _stored_multiplier(raw: object) -> ModelsDevMultiplier:
    return _load(ModelsDevMultiplier, raw)


def _stored_pricing(raw: object) -> ModelsDevPricing:
    return _load(
        ModelsDevPricing,
        raw,
        modalities=_keyed(_stored_modality),
        web_search=_stored_unit_price,
        multipliers=_each(_stored_multiplier),
    )


def _stored_api(raw: object) -> ModelsDevApi:
    return _load(ModelsDevApi, raw)


def _stored_model(raw: object) -> ModelsDevModel:
    return _load(ModelsDevModel, raw, cost=_stored_cost, api=_each(_stored_api), pricing=_stored_pricing)


def _stored_provider(raw: object) -> ModelsDevProvider:
    return _load(ModelsDevProvider, raw, models=_keyed(_stored_model))


def _stored_issues(raw: object) -> ModelsDevIssues:
    return _load(ModelsDevIssues, raw)


def read_catalog_document(raw: object) -> ModelsDevCatalog:
    """Read a document :func:`catalog_document` wrote, or an older price snapshot, which has no format marker.

    The marker, not the shape, says which it is. A stored catalog is the
    gateway's own copy, so it is read whole or not at all: anything this
    version cannot read, a newer format included, raises ``ValueError``.
    """
    if not isinstance(raw, Mapping):
        raise ValueError("The stored models.dev catalog is not a JSON object")
    body = {key: value for key, value in raw.items() if key != _META_KEY}
    meta = raw.get(_META_KEY)
    if not isinstance(meta, Mapping) or "format" not in meta:
        return parse_catalog(body)
    if meta["format"] != CATALOG_FORMAT:
        raise ValueError(f"The stored models.dev catalog has format {meta['format']!r}, not {CATALOG_FORMAT}")
    try:
        return _load(ModelsDevCatalog, body, providers=_keyed(_stored_provider), issues=_stored_issues)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"The stored models.dev catalog cannot be read: {type(exc).__name__}") from exc
