"""An immutable price index over a typed models.dev catalog.

Pure and synchronous: no database, no network. The index answers what a model
costs, its context window and its display metadata, matching a provider and a
model the way the gateway names them against the catalog's own ids. Rates are
USD per million tokens, as ``Decimal`` taken from the JSON number's shortest
spelling.
"""

import re
from bisect import bisect_right
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from gateway.core.metered_pricing import TIER_THRESHOLD_FIELD
from gateway.types.models_dev import (
    RATE_FIELDS,
    ModelsDevCatalog,
    ModelsDevCost,
    ModelsDevModel,
    ModelsDevPricing,
    ModelsDevProvider,
    ModelsDevRates,
    models_dev_provider_id,
    models_dev_provider_ids,
    parse_catalog,
)
from gateway.types.price_rates import RateColumns, columns_from_pricing

REFERENCE_PREFIX = "models.dev:"

_OVER_200K_THRESHOLD = 200_000

# HuggingFace backend names that models.dev files under another provider id.
_HF_BACKEND_PROVIDER_IDS: Mapping[str, tuple[str, ...]] = {
    "together": ("togetherai",),
    "novita": ("novita-ai",),
    "ovh": ("ovhcloud",),
    "nebius": ("nebius", "nebius-ai-studio"),
}

_BEDROCK_PROVIDER_ID = "amazon-bedrock"
_BEDROCK_GEO_PREFIXES = ("us", "eu", "global", "apac", "jp", "au")
_BEDROCK_VERSION_SUFFIX = re.compile(r"(?:-v\d+(?::\d+)?|:\d+)$")

# Upper bound on a believable rate, USD per million tokens. Anything above it,
# negative or not finite is treated as a data error, not a price.
MAX_RATE_PER_MILLION = Decimal(1_000_000)
_RATE_PLACES = Decimal("0.00000001")

_TIER_FIELD_NAMES = {
    "input": "input_price_per_million",
    "output": "output_price_per_million",
    "cache_read": "cache_read_price_per_million",
    "cache_write": "cache_write_price_per_million",
}


@dataclass(frozen=True, slots=True)
class PriceTier:
    """Rates that apply once a request's input reaches ``min_input_tokens``."""

    min_input_tokens: int
    input: Decimal | None = None
    output: Decimal | None = None
    cache_read: Decimal | None = None
    cache_write: Decimal | None = None


@dataclass(frozen=True, slots=True)
class ModelsDevPrice:
    """One catalog model: its rates, context window and display metadata."""

    provider_id: str
    model_id: str
    input: Decimal | None
    output: Decimal | None
    cache_read: Decimal | None
    cache_write: Decimal | None
    tiers: tuple[PriceTier, ...]
    context_window: int | None
    model_name: str | None
    provider_name: str | None
    provider_doc: str | None
    canonical_model_id: str | None
    pricing: ModelsDevPricing | None = None
    """Every price the document gives the model, in its unit; ``None`` for a v1 entry."""
    slot_columns: RateColumns | None = None
    """The columns of the one slot the block prices, for a model with no text price; ``None`` otherwise.

    A transcription model priced per minute of audio and a speech model priced
    per character of text have no text token price, so the rate columns read
    the slot the block does price, in its unit. A block whose text price is
    half given is never read this way: a missing text rate is not free.
    """

    @property
    def reference(self) -> str:
        """Stable ``models.dev:<provider_id>/<model_id>`` identifier."""
        return f"{REFERENCE_PREFIX}{self.provider_id}/{self.model_id}"

    @property
    def priced(self) -> bool:
        return self.input is not None or self.output is not None or self.slot_columns is not None

    def pricing_tiers(self) -> list[dict[str, float | int]]:
        """The tiers in the ``pricing_tiers`` column shape, rates as JSON numbers."""
        return [
            {
                TIER_THRESHOLD_FIELD: tier.min_input_tokens,
                **{
                    _TIER_FIELD_NAMES[name]: float(rate)
                    for name in RATE_FIELDS
                    if (rate := getattr(tier, name)) is not None
                },
            }
            for tier in self.tiers
        ]

    def price_signature(self) -> tuple[object, ...]:
        slot = self.slot_columns
        in_slot = None if slot is None else (slot.unit, slot.input_price_per_million, slot.output_price_per_million)
        return (self.input, self.output, self.cache_read, self.cache_write, self.tiers, in_slot)


def parse_rate(value: object) -> Decimal | None:
    """A JSON number as a rate, or ``None`` when it is not a believable price.

    Rejects booleans, negatives, non-finite numbers, anything above
    ``MAX_RATE_PER_MILLION`` and anything that does not fit ``Numeric(18, 8)``.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        return None
    try:
        rate = Decimal(str(value))
        if not rate.is_finite() or rate < 0 or rate > MAX_RATE_PER_MILLION:
            return None
        rate.quantize(_RATE_PLACES)
    except InvalidOperation:
        return None
    return rate


def invalid_rate_count(catalog: ModelsDevCatalog) -> int:
    """How many rates in a catalog are present but not believable prices."""

    def bad(rates: ModelsDevRates | None) -> int:
        if rates is None:
            return 0
        values = (getattr(rates, name) for name in RATE_FIELDS)
        return len(rates.malformed) + sum(1 for value in values if value is not None and parse_rate(value) is None)

    return sum(
        bad(model.cost.rates) + bad(model.cost.context_over_200k) + sum(bad(tier.rates) for tier in model.cost.tiers)
        for _provider, model in catalog.models()
        if model.cost is not None
    )


def _rates(rates: ModelsDevRates) -> dict[str, Decimal]:
    return {name: rate for name in RATE_FIELDS if (rate := parse_rate(getattr(rates, name))) is not None}


def _tiers(cost: ModelsDevCost) -> tuple[PriceTier, ...]:
    """``tiers`` of the context type, else ``context_over_200k`` as one tier."""
    by_threshold: dict[int, PriceTier] = {}
    for tier in cost.tiers:
        if tier.kind == "context" and tier.size is not None and (rates := _rates(tier.rates)):
            by_threshold[tier.size] = PriceTier(min_input_tokens=tier.size, **rates)
    if not by_threshold and cost.context_over_200k is not None and (rates := _rates(cost.context_over_200k)):
        by_threshold[_OVER_200K_THRESHOLD] = PriceTier(min_input_tokens=_OVER_200K_THRESHOLD, **rates)
    return tuple(by_threshold[threshold] for threshold in sorted(by_threshold))


def _slot_columns(pricing: ModelsDevPricing | None, rates: Mapping[str, Decimal]) -> RateColumns | None:
    """The columns of the one measured slot ``pricing`` prices, for a block with no text price."""
    if pricing is None or (rates.get("input") is not None and rates.get("output") is not None):
        return None
    try:
        columns = columns_from_pricing(pricing)
    except ValueError:
        return None
    return columns if columns.unit not in ("token", "request") else None


def _entry(provider: ModelsDevProvider, model: ModelsDevModel) -> ModelsDevPrice:
    cost = model.cost or ModelsDevCost(rates=ModelsDevRates())
    rates = _rates(cost.rates)
    return ModelsDevPrice(
        provider_id=provider.id,
        model_id=model.id,
        input=rates.get("input"),
        output=rates.get("output"),
        cache_read=rates.get("cache_read"),
        cache_write=rates.get("cache_write"),
        tiers=_tiers(cost),
        context_window=model.context_window,
        model_name=model.name,
        provider_name=provider.name,
        provider_doc=provider.doc,
        canonical_model_id=model.canonical_model_id,
        pricing=model.pricing,
        slot_columns=_slot_columns(model.pricing, rates),
    )


def _split_geo(model: str) -> tuple[str, str]:
    for geo in _BEDROCK_GEO_PREFIXES:
        if model.startswith(f"{geo}."):
            return geo, model[len(geo) + 1 :]
    return "", model


def _strip_suffix(model: str) -> str:
    """Drop the default-deployment suffix, ``-v1:0`` or ``:0``, and no other version."""
    for suffix in ("-v1:0", ":0"):
        if model.endswith(suffix):
            return model[: -len(suffix)]
    return model


def _name_variants(name: str) -> list[str]:
    """The given name, and its default-deployment equivalents.

    A name carrying a version suffix is only ever stripped of ``-v1:0`` or
    ``:0``; a name carrying none gains either. ``-v2:0`` never becomes ``-v1:0``.
    """
    if _BEDROCK_VERSION_SUFFIX.search(name):
        stripped = _strip_suffix(name)
        return [name] if stripped == name else [name, stripped]
    return [name, f"{name}-v1:0", f"{name}:0"]


def _vendor_prefixed(model: str) -> list[tuple[str, str]]:
    """``(head, rest)`` at each dot boundary of a dotted id."""
    attempts: list[tuple[str, str]] = []
    head, separator, rest = model.partition(".")
    while separator and rest:
        attempts.append((head, rest))
        head, separator, rest = rest.partition(".")
    return attempts


class ModelsDevPriceIndex:
    """Immutable lookup over one models.dev catalog."""

    __slots__ = ("_by_canonical", "_by_model_id", "_by_provider")

    def __init__(self, entries: Iterable[ModelsDevPrice]) -> None:
        by_provider: dict[str, dict[str, ModelsDevPrice]] = {}
        by_model_id: dict[str, list[ModelsDevPrice]] = {}
        by_canonical: dict[str, list[ModelsDevPrice]] = {}
        for entry in entries:
            by_provider.setdefault(entry.provider_id, {})[entry.model_id] = entry
            by_model_id.setdefault(entry.model_id.casefold(), []).append(entry)
            if entry.canonical_model_id:
                by_canonical.setdefault(entry.canonical_model_id.casefold(), []).append(entry)
                tail = entry.canonical_model_id.rpartition("/")[2].casefold()
                if tail != entry.canonical_model_id.casefold():
                    by_canonical.setdefault(tail, []).append(entry)
        self._by_provider = by_provider
        self._by_model_id = by_model_id
        self._by_canonical = by_canonical

    @classmethod
    def from_catalog(cls, catalog: ModelsDevCatalog) -> "ModelsDevPriceIndex":
        """Build an index over every model of a typed catalog."""
        return cls(_entry(provider, model) for provider, model in catalog.models())

    @classmethod
    def from_document(cls, raw: Mapping[str, Any]) -> "ModelsDevPriceIndex":
        """Build an index from a parsed ``api.json`` or a stored snapshot, through the one catalog parser."""
        return cls.from_catalog(parse_catalog(raw))

    def __len__(self) -> int:
        return sum(len(models) for models in self._by_provider.values())

    def entries(self) -> Iterator[ModelsDevPrice]:
        """Every catalog entry."""
        for models in self._by_provider.values():
            yield from models.values()

    def get(self, provider_id: str, model_id: str) -> ModelsDevPrice | None:
        """The exact catalog entry, with no matching rules applied."""
        return self._by_provider.get(provider_id, {}).get(model_id)

    def provider_entry(self, provider_id: str) -> ModelsDevPrice | None:
        """Any one entry of ``provider_id``, which carries the provider's name and doc link."""
        for candidate in dict.fromkeys((*models_dev_provider_ids(provider_id), provider_id)):
            if models := self._by_provider.get(candidate):
                return next(iter(models.values()))
        return None

    def resolve(self, provider: str | None, model: str, implementation: str | None = None) -> ModelsDevPrice | None:
        """The entry pricing ``model`` served by ``provider``, or ``None``.

        Tried in order: a HuggingFace pinned-backend selector against the
        backend's own provider; the provider's own entry; the any-llm
        implementation behind the instance; a model id that is unambiguous in
        price across every provider; then the vendor-prefixed spellings of the
        id. Serving providers differ in rate, so the scoped lookups precede the
        agnostic one.
        """
        scoped = [p for p in (provider, implementation) if p]
        if "huggingface" in scoped and ":" in model:
            base, backend = model.rsplit(":", 1)
            for provider_id in _HF_BACKEND_PROVIDER_IDS.get(backend, (backend,)):
                if (found := self.get(provider_id, base)) is not None:
                    return found
        for name in scoped:
            for provider_id in dict.fromkeys((*models_dev_provider_ids(name), name)):
                if (found := self._scoped(provider_id, model)) is not None:
                    return found
        if (found := self._agnostic(model)) is not None:
            return found
        for head, rest in _vendor_prefixed(model):
            provider_id = models_dev_provider_id(head)
            if provider_id in self._by_provider:
                return self._scoped(provider_id, rest) or self._agnostic(rest) or self._agnostic(_strip_suffix(rest))
        return None

    def _scoped(self, provider_id: str, model: str) -> ModelsDevPrice | None:
        if (found := self.get(provider_id, model)) is not None:
            return found
        if provider_id == _BEDROCK_PROVIDER_ID:
            return self._bedrock(model)
        return None

    def _bedrock(self, model: str) -> ModelsDevPrice | None:
        """Bedrock listings for ``model``, never across versions or into another geo.

        Tried in order: the given geo with each default-suffix spelling; the
        geo-less spelling; then, for a geo-less request only, the geo-prefixed
        listings, which must all agree on price.
        """
        geo, name = _split_geo(model)
        variants = _name_variants(name)
        for variant in variants:
            if (found := self.get(_BEDROCK_PROVIDER_ID, f"{geo}.{variant}" if geo else variant)) is not None:
                return found
        if geo:
            for variant in variants:
                if (found := self.get(_BEDROCK_PROVIDER_ID, variant)) is not None:
                    return found
            return None
        listings = {
            (found.provider_id, found.model_id): found
            for variant in variants
            for prefix in _BEDROCK_GEO_PREFIXES
            if (found := self.get(_BEDROCK_PROVIDER_ID, f"{prefix}.{variant}")) is not None
        }
        return self._agreed(list(listings.values()))

    def _agnostic(self, model: str) -> ModelsDevPrice | None:
        """The one entry every listing of ``model`` agrees on in price.

        Candidates are the exact-id and canonical-id matches across providers;
        an unpriced listing among them blocks the answer.
        """
        key = model.casefold()
        candidates = {
            (c.provider_id, c.model_id): c for c in (*self._by_model_id.get(key, ()), *self._by_canonical.get(key, ()))
        }
        return self._agreed(list(candidates.values()))

    @staticmethod
    def _agreed(candidates: list[ModelsDevPrice]) -> ModelsDevPrice | None:
        if not candidates or not all(c.priced for c in candidates):
            return None
        if len({c.price_signature() for c in candidates}) != 1:
            return None
        vendor = (candidates[0].canonical_model_id or "").partition("/")[0]
        agreed = min(candidates, key=lambda c: (c.provider_id != vendor, c.provider_id, c.model_id))
        # The prices agree; the rest of a block (a web search price, an image
        # price) is one host's, so it is given only where every host's is the
        # same. An entry priced in a slot alone has no price without its block.
        same_block = all(c.pricing == agreed.pricing for c in candidates)
        if same_block:
            return agreed
        return None if agreed.slot_columns is not None else replace(agreed, pricing=None)


def _utc(moment: datetime) -> datetime:
    return moment.replace(tzinfo=UTC) if moment.tzinfo is None else moment


@dataclass(frozen=True, slots=True)
class PriceGeneration:
    """An index and the instant it took effect.

    A ``baseline`` generation also answers every earlier date; without one, a
    date before the first generation has no price.
    """

    effective_at: datetime
    index: ModelsDevPriceIndex
    baseline: bool = False


@dataclass(frozen=True, slots=True)
class PriceTimeline:
    """Generations sorted by effective time, with the sort keys computed once."""

    generations: tuple[PriceGeneration, ...]
    _keys: tuple[datetime, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        keys = tuple(_utc(g.effective_at) for g in self.generations)
        if any(later < earlier for earlier, later in zip(keys, keys[1:], strict=False)):
            raise ValueError("price generations must be sorted by effective_at")
        object.__setattr__(self, "_keys", keys)

    @classmethod
    def build(cls, generations: Iterable[PriceGeneration]) -> "PriceTimeline":
        return cls(tuple(sorted(generations, key=lambda g: _utc(g.effective_at))))

    def select(self, as_of: datetime) -> PriceGeneration | None:
        position = bisect_right(self._keys, _utc(as_of))
        if position:
            return self.generations[position - 1]
        first = self.generations[0] if self.generations else None
        return first if first is not None and first.baseline else None


def select_generation(
    generations: Sequence[PriceGeneration] | PriceTimeline, as_of: datetime
) -> PriceGeneration | None:
    """The latest generation in effect at ``as_of``, or ``None`` before the first unless it is a baseline."""
    timeline = generations if isinstance(generations, PriceTimeline) else PriceTimeline.build(generations)
    return timeline.select(as_of)


def resolve_as_of(
    generations: Sequence[PriceGeneration] | PriceTimeline,
    as_of: datetime,
    provider: str | None,
    model: str,
    implementation: str | None = None,
) -> ModelsDevPrice | None:
    """Resolve ``model`` against the generation in effect at ``as_of``."""
    generation = select_generation(generations, as_of)
    return None if generation is None else generation.index.resolve(provider, model, implementation)
