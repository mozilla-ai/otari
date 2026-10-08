"""An immutable price index over a models.dev ``api.json`` document.

Pure and synchronous: no database, no network. The index answers what a model
costs, its context window and its display metadata, matching a provider and a
model the way the gateway names them against the catalog's own ids. Rates are
USD per million tokens, as ``Decimal`` taken from the JSON number's shortest
spelling.
"""

import re
from bisect import bisect_right
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from gateway.core.metered_pricing import TIER_THRESHOLD_FIELD
from gateway.services.model_catalog_service import models_dev_provider_id

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
_BEDROCK_VERSION_SUFFIX = re.compile(r"-v\d+(?::\d+)?$")
_BEDROCK_BARE_SUFFIX = re.compile(r":\d+$")

_RATE_FIELDS = ("input", "output", "cache_read", "cache_write")
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

    @property
    def reference(self) -> str:
        """Stable ``models.dev:<provider_id>/<model_id>`` identifier."""
        return f"{REFERENCE_PREFIX}{self.provider_id}/{self.model_id}"

    @property
    def priced(self) -> bool:
        return self.input is not None or self.output is not None

    def pricing_tiers(self) -> list[dict[str, float | int]]:
        """The tiers in the ``pricing_tiers`` column shape, rates as JSON numbers."""
        return [
            {
                TIER_THRESHOLD_FIELD: tier.min_input_tokens,
                **{
                    _TIER_FIELD_NAMES[name]: float(rate)
                    for name in _RATE_FIELDS
                    if (rate := getattr(tier, name)) is not None
                },
            }
            for tier in self.tiers
        ]

    def _price_signature(self) -> tuple[object, ...]:
        return (self.input, self.output, self.cache_read, self.cache_write, self.tiers)


def _decimal(value: object) -> Decimal | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        rate = Decimal(str(value))
    except InvalidOperation:
        return None
    return rate if rate.is_finite() and rate >= 0 else None


def _rates(raw: object) -> dict[str, Decimal]:
    if not isinstance(raw, dict):
        return {}
    return {name: rate for name in _RATE_FIELDS if (rate := _decimal(raw.get(name))) is not None}


def _tiers(cost: Mapping[str, Any]) -> tuple[PriceTier, ...]:
    """``tiers`` of the context type, else ``context_over_200k`` as one tier."""
    by_threshold: dict[int, PriceTier] = {}
    raw_tiers = cost.get("tiers")
    if isinstance(raw_tiers, list):
        for raw in raw_tiers:
            if not isinstance(raw, dict):
                continue
            kind = raw.get("tier")
            if not isinstance(kind, dict) or kind.get("type") != "context":
                continue
            size = kind.get("size")
            if isinstance(size, bool) or not isinstance(size, (int, float)) or size < 0:
                continue
            rates = _rates(raw)
            if rates:
                by_threshold[int(size)] = PriceTier(min_input_tokens=int(size), **rates)
    elif rates := _rates(cost.get("context_over_200k")):
        by_threshold[_OVER_200K_THRESHOLD] = PriceTier(min_input_tokens=_OVER_200K_THRESHOLD, **rates)
    return tuple(by_threshold[threshold] for threshold in sorted(by_threshold))


def _str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _context_window(model: Mapping[str, Any]) -> int | None:
    limit = model.get("limit")
    value = limit.get("context") if isinstance(limit, dict) else None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _entry(provider: Mapping[str, Any], provider_id: str, model_id: str, model: Mapping[str, Any]) -> ModelsDevPrice:
    cost = model.get("cost")
    cost = cost if isinstance(cost, dict) else {}
    rates = _rates(cost)
    return ModelsDevPrice(
        provider_id=provider_id,
        model_id=model_id,
        input=rates.get("input"),
        output=rates.get("output"),
        cache_read=rates.get("cache_read"),
        cache_write=rates.get("cache_write"),
        tiers=_tiers(cost),
        context_window=_context_window(model),
        model_name=_str(model.get("name")),
        provider_name=_str(provider.get("name")),
        provider_doc=_str(provider.get("doc")),
        canonical_model_id=_str(model.get("canonical_model_id")),
    )


def _bedrock_variants(model: str) -> list[str]:
    """Spellings of a Bedrock model id to try, the given one first.

    Geo prefixes (``us.``) and the ``-v1:0`` or ``:0`` suffix are dropped and
    re-added both ways, because the catalog lists some models with either.
    """
    stem = model
    for geo in _BEDROCK_GEO_PREFIXES:
        if model.startswith(f"{geo}."):
            stem = model[len(geo) + 1 :]
            break
    bases = [model] if stem == model else [model, stem]
    variants: list[str] = []
    for base in bases:
        trimmed = _strip_suffix(base)
        variants.extend([base, trimmed, f"{trimmed}-v1:0", f"{trimmed}:0"])
    trimmed = _strip_suffix(stem)
    for geo in _BEDROCK_GEO_PREFIXES:
        variants.extend([f"{geo}.{stem}", f"{geo}.{trimmed}", f"{geo}.{trimmed}-v1:0", f"{geo}.{trimmed}:0"])
    return list(dict.fromkeys(variants))


def _strip_suffix(model: str) -> str:
    return _BEDROCK_BARE_SUFFIX.sub("", _BEDROCK_VERSION_SUFFIX.sub("", model))


def _vendor_prefixed(model: str) -> list[tuple[str, str]]:
    """``(vendor, model)`` candidates at each dot boundary of a vendor-prefixed id."""
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
                by_canonical.setdefault(tail, []).append(entry)
        self._by_provider = by_provider
        self._by_model_id = by_model_id
        self._by_canonical = by_canonical

    @classmethod
    def from_catalog(cls, raw: Mapping[str, Any]) -> "ModelsDevPriceIndex":
        """Build an index from a parsed ``api.json``, skipping malformed entries."""
        entries: list[ModelsDevPrice] = []
        for key, provider in raw.items():
            if not isinstance(provider, dict) or not isinstance(provider.get("models"), dict):
                continue
            provider_id = _str(provider.get("id")) or str(key)
            for model_key, model in provider["models"].items():
                if isinstance(model, dict):
                    entries.append(_entry(provider, provider_id, _str(model.get("id")) or str(model_key), model))
        return cls(entries)

    def __len__(self) -> int:
        return sum(len(models) for models in self._by_provider.values())

    def get(self, provider_id: str, model_id: str) -> ModelsDevPrice | None:
        """The exact catalog entry, with no matching rules applied."""
        return self._by_provider.get(provider_id, {}).get(model_id)

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
            for provider_id in dict.fromkeys((models_dev_provider_id(name), name)):
                if (found := self._scoped(provider_id, model)) is not None:
                    return found
        if (found := self._unambiguous(model)) is not None:
            return found
        for vendor, rest in _vendor_prefixed(model):
            if (found := self._scoped(models_dev_provider_id(vendor), rest)) is not None:
                return found
            if (found := self._unambiguous(rest) or self._unambiguous(_strip_suffix(rest))) is not None:
                return found
        return None

    def _scoped(self, provider_id: str, model: str) -> ModelsDevPrice | None:
        if (found := self.get(provider_id, model)) is not None:
            return found
        if provider_id == _BEDROCK_PROVIDER_ID:
            for variant in _bedrock_variants(model):
                if (found := self.get(provider_id, variant)) is not None:
                    return found
        return None

    def _unambiguous(self, model: str) -> ModelsDevPrice | None:
        """The one entry every provider listing ``model`` agrees on in price."""
        key = model.casefold()
        candidates = self._by_model_id.get(key) or self._by_canonical.get(key) or []
        priced = [c for c in candidates if c.priced]
        candidates = priced or candidates
        if not candidates or len({c._price_signature() for c in candidates}) != 1:
            return None
        vendor = (candidates[0].canonical_model_id or "").partition("/")[0]
        return min(candidates, key=lambda c: (c.provider_id != vendor, c.provider_id, c.model_id))


@dataclass(frozen=True, slots=True)
class PriceGeneration:
    """An index and the instant it took effect."""

    effective_at: datetime
    index: ModelsDevPriceIndex


def select_generation(generations: Sequence[PriceGeneration], as_of: datetime) -> PriceGeneration | None:
    """The latest generation in effect at ``as_of``, the oldest before the first."""
    if not generations:
        return None
    ordered = sorted(generations, key=lambda g: g.effective_at)
    position = bisect_right([g.effective_at for g in ordered], as_of)
    return ordered[max(position - 1, 0)]


def resolve_as_of(
    generations: Sequence[PriceGeneration],
    as_of: datetime,
    provider: str | None,
    model: str,
    implementation: str | None = None,
) -> ModelsDevPrice | None:
    """Resolve ``model`` against the generation in effect at ``as_of``."""
    generation = select_generation(generations, as_of)
    return None if generation is None else generation.index.resolve(provider, model, implementation)
