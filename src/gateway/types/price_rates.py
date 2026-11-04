"""A stored rate as the ``cost`` block models.dev writes, and the six rate columns as its text view.

A price row's ``rates`` holds one ``cost`` block in the document's own shape:
each modality with its ``input`` and ``output`` price, each price with its
``unit``, ``cache``, ``reasoning`` and ``overrides``, ``web_search`` beside the
modalities, and ``multipliers`` beside them. A price in it is per what
:data:`~gateway.types.price_unit.PRICE_PER` says for its unit, as published.
The six columns (the four per-million rates, ``pricing_tiers`` and ``unit``) are
the one view of it an older reader has. :func:`rates_from_columns` writes a
block for a row given as columns, into the slot :data:`_SLOTS` gives the unit,
and :func:`columns_from_rates` reads the columns back out of any block, the
text prices first. A rate's numbers are written exactly and read as
``Decimal``, so money never rounds through a float; :func:`dump_rates` and
:func:`load_rates` are that encoding.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from types import MappingProxyType
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from gateway.types.models_dev import (
    ModelsDevCachePrices,
    ModelsDevCost,
    ModelsDevModalityCost,
    ModelsDevOverride,
    ModelsDevPricing,
    ModelsDevUnitPrice,
    Number,
    direction_cost,
    parse_pricing,
    pricing_document,
)
from gateway.types.price_unit import PRICE_PER, PRICE_UNITS, PriceUnit

# The field names of a ``pricing_tiers`` entry: the column shape, which a tier is spelled in.
TIER_THRESHOLD = "min_input_tokens"
TIER_RATES = {
    "input": "input_price_per_million",
    "output": "output_price_per_million",
    "cache_read": "cache_read_price_per_million",
    "cache_write": "cache_write_price_per_million",
}

RATES_DESCRIPTION = (
    "Every price of the row as a models.dev cost block: each modality's input and output price with its unit, "
    "cache and overrides, web_search beside the modalities, and multipliers. A price is per its unit as "
    "published (one image, one million tokens, one thousand searches). Given instead of the per-million fields, "
    "which are then read out of it; a row always reads back with both."
)

_MILLION = Decimal(1_000_000)
_COLUMN_PLACES = Decimal("0.00000001")
_COLUMN_LIMIT = Decimal(10) ** 10


@dataclass(frozen=True)
class RateColumns:
    """The six rate columns of a price row: the text view of its ``rates``."""

    unit: PriceUnit
    input_price_per_million: Decimal
    output_price_per_million: Decimal
    cache_read_price_per_million: Decimal | None = None
    cache_write_price_per_million: Decimal | None = None
    pricing_tiers: tuple[Mapping[str, Any], ...] = ()


@dataclass(frozen=True)
class _Slot:
    """Where a unit's column rates sit in a block: ``modality`` ``None`` is the ``web_search`` field."""

    modality: str | None
    given: str
    made: str | None


# The slot a rate given as columns is written into. A per-image price is an
# image made, so the input column of an ``image`` rate is the output price and
# its output column has no slot. ``second`` and ``minute`` are stored ahead of
# their meter; audio is the modality that reports them.
_SLOTS: dict[PriceUnit, _Slot] = {
    "token": _Slot("text", "input", "output"),
    "character": _Slot("text", "input", "output"),
    "request": _Slot("text", "input", "output"),
    "image": _Slot("image", "output", None),
    "megapixel": _Slot("image", "output", None),
    "second": _Slot("audio", "input", "output"),
    "minute": _Slot("audio", "input", "output"),
    "web_search": _Slot(None, "input", None),
}

# The order the columns are read out of a block: the text prices first, then
# the one other slot that is priced.
_READ_ORDER: tuple[_Slot, ...] = (
    _Slot("text", "input", "output"),
    _Slot("image", "output", None),
    _Slot("audio", "input", "output"),
    _Slot("video", "output", None),
    _Slot(None, "input", None),
)


def rates_from_columns(columns: RateColumns) -> dict[str, Any]:
    """The ``cost`` block that says what ``columns`` say, in the slot the unit chooses.

    A per-million column becomes a price per the unit's published quantity.
    Each tier becomes an override on context, and the cache rates the input
    price's ``cache``. The output column of a one-price slot (``image``,
    ``megapixel``, ``web_search``) prices nothing that slot reports, so it is
    not carried.
    """
    slot = _SLOTS[columns.unit]
    per = Decimal(PRICE_PER[columns.unit]) / _MILLION
    given_overrides: list[ModelsDevOverride] = []
    made_overrides: list[ModelsDevOverride] = []
    for tier in columns.pricing_tiers:
        when: Mapping[str, object] = {"context": {"gte": tier[TIER_THRESHOLD]}}
        cache = _cache(tier.get(TIER_RATES["cache_read"]), tier.get(TIER_RATES["cache_write"]), per)
        price = _published(tier.get(TIER_RATES["input"]), per)
        if price is not None or cache is not None:
            given_overrides.append(ModelsDevOverride(when=when, price=price, cache=cache))
        if (price := _published(tier.get(TIER_RATES["output"]), per)) is not None:
            made_overrides.append(ModelsDevOverride(when=when, price=price))
    given = ModelsDevUnitPrice(
        price=columns.input_price_per_million * per,
        unit=columns.unit,
        cache=_cache(columns.cache_read_price_per_million, columns.cache_write_price_per_million, per),
        overrides=tuple(given_overrides),
    )
    if slot.modality is None:
        return pricing_document(ModelsDevPricing(modalities=MappingProxyType({}), web_search=given))
    prices = {slot.given: given}
    if slot.made is not None:
        prices[slot.made] = ModelsDevUnitPrice(
            price=columns.output_price_per_million * per, unit=columns.unit, overrides=tuple(made_overrides)
        )
    modality = ModelsDevModalityCost(input=prices.get("input"), output=prices.get("output"))
    return pricing_document(ModelsDevPricing(modalities=MappingProxyType({slot.modality: modality})))


def columns_from_rates(document: Mapping[str, Any]) -> RateColumns:
    """The six columns a ``cost`` block gives: its text prices, else the one other slot it prices.

    The unit is the price's own, ``token`` where it gives none, and each price
    is scaled back to per million of it. The overrides on context become the
    tiers, each carrying the base rates it does not replace.

    Raises:
        ValueError: The block cannot be read, prices nothing a column can hold,
            prices text input and output in different units, or holds a rate
            no column can.
    """
    pricing, issues = parse_pricing(document, units=PRICE_UNITS, path="rates")
    if issues.count:
        raise ValueError(f"rates cannot be read: {'; '.join(issues.examples)}")
    if issues.unknown_units:
        raise ValueError(f"rates name a unit not known here: {', '.join(issues.unknown_units)}")
    for slot in _READ_ORDER:
        given, made = _prices_in(pricing, slot)
        if given is None and made is None:
            continue
        unit = _unit_of(given if given is not None else made)
        if given is not None and made is not None and _unit_of(made) != unit:
            raise ValueError(f"rates price {slot.modality} input and output in different units")
        return _columns(direction_cost(given, made), unit)
    raise ValueError("rates price nothing a rate column can hold: no text, image, audio, video or web_search price")


def dump_rates(document: Mapping[str, Any]) -> str:
    """``document`` as JSON, every number written exactly and in the order given."""
    return _dump(document)


def load_rates(text: str) -> dict[str, Any]:
    """The JSON object ``text`` holds, its numbers as ``Decimal``.

    Raises:
        ValueError: ``text`` is not a JSON object.
    """
    document = json.loads(text, parse_float=Decimal)
    if not isinstance(document, dict):
        raise ValueError("rates are not a JSON object")
    return document


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CacheRates(_Strict):
    """The cache prices of a price, in its unit."""

    read: float | None = Field(default=None, ge=0, description="Price of a cached read.")
    write: float | dict[str, float] | None = Field(
        default=None, description="Price of a cache write: one price, or one per TTL with the default TTL first."
    )
    storage: float | None = Field(default=None, ge=0, description="Price of cache storage.")
    explicit: "CacheRates | None" = Field(default=None, description="The prices of explicit caching, where apart.")

    @model_validator(mode="after")
    def validate_writes(self) -> Self:
        if isinstance(self.write, dict) and any(price < 0 for price in self.write.values()):
            raise ValueError("a cache write price is not negative")
        if isinstance(self.write, float) and self.write < 0:
            raise ValueError("a cache write price is not negative")
        return self


class RateOverride(_Strict):
    """The prices that replace a price's own when a condition holds."""

    when: dict[str, Any] = Field(description="The condition, as models.dev writes it.")
    price: float | None = Field(default=None, ge=0)
    reasoning: float | None = Field(default=None, ge=0)
    cache: CacheRates | None = None


class UnitRate(_Strict):
    """One price: of a modality's input or output, or of web search."""

    price: float | None = Field(default=None, ge=0, description="USD per the unit's published quantity.")
    unit: PriceUnit | None = Field(default=None, description="What the price is per; token when omitted.")
    reasoning: float | None = Field(default=None, ge=0, description="Price of reasoning tokens, where apart.")
    cache: CacheRates | None = None
    overrides: list[RateOverride] = Field(default_factory=list)


class ModalityRates(_Strict):
    """The input and output prices of one modality."""

    input: UnitRate | None = None
    output: UnitRate | None = None


class RateMultiplier(_Strict):
    """A factor applied to every price, or to the prices it names, when a condition holds."""

    when: dict[str, Any] = Field(description="The condition, as models.dev writes it.")
    factor: float | dict[str, Any] = Field(description="One factor for every price, or an object naming the prices.")

    @model_validator(mode="after")
    def validate_factor(self) -> Self:
        if isinstance(self.factor, float) and self.factor < 0:
            raise ValueError("a multiplier is not negative")
        return self


class RatesDocument(BaseModel):
    """A ``cost`` block on the wire: each other key is a modality (``text``, ``image``, ``audio``, ``video``)."""

    model_config = ConfigDict(extra="allow")
    __pydantic_extra__: dict[str, ModalityRates]

    web_search: UnitRate | None = None
    multipliers: list[RateMultiplier] = Field(default_factory=list)

    def document(self) -> dict[str, Any]:
        """The block as given, without the fields left out."""
        return self.model_dump(exclude_none=True, exclude_defaults=True)


def _prices_in(pricing: ModelsDevPricing, slot: _Slot) -> tuple[ModelsDevUnitPrice | None, ModelsDevUnitPrice | None]:
    if slot.modality is None:
        return pricing.web_search, None
    prices = pricing.modalities.get(slot.modality)
    if prices is None:
        return None, None
    given = getattr(prices, slot.given)
    made = None if slot.made is None else getattr(prices, slot.made)
    return given, made


def _unit_of(price: ModelsDevUnitPrice | None) -> PriceUnit:
    unit = "token" if price is None or price.unit is None else price.unit
    assert unit in PRICE_UNITS
    return unit


def _columns(cost: ModelsDevCost, unit: PriceUnit) -> RateColumns:
    per = Decimal(PRICE_PER[unit]) / _MILLION
    return RateColumns(
        unit=unit,
        input_price_per_million=_column(cost.rates.input, per) or Decimal(0),
        output_price_per_million=_column(cost.rates.output, per) or Decimal(0),
        cache_read_price_per_million=_column(cost.rates.cache_read, per),
        cache_write_price_per_million=_column(cost.rates.cache_write, per),
        pricing_tiers=tuple(
            {
                TIER_THRESHOLD: tier.size,
                **{
                    TIER_RATES[name]: float(rate)
                    for name in TIER_RATES
                    if (rate := _column(getattr(tier.rates, name), per)) is not None
                },
            }
            for tier in cost.tiers
        ),
    )


def _column(price: Number | None, per: Decimal) -> Decimal | None:
    """A published price as a per-million column, rounded as the column is; ``None`` stays absent."""
    if price is None:
        return None
    column = (Decimal(str(price)) / per).quantize(_COLUMN_PLACES)
    if not 0 <= column < _COLUMN_LIMIT:
        raise ValueError(f"a rate of {price} per unit does not fit a rate column")
    return column


def _published(column: object, per: Decimal) -> Decimal | None:
    """A per-million column as a published price."""
    return None if column is None else Decimal(str(column)) * per


def _cache(read: object, write: object, per: Decimal) -> ModelsDevCachePrices | None:
    if read is None and write is None:
        return None
    return ModelsDevCachePrices(read=_published(read, per), write=_published(write, per))


def _dump(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return _number_text(Decimal(repr(value)))
    if isinstance(value, Decimal):
        return _number_text(value)
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, Mapping):
        return "{" + ",".join(f"{json.dumps(str(key))}:{_dump(item)}" for key, item in value.items()) + "}"
    if isinstance(value, Sequence):
        return "[" + ",".join(_dump(item) for item in value) + "]"
    raise TypeError(f"rates cannot hold a {type(value).__name__}")


def _number_text(number: Decimal) -> str:
    if not number.is_finite():
        raise ValueError("rates cannot hold a number that is not finite")
    return format(number.normalize(), "f")
