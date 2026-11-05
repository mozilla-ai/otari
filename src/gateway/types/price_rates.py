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
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from types import MappingProxyType
from typing import Any, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    field_validator,
    model_serializer,
    model_validator,
)

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

# The modalities a block prices in, as models.dev prices them; a test pins the bundled catalog to this list.
MODALITIES = ("text", "image", "audio", "video")

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


def rates_with_columns(block: Mapping[str, Any] | None, columns: RateColumns) -> dict[str, Any]:
    """``block`` with the slot ``columns``' unit chooses rewritten from them, and every other price kept.

    A writer that moves the columns alone moves the text price of a row whose
    block also holds an image or a web search price, so those stay. Where the
    rest of the block would read back as a different price than the columns
    (the unit moved to another slot), the fresh block stands alone.
    """
    fresh = rates_from_columns(columns)
    if not block:
        return fresh
    merged = dict(block)
    for key, prices in fresh.items():
        old = block.get(key)
        if key == "web_search":
            merged[key] = _merged_price(old if isinstance(old, Mapping) else None, prices)
        else:
            kept = dict(old) if isinstance(old, Mapping) else {}
            for direction, price in prices.items():
                kept[direction] = _merged_price(kept.get(direction), price)
            merged[key] = kept
    try:
        read = columns_from_rates(merged)
    except ValueError:
        return fresh
    same = all(getattr(read, name) == getattr(columns, name) for name in _RATE_FIELDS)
    return merged if same and read.unit == columns.unit else fresh


def _merged_price(old: Mapping[str, Any] | None, fresh: Mapping[str, Any]) -> dict[str, Any]:
    """``old`` with what the columns say rewritten: price, unit, cache read and write, and the context overrides.

    The rest of ``old`` is in its unit, so where the columns moved the price to
    another unit the fresh price stands alone.
    """
    if not old or (old.get("unit") or "token") != (fresh.get("unit") or "token"):
        return dict(fresh)
    merged = {key: value for key, value in old.items() if key not in ("price", "unit")}
    merged.update({key: fresh[key] for key in ("price", "unit") if key in fresh})
    cache = {key: value for key, value in (old.get("cache") or {}).items() if key not in ("read", "write")}
    cache.update(fresh.get("cache") or {})
    merged.pop("cache", None)
    if cache:
        merged["cache"] = cache
    overrides = [o for o in old.get("overrides") or [] if not _on_context(o)] + list(fresh.get("overrides") or [])
    merged.pop("overrides", None)
    if overrides:
        merged["overrides"] = overrides
    return merged


def _on_context(override: Mapping[str, Any]) -> bool:
    when = override.get("when")
    return isinstance(when, Mapping) and set(when) == {"context"}


_RATE_FIELDS = tuple(TIER_RATES.values())


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
    return columns_from_pricing(pricing)


def columns_from_pricing(pricing: ModelsDevPricing) -> RateColumns:
    """The six columns a parsed ``cost`` block gives, as :func:`columns_from_rates` reads them from a document.

    Raises:
        ValueError: The block prices nothing a column can hold, prices text
            input and output in different units, or holds a rate no column can.
    """
    for slot in _READ_ORDER:
        given, made = _prices_in(pricing, slot)
        priced = [price for price in (given, made) if price is not None and price.price is not None]
        if not priced:
            continue
        unit = _unit_of(priced[0])
        # An unpriced object that names a unit or carries a cache price is in a unit too; an empty one is not.
        in_a_unit = [p for p in (given, made) if p is not None and (p.price is not None or p.unit or p.cache)]
        if any(_unit_of(price) != unit for price in in_a_unit):
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
    document = json.loads(text, parse_float=Decimal, parse_int=Decimal)
    if not isinstance(document, dict):
        raise ValueError("rates are not a JSON object")
    return document


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _numbers(value: Any) -> Any:
    """``value`` with each ``Decimal`` a row holds as the JSON number it was, for the untyped part of a block."""
    if isinstance(value, dict):
        return {key: _numbers(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_numbers(item) for item in value]
    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral_value() else float(value)
    return value


def _a_price(value: Any) -> Any:
    """``value`` where every leaf is a finite number, zero or more; the refusal names the first that is not."""
    if isinstance(value, dict):
        for item in value.values():
            _a_price(item)
    elif isinstance(value, bool) or not isinstance(value, (int, float, Decimal)) or not _finite(value) or value < 0:
        raise ValueError(f"a price is a finite number, zero or more, not {value!r}")
    return value


def _finite(value: int | float | Decimal) -> bool:
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


class CacheRates(_Strict):
    """The cache prices of a price, in its unit."""

    read: float | None = Field(default=None, ge=0, allow_inf_nan=False, description="Price of a cached read.")
    write: float | dict[str, float] | None = Field(
        default=None, description="Price of a cache write: one price, or one per TTL with the default TTL first."
    )
    storage: float | None = Field(default=None, ge=0, allow_inf_nan=False, description="Price of cache storage.")
    explicit: "CacheRates | None" = Field(default=None, description="The prices of explicit caching, where apart.")

    @field_validator("write", mode="before")
    @classmethod
    def _writes_are_prices(cls, value: Any) -> Any:
        return _a_price(value)


class RateOverride(_Strict):
    """The prices that replace a price's own when a condition holds."""

    when: dict[str, Any] = Field(description="The condition, as models.dev writes it.")
    price: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    reasoning: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    cache: CacheRates | None = None

    @field_validator("when", mode="before")
    @classmethod
    def _as_numbers(cls, value: Any) -> Any:
        return _numbers(value)


class UnitRate(_Strict):
    """One price: of a modality's input or output, or of web search."""

    price: float | None = Field(
        default=None, ge=0, allow_inf_nan=False, description="USD per the unit's published quantity."
    )
    unit: PriceUnit | None = Field(default=None, description="What the price is per; token when omitted.")
    reasoning: float | None = Field(
        default=None, ge=0, allow_inf_nan=False, description="Price of reasoning tokens, where apart."
    )
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

    @field_validator("when", "factor", mode="before")
    @classmethod
    def _as_numbers(cls, value: Any) -> Any:
        return _numbers(value)

    @field_validator("factor", mode="before")
    @classmethod
    def _factors_are_prices(cls, value: Any) -> Any:
        return _a_price(value)


class RatesDocument(BaseModel):
    """A ``cost`` block on the wire: each other key is a modality (``text``, ``image``, ``audio``, ``video``)."""

    # Untyped in the schema on purpose: a typed index signature would hold
    # ``web_search`` and ``multipliers`` to the modality shape in a generated
    # client. Each modality is still read into :class:`ModalityRates` below.
    model_config = ConfigDict(extra="allow")
    __pydantic_extra__: dict[str, Any]

    web_search: UnitRate | None = None
    multipliers: list[RateMultiplier] = Field(default_factory=list)

    @model_validator(mode="after")
    def _modalities_have_the_shape(self) -> Self:
        for modality, prices in (self.__pydantic_extra__ or {}).items():
            if not isinstance(prices, ModalityRates):
                self.__pydantic_extra__[modality] = ModalityRates.model_validate(prices)
        return self

    @model_serializer(mode="wrap")
    def _as_given(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        """The block as the document writes it: a field left out stays out, on the wire as in a row."""
        document: dict[str, Any] = _without_absent(handler(self))
        return document

    def document(self) -> dict[str, Any]:
        """The block as given, without the fields left out."""
        return self.model_dump()


def _without_absent(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            # A condition is untyped and kept as given, a null in it included.
            key: item if key == "when" else _without_absent(item)
            for key, item in value.items()
            if item is not None and not (key in ("overrides", "multipliers") and item == [])
        }
    if isinstance(value, list):
        return [_without_absent(item) for item in value]
    return value


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
    try:
        column = (Decimal(str(price)) / per).quantize(_COLUMN_PLACES)
    except InvalidOperation:
        raise ValueError(f"a rate of {price} per unit does not fit a rate column") from None
    if not 0 <= column < _COLUMN_LIMIT:
        raise ValueError(f"a rate of {price} per unit does not fit a rate column")
    if column == 0 and Decimal(str(price)) > 0:
        raise ValueError(f"a rate of {price} per unit is too small for a rate column, which holds eight decimals")
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
