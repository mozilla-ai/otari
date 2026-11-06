"""Which meter a rate prices, and what one successful attempt costs at it.

The rate's ``unit`` chooses the meter, through one table, :data:`METERS`. A
``token`` rate prices tokens, an ``image`` rate an ``images`` count, and a
``request`` rate a ``requests`` or ``search_units`` count, or one request for
an attempt that reported neither. An endpoint that bills under one slot of
the rate's block (``image`` output, ``audio`` input) is priced at that slot
first: a slot in a measured unit prices the quantity the attempt measured in
it, and nothing else, while a slot in tokens or requests, or none, falls to
the table. Settlement prices through :func:`price_usage`, and the
``require_pricing`` gate asks :func:`can_price` the same rule, so a rate the
gate admits is one settlement prices. A unit with no row prices nothing.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from decimal import Decimal
from math import ceil, isfinite

from any_llm.types.completion import CompletionUsage

from gateway.core.metered_pricing import (
    RATE_FIELDS,
    BillableUsage,
    ChargeLine,
    Direction,
    calculate_metered_cost,
    meter_cost,
    price_billable_usage,
    price_request,
    priced_per_request,
    quantize_cost,
    to_decimal,
)
from gateway.models.pricing import MAX_REPORTED_UNITS, UNIT_METERS, ModelPricing
from gateway.services.pricing._charge import UsageCharge
from gateway.services.pricing._slot_pricing import Quantities, block_is_free, price_slot, slot_rate
from gateway.types.price_unit import PriceUnit

__all__ = [
    "MEASURED_UNITS",
    "METERS",
    "Meter",
    "Slot",
    "UsageCharge",
    "billed_search_units",
    "can_price",
    "estimate_settles",
    "has_completion_meter",
    "meter_for",
    "price_completion",
    "price_usage",
    "request_rate",
    "validated_units",
]

_REQUESTS, _SEARCH_UNITS, _IMAGES = UNIT_METERS

# The slot of a rate's block an endpoint bills under: a modality and a direction.
Slot = tuple[str, Direction]

# The units an endpoint measures a quantity in, rather than reports as tokens or a count.
MEASURED_UNITS: frozenset[PriceUnit] = frozenset({"image", "character", "megapixel", "second", "minute", "web_search"})


@dataclass(frozen=True)
class Meter:
    """One row of :data:`METERS`: how a rate in one unit reads an attempt and prices it.

    ``reads`` names the counts of a report the meter prices, and ``tokens``
    whether it prices a report of tokens. ``needs_report`` says the meter prices
    nothing without one of those; the request meter does not, settling one
    request for an attempt that reported neither. ``settles_completion`` says a
    completion can be settled at the rate.
    """

    reads: tuple[str, ...]
    tokens: bool
    needs_report: bool
    settles_completion: bool
    charge: Callable[[ModelPricing, dict[str, int], BillableUsage | None], UsageCharge | None]

    def admits(self, meter: str | None, *, reports_tokens: bool) -> bool:
        """Whether an attempt that reports ``meter`` and, when ``reports_tokens``, tokens meets this meter."""
        return not self.needs_report or meter in self.reads or (self.tokens and reports_tokens)


def validated_units(raw: Mapping[str, object]) -> dict[str, int]:
    """The known meters of ``raw``, each with a whole count from zero to ``MAX_REPORTED_UNITS``.

    A key that names no meter is dropped, so a newer reporter's meter is ignored
    rather than refused.

    Raises:
        ValueError: A known meter carries a count that is not such a number.
    """
    counts: dict[str, int] = {}
    for meter, count in raw.items():
        if meter not in UNIT_METERS:
            continue
        if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= MAX_REPORTED_UNITS:
            raise ValueError(f"The {meter} count must be a whole number from 0 to {MAX_REPORTED_UNITS}")
        counts[meter] = count
    return counts


def price_usage(
    rate: ModelPricing | None,
    *,
    tokens: BillableUsage | None,
    units: Mapping[str, object],
    quantities: Quantities | None = None,
    slot: Slot | None = None,
) -> UsageCharge | None:
    """What one successful attempt costs at ``rate``, or ``None`` when ``rate`` prices nothing it reported.

    ``slot`` names the slot of the rate's block the endpoint bills under, and
    ``quantities`` what it measured, in each unit it can. A slot priced in a
    measured unit prices the quantity in that unit, or nothing when the
    attempt did not measure it. A rate in tokens or requests prices what the
    attempt reported, as :data:`METERS` says.

    Raises:
        ValueError: ``units`` carries an invalid count (see :func:`validated_units`).
    """
    counts = validated_units(units)
    if rate is None:
        return None
    if slot is not None and _measured_slot(rate, slot) is not None:
        return price_slot(rate, *slot, quantities or {})
    row = meter_for(rate)
    return row.charge(rate, counts, tokens) if row is not None else None


def has_completion_meter(rate: ModelPricing | None) -> bool:
    """Whether a completion can be settled at ``rate``: per request, or by the token meters.

    A rate in another unit (``second``, ``minute``) prices a quantity a
    completion does not measure, so a completion at it is unpriced: refused
    under ``require_pricing``, and otherwise served and recorded with no cost.
    It is never billed as tokens.
    """
    if rate is None:
        return False
    row = meter_for(rate)
    return row is not None and row.settles_completion


def estimate_settles(rate: ModelPricing | None) -> bool:
    """Whether a reservation's estimate can settle a completion that reported no usage.

    It can for an unpriced model, whose estimate is zero, and for any rate a
    completion meter reads. A rate in a unit no completion meter reads held
    nothing and settles nothing, so the row keeps no cost and the needs-pricing
    view still lists it.
    """
    return rate is None or has_completion_meter(rate)


def price_completion(
    rate: ModelPricing, usage: CompletionUsage | None, *, succeeded: bool
) -> tuple[Decimal, dict[str, int], list[ChargeLine]] | None:
    """What one completion costs at ``rate``, or ``None`` when ``rate`` prices nothing it reported.

    A per-request rate settles one request for a call that succeeded, whatever
    the tokens, and nothing for one that failed. Any other rate the token meters
    read settles the reported usage, a failed call's included. A rate in a unit
    no completion meter reads (``second``, ``minute``) prices nothing, so the
    row records its tokens with no cost rather than billing them as tokens.
    """
    if priced_per_request(rate):
        return price_request(rate) if succeeded else None
    if usage is None or not has_completion_meter(rate):
        return None
    return calculate_metered_cost(rate, usage)


def can_price(
    rate: ModelPricing | None,
    *,
    meter: str | None,
    reports_tokens: bool,
    slot: Slot | None = None,
    measures: frozenset[PriceUnit] = frozenset(),
) -> bool:
    """Whether ``rate`` could price an attempt that reports ``meter`` and, when ``reports_tokens``, tokens.

    ``slot`` and ``measures`` say which slot the endpoint bills under and which
    units it measures, so a slot in a measured unit is admitted only when the
    endpoint measures that unit. The ``require_pricing`` gate asks this before
    dispatch, so a rate the endpoint's report cannot meet is refused rather
    than settled unpriced.
    """
    input_rate = to_decimal(rate.input_price_per_million) if rate is not None else None
    if rate is None or input_rate is None:
        return False
    if slot is not None and (found := _measured_slot(rate, slot)) is not None:
        return found in measures
    row = meter_for(rate)
    if row is None:
        return False
    # A free token rate settles free whatever the attempt reports, so it prices any attempt.
    return row.admits(meter, reports_tokens=reports_tokens) or (row.tokens and _free(rate, input_rate))


def _measured_slot(rate: ModelPricing, slot: Slot) -> PriceUnit | None:
    """The measured unit ``rate`` prices ``slot`` in, or ``None`` when the slot is in tokens, requests, or absent."""
    found = slot_rate(rate, *slot)
    return found.unit if found is not None and found.unit in MEASURED_UNITS else None


def request_rate(rate: ModelPricing | None) -> Decimal:
    """USD for one request at a per-request rate, or zero when the rate is absent or unusable."""
    return (_request_rate(rate) if rate is not None else None) or Decimal(0)


def billed_search_units(billed_units: Mapping[str, float] | None) -> int:
    """How many search units a rerank request bills, at least one.

    Reported under ``meta.billed_units`` by providers that bill rerank per query
    batch rather than per token. A fraction rounds up, because the provider
    billed it, and a request that reports none bills one unit: the request itself.
    """
    reported = billed_units.get("search_units") if billed_units else None
    if reported is None or not isfinite(reported) or reported <= 0:
        return 1
    return min(ceil(reported), MAX_REPORTED_UNITS)


def _request_rate(rate: ModelPricing) -> Decimal | None:
    # ``input_price_per_million`` is USD per million requests.
    per_million = to_decimal(rate.input_price_per_million)
    return meter_cost(1, per_million) if per_million is not None else None


def _image_rate(rate: ModelPricing) -> Decimal | None:
    # ``input_price_per_million`` is USD per million images, as a request rate is per million requests.
    per_million = to_decimal(rate.input_price_per_million)
    return meter_cost(1, per_million) if per_million is not None else None


def _per_unit(meter: str, count: int, unit_rate: Decimal | None) -> UsageCharge | None:
    # An unusable rate prices nothing: settling it at $0 would serve the attempt free.
    if unit_rate is None:
        return None
    cost = quantize_cost(count * unit_rate)
    if not cost:
        return UsageCharge(Decimal(0), {}, [])
    # A per-request line names its meter in the singular, as the cost core's request line does.
    line_meter = "request" if meter == _REQUESTS else meter
    line: ChargeLine = {"meter": line_meter, "units": count, "unit_rate": float(unit_rate), "cost": float(cost)}
    return UsageCharge(cost, {meter: count}, [line])


def _charge_tokens(rate: ModelPricing, _: dict[str, int], tokens: BillableUsage | None) -> UsageCharge | None:
    input_rate = to_decimal(rate.input_price_per_million)
    if input_rate is None:
        return None
    if tokens is None or (tokens.total_input_tokens <= 0 and tokens.completion_tokens <= 0):
        # Nothing times a zero rate is nothing, so a free model settles free with no tokens reported.
        return UsageCharge(Decimal(0), {}, []) if _free(rate, input_rate) else None
    cost, meters, lines = price_billable_usage(rate, tokens)
    return UsageCharge(cost, meters, lines)


def _free(rate: ModelPricing, input_rate: Decimal) -> bool:
    """Whether every price ``rate`` holds is zero: the base and cache rates, each tier's, and every slot of its block.

    The columns are the text slot alone, so the block is read too: a rate whose
    text is free but whose audio output is not prices an attempt that made audio.
    """
    if input_rate != 0:
        return False
    others = [to_decimal(getattr(rate, name)) for name in RATE_FIELDS if name != "input_price_per_million"]
    for tier in getattr(rate, "pricing_tiers", None) or []:
        others.extend(to_decimal(tier.get(name)) for name in RATE_FIELDS)
    return all(price is None or price == 0 for price in others) and block_is_free(getattr(rate, "rates", None))


def _charge_requests(rate: ModelPricing, counts: dict[str, int], _: BillableUsage | None) -> UsageCharge | None:
    meter = next((m for m in (_REQUESTS, _SEARCH_UNITS) if m in counts), _REQUESTS)
    return _per_unit(meter, counts.get(meter, 1), _request_rate(rate))


def _charge_images(rate: ModelPricing, counts: dict[str, int], _: BillableUsage | None) -> UsageCharge | None:
    return _per_unit(_IMAGES, counts[_IMAGES], _image_rate(rate)) if _IMAGES in counts else None


# The meter each unit prices by. A unit with no row is a measured one, priced
# only through the slot an endpoint bills under; a new unit is one row here or
# one measure there. An ``image`` rate on a completion is read in tokens, as it
# always was.
METERS: dict[PriceUnit, Meter] = {
    "token": Meter(reads=(), tokens=True, needs_report=True, settles_completion=True, charge=_charge_tokens),
    "request": Meter(
        reads=(_REQUESTS, _SEARCH_UNITS),
        tokens=False,
        needs_report=False,
        settles_completion=True,
        charge=_charge_requests,
    ),
    "image": Meter(reads=(_IMAGES,), tokens=False, needs_report=True, settles_completion=True, charge=_charge_images),
}


def meter_for(rate: ModelPricing) -> Meter | None:
    """The meter ``rate``'s unit prices by, or ``None`` when no meter reads that unit."""
    return METERS.get(rate.unit or "token")
