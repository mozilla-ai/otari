"""Which meter a rate prices, and what one successful attempt costs at it.

The rate's ``unit`` chooses the meter, through one table, :data:`METERS`. A
``token`` rate prices tokens, an ``image`` rate an ``images`` count, and a
``request`` rate a ``requests`` or ``search_units`` count, or one request for
an attempt that reported neither. The one exception is a ``token`` rate on an
endpoint billed per image or per request, which is read in that endpoint's
unit, as the configuration guide documented before the rate's unit decided.
Settlement prices through :func:`price_usage`, and the ``require_pricing``
gate asks :func:`can_price` the same table, so a rate the gate admits is one
settlement prices. A unit with no row prices nothing.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from decimal import Decimal
from math import ceil, isfinite

from any_llm.types.completion import CompletionUsage

from gateway.core.metered_pricing import (
    BillableUsage,
    ChargeLine,
    calculate_metered_cost,
    meter_cost,
    price_billable_usage,
    price_request,
    priced_per_request,
    quantize_cost,
    to_decimal,
)
from gateway.log_config import logger
from gateway.models.pricing import MAX_REPORTED_UNITS, UNIT_METERS, ModelPricing
from gateway.types.price_unit import PriceUnit

_REQUESTS, _SEARCH_UNITS, _IMAGES = UNIT_METERS

# The (model key, meter) pairs read in the endpoint's unit that have already been warned about.
_LEGACY_WARNED: set[tuple[str, str]] = set()


@dataclass(frozen=True)
class UsageCharge:
    """What an attempt settled at: the cost, rounded once, and the meters and charge lines that explain it.

    A free attempt carries no meter and no line, so a breakdown never explains a
    charge that did not happen.
    """

    cost: Decimal
    meters: dict[str, int]
    lines: list[ChargeLine]


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
) -> UsageCharge | None:
    """What one successful attempt costs at ``rate``, or ``None`` when ``rate`` prices nothing it reported.

    A ``token`` rate on an attempt that reported images or requests is read in
    that unit, warning once per model and meter that the rate should name it.

    Raises:
        ValueError: ``units`` carries an invalid count (see :func:`validated_units`).
    """
    counts = validated_units(units)
    if rate is None:
        return None
    row = meter_for(rate)
    return row.charge(rate, counts, tokens) if row is not None else None


def has_completion_meter(rate: ModelPricing | None) -> bool:
    """Whether a completion can be settled at ``rate``: per request, or by the token meters.

    A rate in another unit (``second``, ``minute``) is stored ahead of its meter,
    so a completion at it is unpriced: refused under ``require_pricing``, and
    otherwise served and recorded with no cost. It is never billed as tokens.
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


def can_price(rate: ModelPricing | None, *, meter: str | None, reports_tokens: bool) -> bool:
    """Whether ``rate`` could price an attempt that reports ``meter`` and, when ``reports_tokens``, tokens.

    The ``require_pricing`` gate asks this before dispatch, so a rate the
    endpoint's report cannot meet is refused rather than settled unpriced.
    """
    if rate is None or to_decimal(rate.input_price_per_million) is None:
        return False
    row = meter_for(rate)
    return row is not None and row.admits(meter, reports_tokens=reports_tokens)


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


def _legacy_image_rate(rate: ModelPricing) -> Decimal | None:
    # A token rate on image generation was documented as raw USD per image.
    return to_decimal(rate.input_price_per_million)


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


def _charge_tokens(rate: ModelPricing, counts: dict[str, int], tokens: BillableUsage | None) -> UsageCharge | None:
    for meter, unit_rate in ((_IMAGES, _legacy_image_rate), (_REQUESTS, _request_rate)):
        if meter in counts:
            _warn_legacy(rate, meter)
            return _per_unit(meter, counts[meter], unit_rate(rate))
    if to_decimal(rate.input_price_per_million) is None:
        return None
    if tokens is None or (tokens.total_input_tokens <= 0 and tokens.completion_tokens <= 0):
        return None
    cost, meters, lines = price_billable_usage(rate, tokens)
    return UsageCharge(cost, meters, lines)


def _charge_requests(rate: ModelPricing, counts: dict[str, int], _: BillableUsage | None) -> UsageCharge | None:
    meter = next((m for m in (_REQUESTS, _SEARCH_UNITS) if m in counts), _REQUESTS)
    return _per_unit(meter, counts.get(meter, 1), _request_rate(rate))


def _charge_images(rate: ModelPricing, counts: dict[str, int], _: BillableUsage | None) -> UsageCharge | None:
    return _per_unit(_IMAGES, counts[_IMAGES], _image_rate(rate)) if _IMAGES in counts else None


# The meter each unit prices by. A unit with no row (``second``, ``minute``) is
# stored ahead of its meter and prices nothing; a new unit is one row here.
# An ``image`` rate on a completion is read in tokens, as it always was.
METERS: dict[PriceUnit, Meter] = {
    "token": Meter(
        reads=(_IMAGES, _REQUESTS), tokens=True, needs_report=True, settles_completion=True, charge=_charge_tokens
    ),
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


def _warn_legacy(rate: ModelPricing, meter: str) -> None:
    key = (rate.model_key, meter)
    if key in _LEGACY_WARNED:
        return
    _LEGACY_WARNED.add(key)
    if meter == _IMAGES:
        # An images rate is per million images, where this one is read as raw USD per image.
        fix = "set its unit to image and multiply its input price by 1,000,000"
    else:
        fix = "set its unit to request, keeping its price"
    logger.warning(
        "Pricing %s is a token rate read per %s; %s, as a later release reads it strictly",
        rate.model_key,
        "image" if meter == _IMAGES else "request",
        fix,
    )
