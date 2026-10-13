"""Which meter a rate prices, and what one successful attempt costs at it.

The rate's ``unit`` chooses the meter. A ``tokens`` rate prices tokens, an
``images`` rate an ``images`` count, and a ``requests`` rate a ``requests`` or
``search_units`` count, or one request for an attempt that reported neither. A
rate prices nothing else, so a count never meets a rate written for another
unit. Standalone settlement and a hybrid gateway's control plane both price
through :func:`price_usage`, which is how one request costs the same on either.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from math import ceil, isfinite

from gateway.core.metered_pricing import (
    BillableUsage,
    ChargeLine,
    meter_cost,
    price_billable_usage,
    quantize_cost,
    to_decimal,
)
from gateway.log_config import logger
from gateway.models.pricing import MAX_REPORTED_UNITS, UNIT_METERS, ModelPricing

_REQUESTS, _SEARCH_UNITS, _IMAGES = UNIT_METERS

# The (model key, meter) pairs the legacy reading has already warned about.
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
    legacy_unit_reading: bool,
) -> UsageCharge | None:
    """What one successful attempt costs at ``rate``, or ``None`` when ``rate`` prices nothing it reported.

    ``legacy_unit_reading`` keeps a standalone gateway's older reading of a
    ``tokens`` rate on the endpoints billed per image or per request: the rate is
    read in that endpoint's unit, as the configuration guide documented before
    the rate's unit decided. It warns once per model and meter.

    Raises:
        ValueError: ``units`` carries an invalid count (see :func:`validated_units`).
    """
    counts = validated_units(units)
    if rate is None:
        return None
    unit = rate.unit or "tokens"
    if unit == "requests":
        meter = next((m for m in (_REQUESTS, _SEARCH_UNITS) if m in counts), _REQUESTS)
        return _per_unit(meter, counts.get(meter, 1), _request_rate(rate))
    if unit == "images":
        return _per_unit(_IMAGES, counts[_IMAGES], _image_rate(rate)) if _IMAGES in counts else None
    if unit != "tokens":
        return None
    if legacy_unit_reading:
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


def can_price(rate: ModelPricing | None, *, meter: str | None, reports_tokens: bool, legacy_unit_reading: bool) -> bool:
    """Whether ``rate`` could price an attempt that reports ``meter`` and, when ``reports_tokens``, tokens.

    The ``require_pricing`` gate asks this before dispatch, so a rate the
    endpoint's report cannot meet is refused rather than settled unpriced.
    """
    if rate is None or to_decimal(rate.input_price_per_million) is None:
        return False
    unit = rate.unit or "tokens"
    if unit == "requests":
        return True
    if unit == "images":
        return meter == _IMAGES
    if unit != "tokens":
        return False
    return reports_tokens or (legacy_unit_reading and meter in (_IMAGES, _REQUESTS))


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


def _warn_legacy(rate: ModelPricing, meter: str) -> None:
    key = (rate.model_key, meter)
    if key in _LEGACY_WARNED:
        return
    _LEGACY_WARNED.add(key)
    if meter == _IMAGES:
        # An images rate is per million images, where this one is read as raw USD per image.
        fix = "set its unit to images and multiply its input price by 1,000,000"
    else:
        fix = "set its unit to requests, keeping its price"
    logger.warning(
        "Pricing %s is a token rate read per %s; %s, as a later release reads it strictly",
        rate.model_key,
        "image" if meter == _IMAGES else "request",
        fix,
    )
