"""What a quantity costs at one slot of a rate's block: the price its modality gives one direction, in its unit.

A rate's ``rates`` block prices each modality's input and output in a unit of
its own. An endpoint bills under one such slot (``image`` output, ``audio``
input) and measures a quantity in the units it can. The slot's unit chooses
which quantity is priced, so a per-minute audio rate reads the minutes and a
per-second one the seconds, and a quantity no slot prices is never billed in
another unit. The block is read through the reader every rate goes through, so
an operator's price and a catalog price read alike.
"""

from collections.abc import Mapping
from decimal import Decimal

from gateway.core.metered_pricing import Direction, UnitRate, charge_unit, quantize_cost, slot_price, unit_rate
from gateway.models.pricing import ModelPricing
from gateway.services.pricing._charge import UsageCharge
from gateway.types.models_dev import parse_pricing
from gateway.types.price_unit import PRICE_UNITS, PriceUnit

# What an endpoint measured, in each unit it can measure.
Quantities = Mapping[PriceUnit, int | Decimal]

# The meter a quantity of each unit is reported under on its usage row.
UNIT_METER: dict[PriceUnit, str] = {
    "token": "tokens",
    "request": "requests",
    "image": "images",
    "character": "characters",
    "megapixel": "megapixels",
    "second": "seconds",
    "minute": "minutes",
    "web_search": "searches",
}


def slot_rate(rate: ModelPricing | None, modality: str, direction: Direction) -> UnitRate | None:
    """The price ``rate``'s block gives ``modality`` in ``direction``, or ``None`` when it gives none."""
    if rate is None or not rate.rates:
        return None
    pricing, _ = parse_pricing(rate.rates, units=PRICE_UNITS, path="rates")
    price = slot_price(pricing, modality, direction)
    return None if price is None else unit_rate(price)


def estimate_slot_cost(
    rate: ModelPricing | None, modality: str, direction: Direction, quantities: Quantities
) -> Decimal:
    """What to hold for the quantity ``quantities`` gives in the slot's unit, or zero when neither is there."""
    found = slot_rate(rate, modality, direction)
    if found is None or (quantity := quantities.get(found.unit)) is None:
        return Decimal(0)
    return quantize_cost(found.cost(max(Decimal(quantity), Decimal(0))))


def price_slot(
    rate: ModelPricing | None, modality: str, direction: Direction, quantities: Quantities
) -> UsageCharge | None:
    """What the quantity in the slot's unit costs at ``rate``, or ``None`` when the slot or the quantity is missing.

    The cost is rounded once, and a charge of nothing carries no meter and no
    line. A quantity that is whole is reported as such on the meter and the
    line; a fractional one, as seconds and megapixels are, as the number it is.
    """
    found = slot_rate(rate, modality, direction)
    if found is None or (quantity := quantities.get(found.unit)) is None:
        return None
    meter = UNIT_METER[found.unit]
    charge = charge_unit(found, meter, quantity)
    cost = quantize_cost(charge.cost)
    if not cost:
        return UsageCharge(Decimal(0), {}, [])
    reported = int(quantity) if Decimal(quantity) == int(quantity) else float(quantity)
    return UsageCharge(cost, {meter: reported}, charge.lines)
