"""What a video costs: the per-second price its model's rates give video output.

A video is a job billed when it completes, so settlement prices the seconds the
provider reports at the rate found for the model. The rate is the ``video``
slot of the row's ``rates`` block, read through the reader every rate goes
through, so an operator's price and a catalog price read alike. A model whose
block prices video in another unit, or not at all, has no video rate.
"""

from decimal import Decimal

from gateway.core.metered_pricing import UnitRate, charge_unit, quantize_cost, slot_price, unit_rate
from gateway.models.pricing import ModelPricing
from gateway.services.pricing._usage_pricing import UsageCharge
from gateway.types.models_dev import parse_pricing
from gateway.types.price_unit import PRICE_UNITS

# The meter a video's length is reported under on its usage row.
VIDEO_SECONDS = "seconds"


def video_rate(rate: ModelPricing | None) -> UnitRate | None:
    """The per-second price ``rate`` gives video output, or ``None`` when it gives none."""
    if rate is None or not rate.rates:
        return None
    pricing, _ = parse_pricing(rate.rates, units=PRICE_UNITS, path="rates")
    price = slot_price(pricing, "video", "output")
    if price is None:
        return None
    found = unit_rate(price)
    return found if found is not None and found.unit == "second" else None


def estimate_video_cost(rate: ModelPricing | None, seconds: int) -> Decimal:
    """What to reserve for a video of ``seconds``: its length at the per-second price, or zero when unpriced."""
    video = video_rate(rate)
    return Decimal(0) if video is None else quantize_cost(video.cost(max(seconds, 0)))


def price_video(rate: ModelPricing | None, seconds: int) -> UsageCharge | None:
    """What a completed video of ``seconds`` costs at ``rate``, or ``None`` when it gives no video rate."""
    video = video_rate(rate)
    if video is None:
        return None
    charge = charge_unit(video, VIDEO_SECONDS, seconds)
    return UsageCharge(quantize_cost(charge.cost), {VIDEO_SECONDS: seconds} if seconds else {}, charge.lines)
