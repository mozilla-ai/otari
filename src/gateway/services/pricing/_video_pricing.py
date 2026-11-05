"""What a video costs: the per-second price its model's rates give video output.

A video is a job billed when it completes, so settlement prices the seconds the
provider reports at the rate found for the model. The rate is the ``video``
slot of the row's ``rates`` block, read through the reader every rate goes
through, so an operator's price and a catalog price read alike. A model whose
block prices video in another unit, or not at all, has no video rate.
"""

from decimal import Decimal

from gateway.core.metered_pricing import UnitRate
from gateway.models.pricing import ModelPricing
from gateway.services.pricing._charge import UsageCharge
from gateway.services.pricing._slot_pricing import UNIT_METER, estimate_slot_cost, price_slot, slot_rate

# The meter a video's length is reported under on its usage row.
VIDEO_SECONDS = UNIT_METER["second"]


def video_rate(rate: ModelPricing | None) -> UnitRate | None:
    """The per-second price ``rate`` gives video output, or ``None`` when it gives none."""
    found = slot_rate(rate, "video", "output")
    return found if found is not None and found.unit == "second" else None


def estimate_video_cost(rate: ModelPricing | None, seconds: int) -> Decimal:
    """What to reserve for a video of ``seconds``: its length at the per-second price, or zero when unpriced."""
    return estimate_slot_cost(rate, "video", "output", {"second": seconds})


def price_video(rate: ModelPricing | None, seconds: int) -> UsageCharge | None:
    """What a completed video of ``seconds`` costs at ``rate``, or ``None`` when it gives no video rate."""
    return price_slot(rate, "video", "output", {"second": seconds})
