"""What one successful attempt settled at."""

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal

from gateway.core.metered_pricing import ChargeLine


@dataclass(frozen=True)
class UsageCharge:
    """What an attempt settled at: the cost, rounded once, and the meters and charge lines that explain it.

    A free attempt carries no meter and no line, so a breakdown never explains a
    charge that did not happen. A meter holds a whole count, or the fraction a
    measured quantity is, as seconds and megapixels are.
    """

    cost: Decimal
    meters: Mapping[str, int | float]
    lines: list[ChargeLine]
