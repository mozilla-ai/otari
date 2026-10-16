"""One label's successful calls of a gateway-run tool, as settlement prices them.

The tally in ``services/tool_usage`` builds these and ``services/pricing_service``
prices them. The type lives here, not beside either, so the tools and pricing
domains share it without importing each other.
"""

from dataclasses import dataclass
from decimal import Decimal

__all__ = ["LabelCalls"]


@dataclass(frozen=True, slots=True)
class LabelCalls:
    """A request's successful calls of one tool under one label.

    ``label`` names what served the calls, and ``rate_keys`` the pricing rows
    that may price them, first match winning. ``None`` for either means the
    tool's own ``otari:<tool>`` row; an empty ``rate_keys`` prices nothing.
    ``reported_cost`` sums the costs providers reported, over ``calls`` minus
    ``unreported_calls``, and ``enrichment_cost`` what a search's enrichment
    fetches reported.
    """

    tool: str
    label: str | None
    rate_keys: tuple[str, ...] | None
    calls: int
    unreported_calls: int
    reported_cost: Decimal
    enrichment_cost: Decimal
