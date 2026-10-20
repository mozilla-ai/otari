"""One label's successful calls of a gateway-run tool, and the keys that name what served them.

The tally in ``services/tool_usage`` builds the calls, the web search resolver
names the labels and rate keys, and pricing prices them. They live here, not
beside any of those, so the tools and pricing domains share them without
importing each other.
"""

from dataclasses import dataclass
from decimal import Decimal

__all__ = [
    "GATEWAY_TOOL_PRICING_PROVIDER",
    "ORGANIZATION_KEY_LABEL_PREFIX",
    "PROVIDER_WEB_SEARCH_TOOL",
    "LabelCalls",
    "gateway_tool_pricing_key",
    "provider_web_search_label",
]

GATEWAY_TOOL_PRICING_PROVIDER = "otari"

# The label of a call a workspace's organization key served starts with this,
# then names the provider (``org:tavily``). It is never a pricing row: the
# resolver writes it, and pricing reads it only to tell a call on the
# organization's own key from one the deployment pays for, so both take it from
# here.
ORGANIZATION_KEY_LABEL_PREFIX = "org:"


# The tool a provider-run search is counted under, beside the gateway's own searches.
PROVIDER_WEB_SEARCH_TOOL = "web_search"


def provider_web_search_label(provider: str) -> str:
    """The label, and the rate key, of searches a model's provider ran itself (``anthropic:web_search``)."""
    return f"{provider}:{PROVIDER_WEB_SEARCH_TOOL}"


def gateway_tool_pricing_key(tool: str) -> str:
    """The ``ModelPricing.model_key`` an operator prices a gateway-run tool under.

    ``model_key`` is a free-form ``provider:model`` string, so a tool the gateway
    runs itself is priced as ``otari:<tool>`` (for example ``otari:web_search``).
    A direct search on a configured tool is priced under that tool's
    ``<search-provider>:<tool>`` instead, and under this key only a search a
    workspace's organization key served, whose provider the deployment cannot price.
    """
    return f"{GATEWAY_TOOL_PRICING_PROVIDER}:{tool}"


@dataclass(frozen=True, slots=True)
class LabelCalls:
    """A request's successful calls of one tool under one label.

    ``label`` names what served the calls, and ``rate_keys`` the pricing rows
    that may price them, first match winning. ``None`` for either means the
    tool's own ``otari:<tool>`` row; an empty ``rate_keys`` prices nothing.
    ``reported_cost`` sums the costs providers reported, over ``calls`` minus
    ``unreported_calls``, and ``enrichment_cost`` what a search's enrichment
    fetches reported. ``deployment_paid`` says whose account ran the calls when the
    label cannot: ``None`` reads it off the label.
    """

    tool: str
    label: str | None
    rate_keys: tuple[str, ...] | None
    calls: int
    unreported_calls: int
    reported_cost: Decimal
    enrichment_cost: Decimal
    deployment_paid: bool | None = None
