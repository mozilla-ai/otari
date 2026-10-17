"""The pricing row a gateway-run tool is priced under."""

GATEWAY_TOOL_PRICING_PROVIDER = "otari"


def gateway_tool_pricing_key(tool: str) -> str:
    """The ``ModelPricing.model_key`` an operator prices a gateway-run tool under.

    ``model_key`` is a free-form ``provider:model`` string, so a tool the gateway
    runs itself is priced as ``otari:<tool>`` (for example ``otari:web_search``).
    ``POST /v1/search`` prices a search on a configured tool under that tool's
    ``<search-provider>:<tool>`` instead, and under this key only a search a
    workspace's organization key served, whose provider the deployment cannot price.
    """
    return f"{GATEWAY_TOOL_PRICING_PROVIDER}:{tool}"
