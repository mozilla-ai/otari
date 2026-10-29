"""What makes a stored recommender usable.

The rules are the ones ``validate_decision_providers`` holds a
``decision_providers`` entry to, plus the price: a stored router charges the
caller, so it names a rate.
"""

from urllib.parse import urlsplit

from gateway.core.config import DECISION_PROVIDERS, DECISION_PROVIDERS_SELF_HOSTED
from gateway.services.routing._agent_recommender_price import RecommenderPrice


def check_recommender(*, provider: str, has_api_key: bool, api_base: str | None, price: RecommenderPrice) -> None:
    """Raise ``ValueError`` naming what is wrong with a recommender configuration."""
    if provider not in DECISION_PROVIDERS:
        msg = f"provider must be one of: {', '.join(DECISION_PROVIDERS)}"
        raise ValueError(msg)
    self_hosted = provider in DECISION_PROVIDERS_SELF_HOSTED
    if not has_api_key and not self_hosted:
        msg = f"an API key is required for provider '{provider}'"
        raise ValueError(msg)
    if not api_base and self_hosted:
        msg = f"an API base is required for provider '{provider}'"
        raise ValueError(msg)
    if api_base:
        scheme = urlsplit(api_base).scheme
        if scheme not in ("http", "https"):
            msg = "the API base must be an http or https URL"
            raise ValueError(msg)
        # The key travels in a header to this address.
        if has_api_key and scheme != "https":
            msg = "the API base must use https when an API key is set"
            raise ValueError(msg)
    rates = (price.input_price_per_million, price.output_price_per_million)
    if any(rate < 0 for rate in rates):
        msg = "a price cannot be negative"
        raise ValueError(msg)
    # A stored router charges the caller, so it names a rate; a free decision
    # model is configured in config.yml instead.
    if not any(rate > 0 for rate in rates):
        msg = "set an input or output token price above zero"
        raise ValueError(msg)
