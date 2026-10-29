"""What makes a stored subagent router usable, and the one list the providers come from."""

from decimal import Decimal

import pytest

from gateway.core.config import DECISION_PROVIDERS, DECISION_PROVIDERS_SELF_HOSTED
from gateway.services.routing import RecommenderPrice, check_recommender

PRICED = RecommenderPrice(input_price_per_million=Decimal("1"))


def test_a_keyed_provider_with_a_price_passes() -> None:
    check_recommender(provider="typesafe", has_api_key=True, api_base=None, price=PRICED)
    check_recommender(provider="openrouter", has_api_key=True, api_base="https://proxy.example", price=PRICED)


def test_a_self_hosted_server_needs_no_key_but_an_address() -> None:
    check_recommender(provider="llamacpp", has_api_key=False, api_base="http://decisions.internal:8080", price=PRICED)
    with pytest.raises(ValueError, match="API base is required"):
        check_recommender(provider="llamacpp", has_api_key=False, api_base=None, price=PRICED)


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"provider": "unknown"}, "provider must be one of"),
        ({"has_api_key": False}, "API key is required"),
        ({"api_base": "ftp://proxy.example"}, "http or https"),
        ({"api_base": "http://proxy.example"}, "must use https when an API key is set"),
        ({"price": RecommenderPrice(input_price_per_million=Decimal("-1"))}, "cannot be negative"),
        ({"price": RecommenderPrice()}, "above zero"),
    ],
)
def test_a_configuration_that_cannot_work_is_named(fields: dict[str, object], message: str) -> None:
    config: dict[str, object] = {"provider": "typesafe", "has_api_key": True, "api_base": None, "price": PRICED}
    config.update(fields)

    with pytest.raises(ValueError, match=message):
        check_recommender(**config)  # type: ignore[arg-type]


def test_the_providers_are_the_decision_providers_the_config_knows() -> None:
    """One list, not a restated copy: a provider added to ``decision_providers`` is usable here at once."""
    for provider in DECISION_PROVIDERS:
        self_hosted = provider in DECISION_PROVIDERS_SELF_HOSTED
        check_recommender(
            provider=provider,
            has_api_key=not self_hosted,
            api_base="https://decisions.example" if self_hosted else None,
            price=PRICED,
        )
