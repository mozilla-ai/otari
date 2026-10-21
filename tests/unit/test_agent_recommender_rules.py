"""What a subagent recommender an operator adds must be to work, and what one recommendation costs."""

from decimal import Decimal

import pytest

from gateway.core.config import DECISION_PROVIDERS
from gateway.services.routing import RecommenderPrice
from gateway.services.routing._recommender_rules import check_recommender

PRICED = RecommenderPrice(input_price_per_million=Decimal("0.15"))


@pytest.mark.parametrize("provider", ["typesafe", "openrouter"])
def test_a_hosted_provider_with_a_key_and_a_price_works(provider: str) -> None:
    check_recommender(provider=provider, has_api_key=True, api_base=None, price=PRICED)


def test_a_self_hosted_server_works_with_no_key_on_plain_http() -> None:
    check_recommender(provider="llamacpp", has_api_key=False, api_base="http://decisions:8080", price=PRICED)


@pytest.mark.parametrize(
    ("provider", "has_api_key", "api_base", "complaint"),
    [
        pytest.param("openai", True, None, "provider 'openai' is not a supported decision provider", id="unknown"),
        pytest.param("typesafe", False, None, "api_key is required for provider 'typesafe'", id="no-key"),
        pytest.param("llamacpp", False, None, "api_base is required for provider 'llamacpp'", id="no-address"),
        pytest.param("typesafe", True, "http://proxy.example", "api_base must use https", id="key-over-http"),
        pytest.param("llamacpp", True, "http://decisions:8080", "api_base must use https", id="self-hosted-key"),
        pytest.param("typesafe", True, "ftp://proxy.example", "api_base must use https", id="not-http"),
    ],
)
def test_an_endpoint_that_cannot_work_is_named(
    provider: str, has_api_key: bool, api_base: str | None, complaint: str
) -> None:
    with pytest.raises(ValueError, match=complaint):
        check_recommender(provider=provider, has_api_key=has_api_key, api_base=api_base, price=PRICED)


def test_the_providers_are_the_configurations_decision_providers() -> None:
    """The same list a ``decision_providers`` entry is held to, so the two cannot drift."""
    for provider in DECISION_PROVIDERS:
        check_recommender(provider=provider, has_api_key=True, api_base="https://decisions.example", price=PRICED)


def test_either_token_price_is_enough() -> None:
    check_recommender(
        provider="typesafe",
        has_api_key=True,
        api_base=None,
        price=RecommenderPrice(output_price_per_million=Decimal("0.6")),
    )


def test_a_recommendation_is_never_free() -> None:
    with pytest.raises(ValueError, match="above zero"):
        check_recommender(provider="typesafe", has_api_key=True, api_base=None, price=RecommenderPrice())


def test_a_negative_price_is_refused() -> None:
    price = RecommenderPrice(input_price_per_million=Decimal("1"), output_price_per_million=Decimal("-1"))

    with pytest.raises(ValueError, match="negative"):
        check_recommender(provider="typesafe", has_api_key=True, api_base=None, price=price)


def test_the_hold_prices_the_most_one_call_can_use_rounded_up() -> None:
    # 65,536 input tokens at $0.15 per million is $0.0098304, and 1,024 output at
    # $0.6 is $0.0006144: $0.0104448, held as $0.010445.
    price = RecommenderPrice(input_price_per_million=Decimal("0.15"), output_price_per_million=Decimal("0.6"))

    assert price.hold() == Decimal("0.010445")


def test_the_charge_is_the_tokens_used_rounded_half_up_to_the_micro_dollar() -> None:
    price = RecommenderPrice(input_price_per_million=Decimal("0.15"), output_price_per_million=Decimal("0.6"))

    # 10 input tokens and 5 output: $0.0000015 + $0.000003 = $0.0000045, half up.
    assert price.charge(input_tokens=10, output_tokens=5) == Decimal("0.000005")
    assert price.charge(input_tokens=0, output_tokens=0) == Decimal("0")


def test_a_charge_never_passes_the_hold() -> None:
    price = RecommenderPrice(input_price_per_million=Decimal("3"), output_price_per_million=Decimal("15"))

    assert price.charge(input_tokens=65_536, output_tokens=1_024) <= price.hold()
