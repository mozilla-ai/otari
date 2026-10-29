"""What a stored subagent router charges: tokens at the rates, held at the most a call can use."""

from decimal import Decimal

from gateway.services.routing import HOLD_INPUT_TOKENS, HOLD_OUTPUT_TOKENS, RecommenderPrice

PRICE = RecommenderPrice(input_price_per_million=Decimal("3"), output_price_per_million=Decimal("15"))


def test_a_charge_is_the_tokens_at_the_rates() -> None:
    assert PRICE.charge(input_tokens=1_000_000, output_tokens=1_000_000) == Decimal("18.000000")
    assert PRICE.charge(input_tokens=2_000, output_tokens=100) == Decimal("0.007500")


def test_no_tokens_charge_nothing() -> None:
    assert PRICE.charge(input_tokens=0, output_tokens=0) == Decimal("0.000000")


def test_a_charge_is_rounded_half_up_to_the_micro_dollar() -> None:
    assert PRICE.charge(input_tokens=1, output_tokens=0) == Decimal("0.000003")
    assert RecommenderPrice(input_price_per_million=Decimal("0.5")).charge(input_tokens=1, output_tokens=0) == Decimal(
        "0.000001"
    )
    assert RecommenderPrice(input_price_per_million=Decimal("0.4")).charge(input_tokens=1, output_tokens=0) == Decimal(
        "0.000000"
    )


def test_the_hold_covers_the_largest_call_and_rounds_up() -> None:
    hold = PRICE.hold()

    assert hold == PRICE.charge(input_tokens=HOLD_INPUT_TOKENS, output_tokens=HOLD_OUTPUT_TOKENS)
    assert hold >= PRICE.charge(input_tokens=HOLD_INPUT_TOKENS, output_tokens=HOLD_OUTPUT_TOKENS)
    tiny = RecommenderPrice(input_price_per_million=Decimal("0.00000001"))
    assert tiny.hold() == Decimal("0.000001")
    assert tiny.hold() > tiny.charge(input_tokens=HOLD_INPUT_TOKENS, output_tokens=0)


def test_the_default_price_is_free() -> None:
    assert RecommenderPrice().hold() == Decimal("0.000000")
