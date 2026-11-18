"""The headers a speech response carries its settled cost on."""

from decimal import Decimal

from gateway.core.config import COST_USD_HEADER, PRICING_SOURCE_HEADER, USAGE_UNIT_HEADER, USAGE_UNITS_HEADER
from gateway.models.pricing import ModelPricing
from gateway.services.inference import speech_cost_headers
from gateway.services.pricing import Slot, UsageCharge, price_usage
from gateway.services.pricing_service import ResolvedPricing
from gateway.types.price_rates import RateColumns, rates_from_columns
from gateway.types.price_unit import PriceUnit

_TEXT_INPUT: Slot = ("text", "input")


def _pricing(unit: PriceUnit, rate: str) -> ResolvedPricing:
    columns = RateColumns(unit=unit, input_price_per_million=Decimal(rate), output_price_per_million=Decimal(0))
    return ResolvedPricing(
        ModelPricing(
            model_key="groq:orpheus-v1-english",
            input_price_per_million=Decimal(rate),
            output_price_per_million=Decimal(0),
            unit=unit,
            rates=rates_from_columns(columns),
        ),
        "deployment",
    )


def test_a_character_billed_speech_names_its_cost_source_meter_and_count() -> None:
    pricing = _pricing("character", "15")
    charge = price_usage(
        pricing.pricing, tokens=None, units={"requests": 1}, quantities={"character": 5}, slot=_TEXT_INPUT
    )

    assert speech_cost_headers(pricing=pricing, charge=charge) == {
        COST_USD_HEADER: "0.000075",
        PRICING_SOURCE_HEADER: "deployment",
        USAGE_UNIT_HEADER: "characters",
        USAGE_UNITS_HEADER: "5",
    }


def test_a_request_billed_speech_names_one_request() -> None:
    pricing = _pricing("request", "2000")
    charge = price_usage(
        pricing.pricing, tokens=None, units={"requests": 1}, quantities={"character": 500}, slot=_TEXT_INPUT
    )

    assert speech_cost_headers(pricing=pricing, charge=charge) == {
        COST_USD_HEADER: "0.002000",
        PRICING_SOURCE_HEADER: "deployment",
        USAGE_UNIT_HEADER: "requests",
        USAGE_UNITS_HEADER: "1",
    }


def test_a_zero_rate_carries_the_cost_and_source_and_names_no_meter() -> None:
    """A free attempt carries no meter, as its usage row carries none."""
    pricing = _pricing("character", "0")

    headers = speech_cost_headers(pricing=pricing, charge=UsageCharge(Decimal(0), {}, []))

    assert headers == {COST_USD_HEADER: "0.000000", PRICING_SOURCE_HEADER: "deployment"}


def test_an_unpriced_speech_carries_no_cost_headers() -> None:
    """Cost and source go together, as the inline fields do, so an unpriced request carries neither."""
    assert speech_cost_headers(pricing=None, charge=None) == {}
    assert speech_cost_headers(pricing=_pricing("token", "1"), charge=None) == {}
    assert speech_cost_headers(pricing=None, charge=UsageCharge(Decimal(0), {}, [])) == {}
