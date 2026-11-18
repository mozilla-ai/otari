"""The images response the gateway returns for a provider's result."""

from decimal import Decimal

from any_llm.types.image import Image, ImagesResponse
from openai.types.images_response import Usage, UsageInputTokensDetails

from gateway.models.pricing import ModelPricing
from gateway.services.inference import InferenceService
from gateway.services.pricing_service import ResolvedPricing

PRICING = ResolvedPricing(
    ModelPricing(
        model_key="openai:gpt-image-1",
        input_price_per_million=Decimal("0.04"),
        output_price_per_million=Decimal("0"),
    ),
    "deployment",
)


def _result(usage: Usage | None = None) -> ImagesResponse:
    return ImagesResponse(created=1700000000, data=[Image(url="https://example.com/1.png")], usage=usage)


def test_a_priced_result_with_no_usage_gets_one_holding_both_fields() -> None:
    body = InferenceService.images_response(_result(), pricing=PRICING, cost=Decimal("0.04"))
    assert body["usage"] == {"cost_usd": "0.040000", "pricing_source": "deployment"}
    assert body["data"] == [{"url": "https://example.com/1.png", "b64_json": None, "revised_prompt": None}]


def test_a_priced_result_keeps_the_providers_usage_beside_the_fields() -> None:
    usage = Usage(
        input_tokens=13,
        input_tokens_details=UsageInputTokensDetails(image_tokens=0, text_tokens=13),
        output_tokens=1673,
        total_tokens=1686,
    )
    body = InferenceService.images_response(_result(usage), pricing=PRICING, cost=Decimal("0.0377665"))
    assert body["usage"] == {
        **usage.model_dump(),
        "cost_usd": "0.037767",
        "pricing_source": "deployment",
    }


def test_an_unpriced_result_is_the_providers_body_unchanged() -> None:
    result = _result()
    assert InferenceService.images_response(result, pricing=None, cost=None) == result.model_dump()
    assert InferenceService.images_response(result, pricing=PRICING, cost=None)["usage"] is None
    assert InferenceService.images_response(result, pricing=None, cost=Decimal("0.04"))["usage"] is None


def test_a_providers_own_cost_fields_never_reach_the_caller() -> None:
    usage = Usage.model_validate(
        {
            "input_tokens": 13,
            "input_tokens_details": {"image_tokens": 0, "text_tokens": 13},
            "output_tokens": 1673,
            "total_tokens": 1686,
            "cost_usd": "9.000000",
            "pricing_source": "upstream",
        }
    )
    tokens = {name: value for name, value in usage.model_dump().items() if name not in ("cost_usd", "pricing_source")}

    unpriced = InferenceService.images_response(_result(usage), pricing=None, cost=None)
    assert unpriced["usage"] == tokens

    priced = InferenceService.images_response(_result(usage), pricing=PRICING, cost=Decimal("0.04"))
    assert priced["usage"] == {**tokens, "cost_usd": "0.040000", "pricing_source": "deployment"}
