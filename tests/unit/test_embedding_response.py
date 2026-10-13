"""The embeddings response the gateway returns for a provider's result."""

from decimal import Decimal

from any_llm.types.completion import CreateEmbeddingResponse, Embedding, Usage

from gateway.models.pricing import ModelPricing
from gateway.services.inference import InferenceService
from gateway.services.pricing_service import ResolvedPricing

PRICING = ResolvedPricing(
    ModelPricing(
        model_key="openai:text-embedding-3-small",
        input_price_per_million=Decimal("0.02"),
        output_price_per_million=Decimal("0"),
    ),
    "deployment",
)


def _result(embedding: list[float] | str | None = None) -> CreateEmbeddingResponse:
    return CreateEmbeddingResponse.model_construct(
        data=[Embedding.model_construct(embedding=embedding or [0.1, 0.2], index=0, object="embedding")],
        model="text-embedding-3-small",
        object="list",
        usage=Usage(prompt_tokens=2, total_tokens=2),
    )


def test_a_priced_result_carries_both_fields() -> None:
    body = InferenceService.embeddings_response(_result(), pricing=PRICING, cost=Decimal("0.00002"))
    assert body.model_dump() == {
        "object": "list",
        "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
        "model": "text-embedding-3-small",
        "usage": {"prompt_tokens": 2, "total_tokens": 2, "cost_usd": "0.000020", "pricing_source": "deployment"},
    }


def test_an_unpriced_result_omits_both_fields() -> None:
    body = InferenceService.embeddings_response(_result(), pricing=None, cost=None)
    assert body.model_dump()["usage"] == {"prompt_tokens": 2, "total_tokens": 2}


def test_a_base64_embedding_is_kept_as_its_string() -> None:
    body = InferenceService.embeddings_response(_result("AAAAPw=="), pricing=None, cost=None)
    assert body.data[0].embedding == "AAAAPw=="


def test_a_result_without_usage_has_nothing_to_carry_a_cost_on() -> None:
    result = _result()
    result.usage = None  # type: ignore[assignment]
    body = InferenceService.embeddings_response(result, pricing=PRICING, cost=Decimal("0.000001"))
    assert "usage" not in body.model_dump()
