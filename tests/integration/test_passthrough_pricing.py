"""What each pass-through endpoint holds, settles and records, priced by the rate's unit.

Every case reads the request's usage row (cost, meters, charge lines, pricing
source) and the user's budget (what was held before the call, what the spend
became, and that the hold was released), because a price that is right on the
row and wrong on the budget is still a billing bug.
"""

import time
from collections.abc import Callable, Generator
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from any_llm.types.audio import Transcription
from any_llm.types.completion import CreateEmbeddingResponse, Embedding, Usage
from any_llm.types.image import Image, ImagesResponse
from any_llm.types.rerank import RerankMeta, RerankResponse, RerankResult
from fastapi.testclient import TestClient
from openai.types.images_response import Usage as ImageUsage
from openai.types.images_response import UsageInputTokensDetails
from sqlalchemy import select
from sqlalchemy.orm import Session

from gateway.core.config import API_KEY_HEADER, API_ROOT, GatewayConfig
from gateway.models.usage import UsageLog

from .conftest import build_test_client, held_reservations

USER_ID = "passthrough-pricing-user"


def _price(client: TestClient, headers: dict[str, str], model: str, *, unit: str, rate: float) -> None:
    response = client.post(
        f"{API_ROOT}/pricing",
        json={"model_key": model, "input_price_per_million": rate, "output_price_per_million": 0.0, "unit": unit},
        headers=headers,
    )
    assert response.status_code == 200, response.text


def _price_rates(client: TestClient, headers: dict[str, str], model: str, rates: dict[str, Any]) -> None:
    response = client.post(f"{API_ROOT}/pricing", json={"model_key": model, "rates": rates}, headers=headers)
    assert response.status_code == 200, response.text


def _user_key(client: TestClient, headers: dict[str, str]) -> dict[str, str]:
    budget = client.post(f"{API_ROOT}/budgets", json={"max_budget": 100.0}, headers=headers)
    assert budget.status_code == 200, budget.text
    user = client.post(
        f"{API_ROOT}/users", json={"user_id": USER_ID, "budget_id": budget.json()["budget_id"]}, headers=headers
    )
    assert user.status_code == 200, user.text
    key = client.post(f"{API_ROOT}/keys", json={"key_name": "pricing", "user_id": USER_ID}, headers=headers)
    assert key.status_code == 200, key.text
    return {API_KEY_HEADER: f"Bearer {key.json()['key']}"}


def _row(make_session: Callable[[], Session], *, timeout: float = 3.0) -> UsageLog:
    """The request's usage row, polled because a background log writer commits it."""
    deadline = time.monotonic() + timeout
    while True:
        with make_session() as db:
            row = db.execute(select(UsageLog).where(UsageLog.user_id == USER_ID)).scalar_one_or_none()
            if row is not None:
                db.expunge(row)
                return row
        assert time.monotonic() < deadline, "the usage row was never written"
        time.sleep(0.1)


def _settled(client: TestClient, headers: dict[str, str], cost: float) -> None:
    """The spend is the row's cost and nothing is still held."""
    user = client.get(f"{API_ROOT}/users/{USER_ID}", headers=headers).json()
    assert user["spend"] == pytest.approx(cost)
    assert user["reserved"] == pytest.approx(0.0)


def _images(n: int, usage: ImageUsage | None = None) -> ImagesResponse:
    return ImagesResponse(created=0, data=[Image(url=f"https://example.com/{i}.png") for i in range(n)], usage=usage)


def _generate(
    client: TestClient, headers: dict[str, str], result: ImagesResponse, *, n: int, size: str | None = None
) -> Any:
    body: dict[str, Any] = {"model": "openai:dall-e-3", "prompt": "a cat", "n": n}
    if size is not None:
        body["size"] = size
    with patch("gateway.api.routes.images.aimage_generation", new=AsyncMock(return_value=result)):
        return client.post(f"{API_ROOT}/images/generations", json=body, headers=headers)


def test_a_megapixel_rate_holds_and_settles_the_pixels_of_the_size_asked_for(
    client: TestClient, master_key_header: dict[str, str], db_session_factory: Callable[[], Session]
) -> None:
    """Three 1024x1024 images held, two settled, each 1.048576 megapixels at $0.05."""
    _price_rates(
        client, master_key_header, "openai:dall-e-3", {"image": {"output": {"price": 0.05, "unit": "megapixel"}}}
    )
    headers = _user_key(client, master_key_header)

    with held_reservations() as held:
        response = _generate(client, headers, _images(2), n=3, size="1024x1024")

    assert response.status_code == 200, response.text
    assert held == [Decimal("0.157286")]
    row = _row(db_session_factory)
    assert row.cost == Decimal("0.104858")
    assert row.billing_meters == {"megapixels": 2.097152}
    assert row.pricing_breakdown == [
        {"meter": "megapixels", "units": 2.097152, "unit": "megapixel", "unit_rate": 0.05, "cost": 0.1048576}
    ]
    _settled(client, master_key_header, 0.104858)


def test_a_megapixel_rate_prices_nothing_an_image_of_no_stated_size(
    client: TestClient, master_key_header: dict[str, str], db_session_factory: Callable[[], Session]
) -> None:
    """A size the request does not name measures no pixels, so the row keeps no cost rather than a guess."""
    _price_rates(
        client, master_key_header, "openai:dall-e-3", {"image": {"output": {"price": 0.05, "unit": "megapixel"}}}
    )
    headers = _user_key(client, master_key_header)

    with held_reservations() as held:
        response = _generate(client, headers, _images(1), n=1, size="auto")

    assert response.status_code == 200, response.text
    assert held == [Decimal(0)]
    assert _row(db_session_factory).cost is None
    _settled(client, master_key_header, 0.0)


def _speak(client: TestClient, headers: dict[str, str], text: str) -> Any:
    with patch("gateway.api.routes.audio.aspeech", new=AsyncMock(return_value=b"audio")):
        return client.post(
            f"{API_ROOT}/audio/speech",
            json={"model": "openai:tts-1", "input": text, "voice": "alloy"},
            headers=headers,
        )


def test_a_character_rate_holds_and_settles_the_characters_spoken(
    client: TestClient, master_key_header: dict[str, str], db_session_factory: Callable[[], Session]
) -> None:
    """Eleven characters at $15 per million, held before the call and settled after it."""
    _price_rates(client, master_key_header, "openai:tts-1", {"text": {"input": {"price": 15, "unit": "character"}}})
    headers = _user_key(client, master_key_header)

    with held_reservations() as held:
        response = _speak(client, headers, "Hello world")

    assert response.status_code == 200, response.text
    assert held == [Decimal("0.000165")]
    assert response.headers["Otari-Cost-USD"] == "0.000165"
    assert response.headers["Otari-Pricing-Source"] == "deployment"
    assert response.headers["Otari-Usage-Unit"] == "characters"
    assert response.headers["Otari-Usage-Units"] == "11"
    row = _row(db_session_factory)
    assert row.cost == Decimal("0.000165")
    assert row.billing_meters == {"characters": 11}
    assert row.pricing_breakdown == [
        {"meter": "characters", "units": 11, "unit": "character", "rate_per_million": 15.0, "cost": 0.000165}
    ]
    _settled(client, master_key_header, 0.000165)


def test_an_images_rate_holds_the_requested_images_and_settles_the_returned_ones(
    client: TestClient, master_key_header: dict[str, str], db_session_factory: Callable[[], Session]
) -> None:
    # $0.04 an image, per million images: what the dashboard stores for $40 per 1,000.
    _price(client, master_key_header, "openai:dall-e-3", unit="image", rate=40000.0)
    headers = _user_key(client, master_key_header)

    with held_reservations() as held:
        response = _generate(client, headers, _images(2), n=3)

    assert response.status_code == 200, response.text
    assert held == [Decimal("0.12")]
    row = _row(db_session_factory)
    assert row.cost == Decimal("0.08")
    assert row.pricing_source == "deployment"
    assert row.billing_meters == {"images": 2}
    assert row.pricing_breakdown == [{"meter": "images", "units": 2, "unit": "image", "unit_rate": 0.04, "cost": 0.08}]
    # The response carries what the row records, as a completion's usage does.
    assert response.json()["usage"] == {"cost_usd": f"{row.cost:.6f}", "pricing_source": row.pricing_source}
    _settled(client, master_key_header, 0.08)


def test_a_token_rate_on_images_prices_the_tokens_it_reports(
    client: TestClient, master_key_header: dict[str, str], db_session_factory: Callable[[], Session]
) -> None:
    """A token rate reads as tokens on every endpoint: nothing is held by count, and the tokens settle."""
    _price(client, master_key_header, "openai:dall-e-3", unit="token", rate=0.04)
    headers = _user_key(client, master_key_header)
    usage = ImageUsage(
        input_tokens=50,
        output_tokens=4160,
        total_tokens=4210,
        input_tokens_details=UsageInputTokensDetails(image_tokens=0, text_tokens=50),
    )

    with held_reservations() as held:
        response = _generate(client, headers, _images(1, usage), n=1)

    assert response.status_code == 200, response.text
    assert held == [Decimal(0)]
    row = _row(db_session_factory)
    assert row.cost == Decimal("0.000002")
    assert row.billing_meters is not None and row.billing_meters["total_input_tokens"] == 50
    assert (row.prompt_tokens, row.completion_tokens, row.total_tokens) == (50, 4160, 4210)
    # The provider's token counts stay on ``usage``, with the settled cost beside them.
    assert response.json()["usage"] == {
        **usage.model_dump(),
        "cost_usd": f"{row.cost:.6f}",
        "pricing_source": row.pricing_source,
    }
    _settled(client, master_key_header, 0.000002)


def test_a_requests_rate_on_images_bills_one_request_not_its_rate_per_image(
    client: TestClient, master_key_header: dict[str, str], db_session_factory: Callable[[], Session]
) -> None:
    """A per-request price read as dollars per image billed a million times too much."""
    _price(client, master_key_header, "openai:dall-e-3", unit="request", rate=2000.0)
    headers = _user_key(client, master_key_header)

    with held_reservations() as held:
        response = _generate(client, headers, _images(2), n=2)

    assert response.status_code == 200, response.text
    assert held == [Decimal("0.002")]
    row = _row(db_session_factory)
    assert row.cost == Decimal("0.002")
    assert row.billing_meters == {"requests": 1}
    _settled(client, master_key_header, 0.002)


def _rerank(client: TestClient, headers: dict[str, str], result: RerankResponse, documents: list[str]) -> Any:
    with patch("gateway.api.routes.rerank.arerank", new=AsyncMock(return_value=result)):
        return client.post(
            f"{API_ROOT}/rerank",
            json={"model": "cohere:rerank-v3.5", "query": "q", "documents": documents},
            headers=headers,
        )


def test_rerank_holds_the_search_units_its_documents_imply_and_settles_the_billed_ones(
    client: TestClient, master_key_header: dict[str, str], db_session_factory: Callable[[], Session]
) -> None:
    _price(client, master_key_header, "cohere:rerank-v3.5", unit="request", rate=2000.0)
    headers = _user_key(client, master_key_header)
    result = RerankResponse(
        id="r",
        model="rerank-v3.5",
        results=[RerankResult(index=0, relevance_score=0.9)],
        meta=RerankMeta(billed_units={"search_units": 3}),
    )

    with held_reservations() as held:
        response = _rerank(client, headers, result, ["short document"] * 250)

    assert response.status_code == 200, response.text
    assert held == [Decimal("0.006")]
    row = _row(db_session_factory)
    assert row.cost == Decimal("0.006")
    assert row.billing_meters == {"search_units": 3}
    _settled(client, master_key_header, 0.006)


def _embed(client: TestClient, headers: dict[str, str]) -> Any:
    result = CreateEmbeddingResponse(
        data=[Embedding(embedding=[0.1], index=0, object="embedding")],
        model="text-embedding-3-small",
        object="list",
        usage=Usage(prompt_tokens=1000, total_tokens=1000),
    )
    with patch("gateway.api.routes.embeddings.aembedding", new=AsyncMock(return_value=result)):
        return client.post(
            f"{API_ROOT}/embeddings", json={"model": "openai:text-embedding-3-small", "input": "hi"}, headers=headers
        )


def test_a_requests_rate_on_embeddings_bills_one_request_not_the_tokens(
    client: TestClient, master_key_header: dict[str, str], db_session_factory: Callable[[], Session]
) -> None:
    """A per-call price multiplied by the token count billed a thousand calls for one."""
    _price(client, master_key_header, "openai:text-embedding-3-small", unit="request", rate=5000.0)
    headers = _user_key(client, master_key_header)

    response = _embed(client, headers)

    assert response.status_code == 200, response.text
    assert response.json()["usage"]["cost_usd"] == "0.005000"
    row = _row(db_session_factory)
    assert row.cost == Decimal("0.005")
    assert row.billing_meters == {"requests": 1}
    assert (row.prompt_tokens, row.total_tokens) == (1000, 1000)
    _settled(client, master_key_header, 0.005)


def _transcribe(client: TestClient, headers: dict[str, str]) -> Any:
    with patch("gateway.api.routes.audio.atranscription", new=AsyncMock(return_value=Transcription(text="hi"))):
        return client.post(
            f"{API_ROOT}/audio/transcriptions",
            data={"model": "openai:whisper-1"},
            files={"file": ("a.mp3", b"audio", "audio/mpeg")},
            headers=headers,
        )


def test_an_unpriced_transcription_is_recorded_with_no_cost(
    client: TestClient, master_key_header: dict[str, str], db_session_factory: Callable[[], Session]
) -> None:
    """A missing cost is unknown, not free, so the row keeps none and the needs-pricing view lists it."""
    headers = _user_key(client, master_key_header)

    response = _transcribe(client, headers)

    assert response.status_code == 200, response.text
    row = _row(db_session_factory)
    assert row.cost is None
    assert row.billing_meters is None
    _settled(client, master_key_header, 0.0)


def test_an_images_rate_on_a_per_request_endpoint_settles_unpriced_rather_than_per_image(
    client: TestClient, master_key_header: dict[str, str], db_session_factory: Callable[[], Session]
) -> None:
    """The rate cannot price a request, so the row keeps no cost: never the rate read per image."""
    _price(client, master_key_header, "openai:whisper-1", unit="image", rate=0.5)
    headers = _user_key(client, master_key_header)

    response = _transcribe(client, headers)

    assert response.status_code == 200, response.text
    assert _row(db_session_factory).cost is None
    _settled(client, master_key_header, 0.0)


@pytest.fixture
def strict_client(postgres_url: str) -> Generator[TestClient]:
    """A deployment that refuses a model no rate can price (``require_pricing``)."""
    config = GatewayConfig(
        database_url=postgres_url,
        master_key="test-master-key",
        host="127.0.0.1",
        port=8000,
        auto_migrate=False,
        require_pricing=True,
        default_pricing=False,
        model_discovery=False,
        providers={"openai": {"api_key": "sk-test"}},
    )
    yield from build_test_client(config)


def test_require_pricing_refuses_a_rate_the_endpoint_cannot_settle(strict_client: TestClient) -> None:
    """An images rate on embeddings would settle unpriced; it is refused before the provider is called."""
    master = {API_KEY_HEADER: "Bearer test-master-key"}
    _price(strict_client, master, "openai:text-embedding-3-small", unit="image", rate=0.04)
    headers = _user_key(strict_client, master)
    provider = AsyncMock()

    with patch("gateway.api.routes.embeddings.aembedding", new=provider):
        response = strict_client.post(
            f"{API_ROOT}/embeddings", json={"model": "openai:text-embedding-3-small", "input": "hi"}, headers=headers
        )

    assert response.status_code == 402, response.text
    provider.assert_not_called()
    _settled(strict_client, master, 0.0)


def test_require_pricing_refuses_a_measured_unit_the_request_cannot_measure(strict_client: TestClient) -> None:
    """A per-megapixel rate cannot price an image of no stated size, so the request is refused before the call."""
    master = {API_KEY_HEADER: "Bearer test-master-key"}
    _price_rates(strict_client, master, "openai:dall-e-3", {"image": {"output": {"price": 0.05, "unit": "megapixel"}}})
    headers = _user_key(strict_client, master)
    provider = AsyncMock()

    with patch("gateway.api.routes.images.aimage_generation", new=provider):
        refused = strict_client.post(
            f"{API_ROOT}/images/generations",
            json={"model": "openai:dall-e-3", "prompt": "a cat", "size": "auto"},
            headers=headers,
        )
    assert refused.status_code == 402, refused.text
    provider.assert_not_called()

    response = _generate(strict_client, headers, _images(1), n=1, size="1024x1024")
    assert response.status_code == 200, response.text
    _settled(strict_client, master, 0.052429)


def test_require_pricing_refuses_an_unpriced_speech_model(strict_client: TestClient) -> None:
    """Speech is no longer served without a rate: an unpriced model is refused before the provider is called."""
    master = {API_KEY_HEADER: "Bearer test-master-key"}
    headers = _user_key(strict_client, master)
    provider = AsyncMock(return_value=b"audio")

    with patch("gateway.api.routes.audio.aspeech", new=provider):
        response = strict_client.post(
            f"{API_ROOT}/audio/speech",
            json={"model": "openai:tts-1", "input": "Hello world", "voice": "alloy"},
            headers=headers,
        )

    assert response.status_code == 402, response.text
    assert response.headers["Otari-Error-Code"] == "pricing_required"
    provider.assert_not_called()
    _settled(strict_client, master, 0.0)


def test_require_pricing_refuses_a_token_rate_on_speech(strict_client: TestClient) -> None:
    """The audio reports no tokens, so a token rate would settle unpriced; it is refused instead."""
    master = {API_KEY_HEADER: "Bearer test-master-key"}
    _price(strict_client, master, "openai:tts-1", unit="token", rate=0.6)
    headers = _user_key(strict_client, master)
    provider = AsyncMock(return_value=b"audio")

    with patch("gateway.api.routes.audio.aspeech", new=provider):
        response = strict_client.post(
            f"{API_ROOT}/audio/speech",
            json={"model": "openai:tts-1", "input": "Hello world", "voice": "alloy"},
            headers=headers,
        )

    assert response.status_code == 402, response.text
    detail = response.json()["detail"]
    assert detail.startswith("Model 'openai:tts-1' is priced per token, which this request cannot be billed on")
    assert "price it per character or per request" in detail
    assert "No pricing is configured" not in detail
    provider.assert_not_called()
    _settled(strict_client, master, 0.0)


def test_require_pricing_admits_a_character_rate_on_speech(strict_client: TestClient) -> None:
    master = {API_KEY_HEADER: "Bearer test-master-key"}
    _price(strict_client, master, "openai:tts-1", unit="character", rate=15.0)
    headers = _user_key(strict_client, master)

    response = _speak(strict_client, headers, "Hello world")

    assert response.status_code == 200, response.text
    assert response.headers["Otari-Cost-USD"] == "0.000165"
    assert response.headers["Otari-Usage-Units"] == "11"
    _settled(strict_client, master, 0.000165)


def test_require_pricing_admits_a_rate_the_endpoint_can_settle(strict_client: TestClient) -> None:
    master = {API_KEY_HEADER: "Bearer test-master-key"}
    _price(strict_client, master, "openai:text-embedding-3-small", unit="token", rate=2.0)
    headers = _user_key(strict_client, master)

    response = _embed(strict_client, headers)

    assert response.status_code == 200, response.text
    assert response.json()["usage"]["cost_usd"] == "0.002000"
