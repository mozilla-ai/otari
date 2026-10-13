"""What each pass-through endpoint holds, settles and records, priced by the rate's unit.

Every case reads the request's usage row (cost, meters, charge lines, pricing
source) and the user's budget (what was held before the call, what the spend
became, and that the hold was released), because a price that is right on the
row and wrong on the budget is still a billing bug.
"""

import time
from collections.abc import Callable, Generator, Iterator
from contextlib import contextmanager
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
from gateway.services.budgets import reserve_budget

from .conftest import build_test_client

USER_ID = "passthrough-pricing-user"


def _price(client: TestClient, headers: dict[str, str], model: str, *, unit: str, rate: float) -> None:
    response = client.post(
        f"{API_ROOT}/pricing",
        json={"model_key": model, "input_price_per_million": rate, "output_price_per_million": 0.0, "unit": unit},
        headers=headers,
    )
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


@contextmanager
def _held() -> Iterator[list[Decimal]]:
    """Every amount the request reserved before calling the provider."""
    held: list[Decimal] = []

    async def capture(*args: Any, **kwargs: Any) -> Any:
        held.append(Decimal(str(args[2])))
        return await reserve_budget(*args, **kwargs)

    with patch("gateway.api.routes._passthrough.reserve_budget", side_effect=capture):
        yield held


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


def _generate(client: TestClient, headers: dict[str, str], result: ImagesResponse, *, n: int) -> Any:
    with patch("gateway.api.routes.images.aimage_generation", new=AsyncMock(return_value=result)):
        return client.post(
            f"{API_ROOT}/images/generations",
            json={"model": "openai:dall-e-3", "prompt": "a cat", "n": n},
            headers=headers,
        )


def test_an_images_rate_holds_the_requested_images_and_settles_the_returned_ones(
    client: TestClient, master_key_header: dict[str, str], db_session_factory: Callable[[], Session]
) -> None:
    # $0.04 an image, per million images: what the dashboard stores for $40 per 1,000.
    _price(client, master_key_header, "openai:dall-e-3", unit="images", rate=40000.0)
    headers = _user_key(client, master_key_header)

    with _held() as held:
        response = _generate(client, headers, _images(2), n=3)

    assert response.status_code == 200, response.text
    assert held == [Decimal("0.12")]
    row = _row(db_session_factory)
    assert row.cost == Decimal("0.08")
    assert row.pricing_source == "deployment"
    assert row.billing_meters == {"images": 2}
    assert row.pricing_breakdown == [{"meter": "images", "units": 2, "unit_rate": 0.04, "cost": 0.08}]
    _settled(client, master_key_header, 0.08)


def test_a_token_rate_on_images_keeps_its_per_image_reading_and_records_the_tokens(
    client: TestClient, master_key_header: dict[str, str], db_session_factory: Callable[[], Session]
) -> None:
    """The legacy reading: the money a configuration produced before the rate's unit decided."""
    _price(client, master_key_header, "openai:dall-e-3", unit="tokens", rate=0.04)
    headers = _user_key(client, master_key_header)
    usage = ImageUsage(
        input_tokens=50,
        output_tokens=4160,
        total_tokens=4210,
        input_tokens_details=UsageInputTokensDetails(image_tokens=0, text_tokens=50),
    )

    with _held() as held:
        response = _generate(client, headers, _images(1, usage), n=1)

    assert response.status_code == 200, response.text
    assert held == [Decimal("0.04")]
    row = _row(db_session_factory)
    assert row.cost == Decimal("0.04")
    assert row.billing_meters == {"images": 1}
    assert (row.prompt_tokens, row.completion_tokens, row.total_tokens) == (50, 4160, 4210)
    _settled(client, master_key_header, 0.04)


def test_a_requests_rate_on_images_bills_one_request_not_its_rate_per_image(
    client: TestClient, master_key_header: dict[str, str], db_session_factory: Callable[[], Session]
) -> None:
    """A per-request price read as dollars per image billed a million times too much."""
    _price(client, master_key_header, "openai:dall-e-3", unit="requests", rate=2000.0)
    headers = _user_key(client, master_key_header)

    with _held() as held:
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
    _price(client, master_key_header, "cohere:rerank-v3.5", unit="requests", rate=2000.0)
    headers = _user_key(client, master_key_header)
    result = RerankResponse(
        id="r",
        model="rerank-v3.5",
        results=[RerankResult(index=0, relevance_score=0.9)],
        meta=RerankMeta(billed_units={"search_units": 3}),
    )

    with _held() as held:
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
    _price(client, master_key_header, "openai:text-embedding-3-small", unit="requests", rate=5000.0)
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


def test_an_unpriced_transcription_is_recorded_free(
    client: TestClient, master_key_header: dict[str, str], db_session_factory: Callable[[], Session]
) -> None:
    headers = _user_key(client, master_key_header)

    response = _transcribe(client, headers)

    assert response.status_code == 200, response.text
    row = _row(db_session_factory)
    assert row.cost == Decimal(0)
    assert row.billing_meters is None
    _settled(client, master_key_header, 0.0)


def test_an_images_rate_on_a_per_request_endpoint_settles_free_rather_than_per_image(
    client: TestClient, master_key_header: dict[str, str], db_session_factory: Callable[[], Session]
) -> None:
    """The rate cannot price a request, and this endpoint's unpriced is free: never the rate read per image."""
    _price(client, master_key_header, "openai:whisper-1", unit="images", rate=0.5)
    headers = _user_key(client, master_key_header)

    response = _transcribe(client, headers)

    assert response.status_code == 200, response.text
    assert _row(db_session_factory).cost == Decimal(0)
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
    _price(strict_client, master, "openai:text-embedding-3-small", unit="images", rate=0.04)
    headers = _user_key(strict_client, master)
    provider = AsyncMock()

    with patch("gateway.api.routes.embeddings.aembedding", new=provider):
        response = strict_client.post(
            f"{API_ROOT}/embeddings", json={"model": "openai:text-embedding-3-small", "input": "hi"}, headers=headers
        )

    assert response.status_code == 402, response.text
    provider.assert_not_called()
    _settled(strict_client, master, 0.0)


def test_require_pricing_admits_a_rate_the_endpoint_can_settle(strict_client: TestClient) -> None:
    master = {API_KEY_HEADER: "Bearer test-master-key"}
    _price(strict_client, master, "openai:text-embedding-3-small", unit="tokens", rate=2.0)
    headers = _user_key(strict_client, master)

    response = _embed(strict_client, headers)

    assert response.status_code == 200, response.text
    assert response.json()["usage"]["cost_usd"] == "0.002000"
