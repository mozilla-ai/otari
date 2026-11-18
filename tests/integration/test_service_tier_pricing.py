"""A completion is held at the tier it asks for and settled at the tier the provider reports serving it at."""

import time
from collections.abc import AsyncIterator, Callable
from decimal import Decimal
from typing import Any
from unittest.mock import patch

import pytest
from any_llm.types.completion import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    Choice,
    ChoiceDelta,
    ChunkChoice,
    CompletionUsage,
)
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from gateway.core.config import API_KEY_HEADER, API_ROOT
from gateway.models.usage import UsageLog
from gateway.services.budgets import reserve_budget

from .conftest import MODEL_NAME

USER_ID = "service-tier-user"
FLEX = {"service_tier": {"equals": "flex"}}
PRIORITY = {"service_tier": {"equals": "priority"}}
RATES = {
    "text": {"input": {"price": 1}, "output": {"price": 10}},
    "multipliers": [{"when": FLEX, "factor": 0.5}, {"when": PRIORITY, "factor": 2}],
}
USAGE = CompletionUsage(prompt_tokens=1000, completion_tokens=100, total_tokens=1100)
# 1,000 input tokens at $1 and 100 output tokens at $10, per million.
BASE_COST = Decimal("0.002")


@pytest.fixture
def headers(client: TestClient, master_key_header: dict[str, str]) -> dict[str, str]:
    priced = client.post(
        f"{API_ROOT}/pricing", json={"model_key": MODEL_NAME, "rates": RATES}, headers=master_key_header
    )
    assert priced.status_code == 200, priced.text
    budget = client.post(f"{API_ROOT}/budgets", json={"max_budget": 10.0}, headers=master_key_header)
    assert budget.status_code == 200, budget.text
    user = client.post(
        f"{API_ROOT}/users",
        json={"user_id": USER_ID, "budget_id": budget.json()["budget_id"]},
        headers=master_key_header,
    )
    assert user.status_code == 200, user.text
    key = client.post(f"{API_ROOT}/keys", json={"key_name": "tier-key", "user_id": USER_ID}, headers=master_key_header)
    assert key.status_code == 200, key.text
    return {API_KEY_HEADER: f"Bearer {key.json()['key']}"}


def _chat(
    client: TestClient,
    headers: dict[str, str],
    *,
    stream: bool,
    asked: str | None,
    served: str | None,
    usage: CompletionUsage | None = USAGE,
    usage_names_tier: bool = True,
) -> Decimal:
    """Send one chat request asking for ``asked`` that the provider serves at ``served``; return what it held.

    ``usage_names_tier`` False sends the tier on the stream's content chunk alone.
    """

    async def _acompletion(**_kwargs: Any) -> Any:
        if not stream:
            return ChatCompletion(
                id="chatcmpl-tier",
                object="chat.completion",
                created=0,
                model=MODEL_NAME,
                choices=[
                    Choice(index=0, message=ChatCompletionMessage(role="assistant", content="hi"), finish_reason="stop")
                ],
                usage=usage,
                service_tier=served,
            )

        async def _stream() -> AsyncIterator[ChatCompletionChunk]:
            yield ChatCompletionChunk(
                id="chunk-1",
                object="chat.completion.chunk",
                created=0,
                model=MODEL_NAME,
                choices=[ChunkChoice(index=0, delta=ChoiceDelta(role="assistant", content="hi"), finish_reason="stop")],
                service_tier=served,
            )
            yield ChatCompletionChunk(
                id="chunk-1",
                object="chat.completion.chunk",
                created=0,
                model=MODEL_NAME,
                choices=[],
                usage=usage,
                service_tier=served if usage_names_tier else None,
            )

        return _stream()

    held: list[Decimal] = []

    async def _capture(*args: Any, **kwargs: Any) -> Any:
        held.append(Decimal(str(args[2])))
        return await reserve_budget(*args, **kwargs)

    body: dict[str, Any] = {"model": MODEL_NAME, "messages": [{"role": "user", "content": "hello"}], "stream": stream}
    if asked is not None:
        body["service_tier"] = asked
    with (
        patch("gateway.api.routes.chat.acompletion", side_effect=_acompletion),
        patch("gateway.api.routes._pipeline.reserve_budget", side_effect=_capture),
    ):
        response = client.post(f"{API_ROOT}/chat/completions", json=body, headers=headers)
    assert response.status_code == 200, response.text
    if stream:
        assert "[DONE]" in response.text
    assert len(held) == 1
    return held[0]


def _rows(make_session: Callable[[], Session], count: int, *, timeout: float = 3.0) -> list[UsageLog]:
    """The user's usage rows, oldest first, polled because a background log writer commits them."""
    deadline = time.monotonic() + timeout
    while True:
        with make_session() as db:
            rows = list(
                db.execute(select(UsageLog).where(UsageLog.user_id == USER_ID).order_by(UsageLog.timestamp)).scalars()
            )
            if len(rows) >= count:
                for row in rows:
                    db.expunge(row)
                return rows
        assert time.monotonic() < deadline, f"only {len(rows)} of {count} usage rows were written"
        time.sleep(0.1)


@pytest.mark.parametrize("stream", [False, True])
def test_a_request_is_settled_at_the_tier_the_provider_served(
    client: TestClient, headers: dict[str, str], db_session_factory: Callable[[], Session], stream: bool
) -> None:
    _chat(client, headers, stream=stream, asked="flex", served="flex")
    _chat(client, headers, stream=stream, asked=None, served="priority")

    flex, priority = _rows(db_session_factory, 2)

    assert flex.cost == BASE_COST / 2
    assert [line["rules"] for line in flex.pricing_breakdown or []] == [[FLEX], [FLEX]]
    assert priority.cost == BASE_COST * 2
    assert [line["rules"] for line in priority.pricing_breakdown or []] == [[PRIORITY], [PRIORITY]]


@pytest.mark.parametrize("stream", [False, True])
def test_a_tier_asked_for_and_not_served_is_not_billed(
    client: TestClient, headers: dict[str, str], db_session_factory: Callable[[], Session], stream: bool
) -> None:
    _chat(client, headers, stream=stream, asked="priority", served="default")
    _chat(client, headers, stream=stream, asked="flex", served=None)

    rows = _rows(db_session_factory, 2)

    assert [row.cost for row in rows] == [BASE_COST, BASE_COST]
    assert all("rules" not in line for row in rows for line in row.pricing_breakdown or [])


def test_the_hold_is_priced_at_the_tier_asked_for(client: TestClient, headers: dict[str, str]) -> None:
    base = _chat(client, headers, stream=False, asked=None, served=None)
    flex = _chat(client, headers, stream=False, asked="flex", served="flex")
    priority = _chat(client, headers, stream=False, asked="priority", served="priority")

    assert base > 0
    assert (flex, priority) == (base / 2, base * 2)


def test_a_stream_is_settled_at_the_tier_a_content_chunk_reported(
    client: TestClient, headers: dict[str, str], db_session_factory: Callable[[], Session]
) -> None:
    _chat(client, headers, stream=True, asked=None, served="flex", usage_names_tier=False)

    (row,) = _rows(db_session_factory, 1)

    assert row.cost == BASE_COST / 2


def test_a_flat_rate_with_no_usage_is_settled_at_the_tier_its_result_reported(
    client: TestClient,
    headers: dict[str, str],
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
) -> None:
    flat = {"text": {"input": {"price": 0.01, "unit": "request"}}, "multipliers": [{"when": FLEX, "factor": 0.5}]}
    priced = client.post(
        f"{API_ROOT}/pricing", json={"model_key": MODEL_NAME, "rates": flat}, headers=master_key_header
    )
    assert priced.status_code == 200, priced.text

    _chat(client, headers, stream=False, asked=None, served="flex", usage=None)

    (row,) = _rows(db_session_factory, 1)

    assert row.cost == Decimal("0.005")
