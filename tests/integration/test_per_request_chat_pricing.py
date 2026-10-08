"""A completion model priced ``unit: requests`` is charged a flat amount per successful call.

Some upstreams served over chat completions bill per call and report no token
usage (an answer endpoint behind an OpenAI-compatible provider). These pin that
such a model reserves and settles one request's price, streamed or not, that a
failed call costs nothing, and that a model priced per token is unaffected.
"""

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

from gateway.api.routes import _pipeline
from gateway.core.config import API_KEY_HEADER, API_ROOT
from gateway.models.usage import UsageLog

from .conftest import MODEL_NAME

USER_ID = "per-request-user"
# $0.005 per request, stored as USD per million requests.
PER_REQUEST_RATE = 5000.0
FLAT_COST = Decimal("0.005")
# Long enough that a token estimate at the same rate (1,000 prompt tokens at
# $5,000 per million) would hold $5, a thousand times the flat price.
LONG_PROMPT = "x" * 4000
ZERO_USAGE = CompletionUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
REPORTED_USAGE = CompletionUsage(prompt_tokens=12, completion_tokens=34, total_tokens=46)


def _price(client: TestClient, headers: dict[str, str], *, unit: str, output: float = 0.0) -> None:
    response = client.post(
        f"{API_ROOT}/pricing",
        json={
            "model_key": MODEL_NAME,
            "input_price_per_million": PER_REQUEST_RATE,
            "output_price_per_million": output,
            "unit": unit,
        },
        headers=headers,
    )
    assert response.status_code == 200, response.text


def _user_key(client: TestClient, headers: dict[str, str], *, max_budget: float = 1.0) -> dict[str, str]:
    budget = client.post(f"{API_ROOT}/budgets", json={"max_budget": max_budget}, headers=headers)
    assert budget.status_code == 200, budget.text
    user = client.post(
        f"{API_ROOT}/users",
        json={"user_id": USER_ID, "budget_id": budget.json()["budget_id"]},
        headers=headers,
    )
    assert user.status_code == 200, user.text
    key = client.post(f"{API_ROOT}/keys", json={"key_name": "per-request-key", "user_id": USER_ID}, headers=headers)
    assert key.status_code == 200, key.text
    return {API_KEY_HEADER: f"Bearer {key.json()['key']}"}


def _completion(usage: CompletionUsage | None) -> ChatCompletion:
    return ChatCompletion(
        id="chatcmpl-per-request",
        object="chat.completion",
        created=0,
        model=MODEL_NAME,
        choices=[Choice(index=0, message=ChatCompletionMessage(role="assistant", content="hi"), finish_reason="stop")],
        usage=usage,
    )


def _chunks(usage: CompletionUsage | None) -> list[ChatCompletionChunk]:
    chunks = [
        ChatCompletionChunk(
            id="chunk-1",
            object="chat.completion.chunk",
            created=0,
            model=MODEL_NAME,
            choices=[ChunkChoice(index=0, delta=ChoiceDelta(role="assistant", content="hi"), finish_reason="stop")],
        )
    ]
    if usage is not None:
        chunks.append(
            ChatCompletionChunk(
                id="chunk-1", object="chat.completion.chunk", created=0, model=MODEL_NAME, choices=[], usage=usage
            )
        )
    return chunks


def _chat(
    client: TestClient,
    headers: dict[str, str],
    *,
    stream: bool,
    usage: CompletionUsage | None = ZERO_USAGE,
    fail: bool = False,
) -> tuple[Any, list[Decimal]]:
    """Send one chat request and return the response and every amount reserved for it."""

    async def _acompletion(**_kwargs: Any) -> Any:
        if fail and not stream:
            raise RuntimeError("upstream broke")
        if not stream:
            return _completion(usage)

        async def _stream() -> AsyncIterator[ChatCompletionChunk]:
            for chunk in _chunks(usage):
                yield chunk
            if fail:
                raise RuntimeError("upstream broke")

        return _stream()

    reserved: list[Decimal] = []
    real_reserve = _pipeline.reserve_budget

    async def _capture(*args: Any, **kwargs: Any) -> Any:
        reserved.append(Decimal(str(args[2])))
        return await real_reserve(*args, **kwargs)

    with (
        patch("gateway.api.routes.chat.acompletion", side_effect=_acompletion),
        patch.object(_pipeline, "reserve_budget", side_effect=_capture),
    ):
        response = client.post(
            f"{API_ROOT}/chat/completions",
            json={"model": MODEL_NAME, "messages": [{"role": "user", "content": LONG_PROMPT}], "stream": stream},
            headers=headers,
        )
    return response, reserved


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


def _user(client: TestClient, headers: dict[str, str]) -> dict[str, Any]:
    response = client.get(f"{API_ROOT}/users/{USER_ID}", headers=headers)
    assert response.status_code == 200, response.text
    body: dict[str, Any] = response.json()
    return body


@pytest.mark.parametrize("stream", [False, True])
def test_a_successful_call_is_reserved_and_settled_at_one_request(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
    stream: bool,
) -> None:
    _price(client, master_key_header, unit="requests")
    headers = _user_key(client, master_key_header)

    response, reserved = _chat(client, headers, stream=stream)

    assert response.status_code == 200, response.text
    assert reserved == [FLAT_COST]
    row = _row(db_session_factory)
    assert row.status == "success"
    assert row.cost == FLAT_COST
    assert row.billing_meters == {"requests": 1}
    assert row.pricing_breakdown == [{"meter": "request", "units": 1, "unit_rate": 0.005, "cost": 0.005}]
    assert row.pricing_source == "deployment"
    user = _user(client, master_key_header)
    assert user["spend"] == pytest.approx(0.005)
    assert user["reserved"] == pytest.approx(0.0)


@pytest.mark.parametrize("stream", [False, True])
def test_reported_tokens_are_recorded_but_not_priced(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
    stream: bool,
) -> None:
    # A non-zero output rate a token pricing would apply, so a token charge would show.
    _price(client, master_key_header, unit="requests", output=1_000_000.0)
    headers = _user_key(client, master_key_header)

    response, _ = _chat(client, headers, stream=stream, usage=REPORTED_USAGE)

    assert response.status_code == 200, response.text
    row = _row(db_session_factory)
    assert (row.prompt_tokens, row.completion_tokens, row.total_tokens) == (12, 34, 46)
    assert row.cost == FLAT_COST
    assert row.billing_meters == {"requests": 1}


@pytest.mark.parametrize("stream", [False, True])
def test_a_call_that_reports_no_usage_is_still_charged(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
    stream: bool,
) -> None:
    """For a stream this bypasses ``stream_missing_usage_policy``: the request is the billed unit."""
    _price(client, master_key_header, unit="requests")
    headers = _user_key(client, master_key_header)

    response, _ = _chat(client, headers, stream=stream, usage=None)

    assert response.status_code == 200, response.text
    row = _row(db_session_factory)
    assert row.status == "success"
    assert row.cost == FLAT_COST
    assert row.pricing_breakdown == [{"meter": "request", "units": 1, "unit_rate": 0.005, "cost": 0.005}]
    user = _user(client, master_key_header)
    assert user["spend"] == pytest.approx(0.005)
    assert user["reserved"] == pytest.approx(0.0)


@pytest.mark.parametrize("stream", [False, True])
def test_a_failed_call_costs_nothing_and_releases_its_hold(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
    stream: bool,
) -> None:
    _price(client, master_key_header, unit="requests")
    headers = _user_key(client, master_key_header)

    # The stream fails after reporting usage, which a token-priced model would still owe for.
    response, reserved = _chat(client, headers, stream=stream, usage=REPORTED_USAGE, fail=True)

    if stream:
        assert response.status_code == 200, response.text
        assert "An error occurred during streaming" in response.text
    else:
        assert response.status_code >= 500, response.text
    assert reserved == [FLAT_COST]
    row = _row(db_session_factory)
    assert row.status == "error"
    assert not row.cost
    assert not row.pricing_breakdown
    user = _user(client, master_key_header)
    assert user["spend"] == pytest.approx(0.0)
    assert user["reserved"] == pytest.approx(0.0)


def test_the_budget_gate_admits_against_the_flat_price(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """A budget below one request's price refuses the call before it is made."""
    _price(client, master_key_header, unit="requests")
    headers = _user_key(client, master_key_header, max_budget=0.004)

    response, _ = _chat(client, headers, stream=False)

    assert response.status_code == 403, response.text


@pytest.mark.parametrize("stream", [False, True])
def test_a_model_priced_per_token_is_unchanged(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
    stream: bool,
) -> None:
    """The same rate under ``unit: tokens`` reserves a token estimate and settles the reported tokens."""
    _price(client, master_key_header, unit="tokens")
    headers = _user_key(client, master_key_header, max_budget=100.0)

    response, reserved = _chat(client, headers, stream=stream, usage=REPORTED_USAGE)

    assert response.status_code == 200, response.text
    # At least the 1,000 prompt tokens the 4,000-character message alone makes, at
    # $5,000 per million; the estimate also counts the request's other characters.
    assert len(reserved) == 1
    assert reserved[0] >= Decimal("5")
    row = _row(db_session_factory)
    assert row.cost == Decimal("0.06")
    assert row.billing_meters is not None
    assert row.billing_meters["total_input_tokens"] == 12
    assert row.pricing_breakdown == [{"meter": "input", "units": 12, "rate_per_million": 5000.0, "cost": 0.06}]
    user = _user(client, master_key_header)
    assert user["spend"] == pytest.approx(0.06)
    assert user["reserved"] == pytest.approx(0.0)
