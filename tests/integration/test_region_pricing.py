"""A request is held at the region it asks to run in and settled at the region the provider reports."""

import time
from collections.abc import Callable
from decimal import Decimal
from typing import Any, cast
from unittest.mock import patch

import pytest
from any_llm.types.completion import ChatCompletion, ChatCompletionMessage, Choice, CompletionUsage
from any_llm.types.messages import MessageResponse, MessageUsage, TextBlock
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from gateway.core.config import API_KEY_HEADER, API_ROOT
from gateway.models.usage import UsageLog
from gateway.services.budgets import reserve_budget

USER_ID = "region-user"
MODEL = "anthropic:claude-opus-4-6"
US = {"region": {"equals": "us"}}
RATES = {"text": {"input": {"price": 1}, "output": {"price": 10}}, "multipliers": [{"when": US, "factor": 1.1}]}
# 1,000 input tokens at $1 and 100 output tokens at $10, per million.
BASE_COST = Decimal("0.002")


@pytest.fixture
def headers(client: TestClient, master_key_header: dict[str, str]) -> dict[str, str]:
    priced = client.post(f"{API_ROOT}/pricing", json={"model_key": MODEL, "rates": RATES}, headers=master_key_header)
    assert priced.status_code == 200, priced.text
    budget = client.post(f"{API_ROOT}/budgets", json={"max_budget": 10.0}, headers=master_key_header)
    assert budget.status_code == 200, budget.text
    user = client.post(
        f"{API_ROOT}/users",
        json={"user_id": USER_ID, "budget_id": budget.json()["budget_id"]},
        headers=master_key_header,
    )
    assert user.status_code == 200, user.text
    key = client.post(
        f"{API_ROOT}/keys", json={"key_name": "region-key", "user_id": USER_ID}, headers=master_key_header
    )
    assert key.status_code == 200, key.text
    return {API_KEY_HEADER: f"Bearer {key.json()['key']}"}


def _message(region: str | None) -> MessageResponse:
    return MessageResponse(
        id="msg_region",
        type="message",
        role="assistant",
        model="claude-opus-4-6",
        content=[TextBlock(type="text", text="hi", citations=None)],
        stop_reason=cast(Any, "end_turn"),
        stop_sequence=None,
        usage=MessageUsage(input_tokens=1000, output_tokens=100, inference_geo=region),
        container=None,
    )


def _send_message(client: TestClient, headers: dict[str, str], *, asked: str | None, served: str | None) -> Decimal:
    """Send one Messages request asking to run in ``asked`` that the provider runs in ``served``; return its hold."""
    held: list[Decimal] = []
    sent: dict[str, Any] = {}

    async def _amessages(**kwargs: Any) -> MessageResponse:
        sent.update(kwargs)
        return _message(served)

    async def _capture(*args: Any, **kwargs: Any) -> Any:
        held.append(Decimal(str(args[2])))
        return await reserve_budget(*args, **kwargs)

    body: dict[str, Any] = {"model": MODEL, "messages": [{"role": "user", "content": "hello"}], "max_tokens": 100}
    if asked is not None:
        body["inference_geo"] = asked
    with (
        patch("gateway.api.routes.messages.amessages", new=_amessages),
        patch("gateway.api.routes._pipeline.reserve_budget", side_effect=_capture),
    ):
        response = client.post(f"{API_ROOT}/messages", json=body, headers=headers)
    assert response.status_code == 200, response.text
    assert sent.get("inference_geo") == asked
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


def test_a_message_is_settled_at_the_region_the_provider_ran_it_in(
    client: TestClient, headers: dict[str, str], db_session_factory: Callable[[], Session]
) -> None:
    _send_message(client, headers, asked="us", served="us")
    _send_message(client, headers, asked=None, served="us")
    _send_message(client, headers, asked="us", served=None)
    _send_message(client, headers, asked=None, served="global")

    asked_us, default_us, unreported, global_ = _rows(db_session_factory, 4)

    assert asked_us.cost == default_us.cost == BASE_COST * Decimal("1.1")
    assert [line["rules"] for line in default_us.pricing_breakdown or []] == [[US], [US]]
    assert unreported.cost == global_.cost == BASE_COST


def test_the_hold_is_priced_at_the_region_asked_for(client: TestClient, headers: dict[str, str]) -> None:
    base = _send_message(client, headers, asked=None, served=None)
    us = _send_message(client, headers, asked="us", served="us")

    assert base > 0
    assert us == base * Decimal("1.1")


def test_a_chat_completion_is_settled_at_the_region_its_usage_reports(
    client: TestClient, headers: dict[str, str], db_session_factory: Callable[[], Session]
) -> None:
    async def _acompletion(**_kwargs: Any) -> ChatCompletion:
        return ChatCompletion(
            id="chatcmpl-region",
            object="chat.completion",
            created=0,
            model=MODEL,
            choices=[
                Choice(index=0, message=ChatCompletionMessage(role="assistant", content="hi"), finish_reason="stop")
            ],
            usage=CompletionUsage(prompt_tokens=1000, completion_tokens=100, total_tokens=1100, inference_geo="us"),
        )

    with patch("gateway.api.routes.chat.acompletion", side_effect=_acompletion):
        response = client.post(
            f"{API_ROOT}/chat/completions",
            json={"model": MODEL, "messages": [{"role": "user", "content": "hello"}]},
            headers=headers,
        )
    assert response.status_code == 200, response.text

    (row,) = _rows(db_session_factory, 1)

    assert row.cost == BASE_COST * Decimal("1.1")
