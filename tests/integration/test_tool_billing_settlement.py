"""Settlement tests for gateway-run tool calls.

The point of these is ``users.spend``, not the usage row. A tool charge that
lands on the row but not in the spend ledger is the exact defect this feature
exists to close: ``refund_reservation`` deliberately releases a hold *without*
recording spend (``services/budget_service.py``), so every failure path that
runs tool calls has to reconcile instead of refund. Asserting only the row would
pass while the budget silently leaked.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from any_llm.types.completion import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageFunctionToolCall,
    Choice,
    CompletionUsage,
    Function,
)
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from gateway.core.config import API_KEY_HEADER
from gateway.models.entities import UsageLog, User
from gateway.services.tool_usage import TOOL_METER_NAMESPACE

from .conftest import MODEL_NAME

# Reuse the fail-closed client from the require_pricing suite: the tool gate has the
# same posture and there is no reason to stand up a second one.
from .test_require_pricing import strict_pricing_client as strict_pricing_client  # noqa: PLC0414

# $0.01 per call, in the per-million-requests convention ``flat_request_cost`` uses.
_SEARCH_RATE_PER_MILLION = 10_000.0
_SEARCH_UNIT_COST = 0.01


def _completion(*, tool_call: bool, prompt_tokens: int = 10, completion_tokens: int = 5) -> ChatCompletion:
    """A provider response, optionally asking for the gateway's web_search tool."""
    tool_calls = (
        [
            ChatCompletionMessageFunctionToolCall(
                id="call_1",
                type="function",
                function=Function(name="web_search", arguments='{"query": "otari"}'),
            )
        ]
        if tool_call
        else None
    )
    message = ChatCompletionMessage(
        role="assistant",
        content=None if tool_call else "done",
        tool_calls=tool_calls,
    )
    return ChatCompletion(
        id="chatcmpl-test",
        created=0,
        model=MODEL_NAME,
        object="chat.completion",
        choices=[
            Choice(
                finish_reason="tool_calls" if tool_call else "stop",
                index=0,
                message=message,
            )
        ],
        usage=CompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


@pytest.fixture
def search_pricing(client: TestClient, master_key_header: dict[str, str]) -> dict[str, Any]:
    """Price the gateway's own web_search tool under ``otari:web_search``."""
    response = client.post(
        "/v1/pricing",
        json={
            "model_key": "otari:web_search",
            "input_price_per_million": _SEARCH_RATE_PER_MILLION,
            "output_price_per_million": 0.0,
        },
        headers=master_key_header,
    )
    assert response.status_code == 200
    return dict(response.json())


def _spend(db_session_factory: Callable[[], Session], user_id: str) -> float:
    db = db_session_factory()
    try:
        user = db.query(User).filter(User.user_id == user_id).first()
        return float(user.spend) if user else 0.0
    finally:
        db.close()


def _latest_row(db_session_factory: Callable[[], Session], user_id: str) -> UsageLog:
    db = db_session_factory()
    try:
        row = db.query(UsageLog).filter(UsageLog.user_id == user_id).order_by(UsageLog.timestamp.desc()).first()
        assert row is not None, "expected a usage row for the request"
        db.expunge(row)
        return row
    finally:
        db.close()


@pytest.mark.asyncio
async def test_search_calls_are_metered_priced_and_spent(
    client: TestClient,
    api_key_header: dict[str, str],
    api_key_obj: dict[str, Any],
    model_pricing: dict[str, Any],
    search_pricing: dict[str, Any],
    db_session_factory: Callable[[], Session],
) -> None:
    """Two searches land on the row as a meter, a charge line, cost, AND spend."""
    user_id = api_key_obj["user_id"]
    before = _spend(db_session_factory, user_id)

    # Iteration 1 and 2 each ask for a search; iteration 3 answers.
    responses = [
        _completion(tool_call=True),
        _completion(tool_call=True),
        _completion(tool_call=False),
    ]

    with (
        # The tool loop resolves ``acompletion`` as a module global at call time,
        # so patching it there is what intercepts every iteration.
        patch("gateway.services.mcp_loop.acompletion", new=AsyncMock(side_effect=responses)),
        patch(
            "gateway.services.web_search_backend.WebSearchBackend._search_tool",
            new=AsyncMock(return_value="search results for otari"),
        ),
        patch.dict("os.environ", {"OTARI_WEB_SEARCH_URL": "http://web-search.invalid"}),
    ):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": "what is otari"}],
                "tools": [{"type": "otari_web_search"}],
            },
            headers=api_key_header,
        )

    assert response.status_code == 200, response.text
    row = _latest_row(db_session_factory, user_id)

    tools = (row.billing_meters or {})[TOOL_METER_NAMESPACE]
    assert tools["web_search"]["billed"] == 2
    assert tools["web_search"]["errors"] == 0
    assert tools["web_search"]["unit_rate"] == pytest.approx(_SEARCH_UNIT_COST)

    line = next(entry for entry in (row.pricing_breakdown or []) if entry["meter"] == "web_search_calls")
    assert line["units"] == 2
    assert line["cost"] == pytest.approx(2 * _SEARCH_UNIT_COST)

    # The row's cost carries the searches on top of the tokens.
    assert row.cost is not None
    assert row.cost >= 2 * _SEARCH_UNIT_COST

    # The part that matters: the money reached the ledger, not just the log.
    after = _spend(db_session_factory, user_id)
    assert after - before == pytest.approx(row.cost, rel=1e-6)


@pytest.mark.asyncio
async def test_failed_search_is_counted_but_never_billed(
    client: TestClient,
    api_key_header: dict[str, str],
    api_key_obj: dict[str, Any],
    model_pricing: dict[str, Any],
    search_pricing: dict[str, Any],
    db_session_factory: Callable[[], Session],
) -> None:
    """A search that fails is visible as an error and costs nothing.

    The backends return ``[tool error] …`` as an ordinary value rather than
    raising, so classification is on the sentinel; billing every returned string
    would charge for failures.
    """
    user_id = api_key_obj["user_id"]
    responses = [_completion(tool_call=True), _completion(tool_call=False)]

    with (
        patch("gateway.services.mcp_loop.acompletion", new=AsyncMock(side_effect=responses)),
        patch(
            "gateway.services.web_search_backend.WebSearchBackend._search_tool",
            new=AsyncMock(return_value="[tool error] backend unreachable"),
        ),
        patch.dict("os.environ", {"OTARI_WEB_SEARCH_URL": "http://web-search.invalid"}),
    ):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": "what is otari"}],
                "tools": [{"type": "otari_web_search"}],
            },
            headers=api_key_header,
        )

    assert response.status_code == 200, response.text
    row = _latest_row(db_session_factory, user_id)
    tools = (row.billing_meters or {})[TOOL_METER_NAMESPACE]
    assert tools["web_search"] == {"billed": 0, "errors": 1}
    assert not any(entry["meter"] == "web_search_calls" for entry in (row.pricing_breakdown or []))


@pytest.mark.asyncio
async def test_unpriced_tool_is_refused_when_require_pricing_is_on(
    strict_pricing_client: TestClient,
) -> None:
    """No ``otari:web_search`` price means a 402 before the provider is called.

    Same posture as an unpriced model. The check runs at admission, so nothing
    upstream is spent on a request that could never be billed. Note this fixture
    set deliberately omits ``search_pricing``.
    """
    strict_pricing_client.post(
        "/v1/users", json={"user_id": "tool-user"}, headers={API_KEY_HEADER: "Bearer test-master-key"}
    )
    # Price the model so the model gate passes and the tool gate is what fires.
    strict_pricing_client.post(
        "/v1/pricing",
        json={"model_key": "openai:gpt-4o", "input_price_per_million": 1.0, "output_price_per_million": 1.0},
        headers={API_KEY_HEADER: "Bearer test-master-key"},
    )

    with (
        patch("gateway.services.mcp_loop.acompletion", new=AsyncMock()) as provider,
        patch.dict("os.environ", {"OTARI_WEB_SEARCH_URL": "http://web-search.invalid"}),
    ):
        response = strict_pricing_client.post(
            "/v1/chat/completions",
            json={
                "model": "openai:gpt-4o",
                "messages": [{"role": "user", "content": "hi"}],
                "user": "tool-user",
                "tools": [{"type": "otari_web_search"}],
            },
            headers={API_KEY_HEADER: "Bearer test-master-key"},
        )

    assert response.status_code == 402, response.text
    # Chat completions map a gateway rejection to a bare ``detail`` string.
    detail = response.json()["detail"]
    assert "otari:web_search" in detail
    assert "require_pricing" in detail
    provider.assert_not_called()
