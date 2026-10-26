"""Settlement tests for gateway-run tool calls.

These tests assert ``users.spend`` and not only the usage row.
``refund_reservation`` releases a hold without recording spend, so a failure path that ran tool calls must reconcile.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from decimal import Decimal
from typing import Any, cast
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
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from gateway.api.routes.chat import log_usage
from gateway.core.config import API_KEY_HEADER, API_ROOT, GatewayConfig
from gateway.core.settings.pricing import PricingConfig
from gateway.core.usage import GatewayUsage
from gateway.models.pricing import ModelPricing
from gateway.models.usage import UsageLog
from gateway.models.users import User
from gateway.services.pricing_service import configure_default_pricing
from gateway.services.tool_usage import TOOL_METER_NAMESPACE, ToolUsageTally
from gateway.services.web_fetch_service import WebFetchResult, WebFetchService
from gateway.services.web_retrieval_policy import canonicalize_web_url

from .conftest import MODEL_NAME, build_test_client, declared_test_providers

# Reuse the fail-closed client from the require_pricing suite: the tool gate has the
# same posture and there is no reason to stand up a second one.
from .test_require_pricing import strict_pricing_client as strict_pricing_client  # noqa: PLC0414

# $0.01 per call, in the per-million-requests convention ``flat_request_cost`` uses.
_SEARCH_RATE_PER_MILLION = 10_000.0
_SEARCH_UNIT_COST = 0.01
_FAKE_FETCH = "fake-fetch"
_FAKE_SEARCH = "fake-search"


def _completion(
    *,
    tool_call: bool,
    tool_name: str = "web_search",
    arguments: str = '{"query": "otari"}',
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
) -> ChatCompletion:
    """A provider response, optionally asking for the gateway's web_search tool."""
    tool_calls = (
        [
            ChatCompletionMessageFunctionToolCall(
                id="call_1",
                type="function",
                function=Function(name=tool_name, arguments=arguments),
            )
        ]
        if tool_call
        else None
    )
    message = ChatCompletionMessage(
        role="assistant",
        content=None if tool_call else "done",
        tool_calls=cast(Any, tool_calls),
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


def _fetch_batch_completion(count: int) -> ChatCompletion:
    calls = [
        ChatCompletionMessageFunctionToolCall(
            id=f"call_{index}",
            type="function",
            function=Function(name="web_fetch", arguments='{"url":"https://example.com"}'),
        )
        for index in range(count)
    ]
    return ChatCompletion(
        id="chatcmpl-fetch-batch",
        created=0,
        model=MODEL_NAME,
        object="chat.completion",
        choices=[
            Choice(
                finish_reason="tool_calls",
                index=0,
                message=ChatCompletionMessage(role="assistant", content=None, tool_calls=cast(Any, calls)),
            )
        ],
        usage=CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


def _serve_fetch_on_fake(monkeypatch: pytest.MonkeyPatch, config: GatewayConfig, **options: Any) -> None:
    """Serve the fetch tool on a ``fake`` fetch instance, named in ``web_fetch_default_tool``, with these options."""
    monkeypatch.setattr(config, "web_fetch_enabled", True)
    monkeypatch.setattr(config, "fetch_tools", {_FAKE_FETCH: {"provider": "fake", "options": options}})
    monkeypatch.setattr(config, "web_fetch_default_tool", _FAKE_FETCH)


@pytest.fixture
def search_pricing(client: TestClient, master_key_header: dict[str, str]) -> dict[str, Any]:
    """Price the gateway's own web_search tool under ``otari:web_search``."""
    response = client.post(
        f"{API_ROOT}/pricing",
        json={
            "model_key": "otari:web_search",
            "input_price_per_million": _SEARCH_RATE_PER_MILLION,
            "output_price_per_million": 0.0,
        },
        headers=master_key_header,
    )
    assert response.status_code == 200
    return dict(response.json())


@pytest.fixture
def fetch_pricing(client: TestClient, master_key_header: dict[str, str]) -> dict[str, Any]:
    response = client.post(
        f"{API_ROOT}/pricing",
        json={
            "model_key": "otari:web_fetch",
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
            "gateway.services.web_retrieval_backend.WebSearchBackend._search_tool",
            new=AsyncMock(return_value="search results for otari"),
        ),
        patch.dict("os.environ", {"OTARI_WEB_SEARCH_URL": "http://web-search.invalid"}),
    ):
        response = client.post(
            f"{API_ROOT}/chat/completions",
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
    assert line["label"] == "otari:web_search"
    assert line["units"] == 2
    assert line["cost"] == pytest.approx(2 * _SEARCH_UNIT_COST)

    # The row's cost carries the searches on top of the tokens.
    assert row.cost is not None
    assert row.cost >= 2 * _SEARCH_UNIT_COST

    # The part that matters: the money reached the ledger, not just the log.
    after = _spend(db_session_factory, user_id)
    assert after - before == pytest.approx(float(row.cost), rel=1e-6)


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
            "gateway.services.web_retrieval_backend.WebSearchBackend._search_tool",
            new=AsyncMock(return_value="[tool error] backend unreachable"),
        ),
        patch.dict("os.environ", {"OTARI_WEB_SEARCH_URL": "http://web-search.invalid"}),
    ):
        response = client.post(
            f"{API_ROOT}/chat/completions",
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
async def test_a_search_with_no_hits_is_billed_at_the_rate(
    client: TestClient,
    api_key_header: dict[str, str],
    api_key_obj: dict[str, Any],
    model_pricing: dict[str, Any],
    search_pricing: dict[str, Any],
    db_session_factory: Callable[[], Session],
) -> None:
    """A zero-result success is billed; the old backend reports no cost, so the rate prices it."""
    user_id = api_key_obj["user_id"]
    responses = [_completion(tool_call=True), _completion(tool_call=False)]

    with (
        patch("gateway.services.mcp_loop.acompletion", new=AsyncMock(side_effect=responses)),
        patch("gateway.services.web_retrieval_backend.WebSearchBackend._search", new=AsyncMock(return_value=[])),
        patch.dict("os.environ", {"OTARI_WEB_SEARCH_URL": "http://web-search.invalid"}),
    ):
        response = client.post(
            f"{API_ROOT}/chat/completions",
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": "what is otari"}],
                "tools": [{"type": "otari_web_search"}],
            },
            headers=api_key_header,
        )

    assert response.status_code == 200, response.text
    row = _latest_row(db_session_factory, user_id)
    assert (row.billing_meters or {})[TOOL_METER_NAMESPACE]["web_search"]["billed"] == 1
    line = next(entry for entry in (row.pricing_breakdown or []) if entry["meter"] == "web_search_calls")
    assert line == {
        "meter": "web_search_calls",
        "label": "otari:web_search",
        "units": 1,
        "unit_rate": pytest.approx(_SEARCH_UNIT_COST),
        "cost": pytest.approx(_SEARCH_UNIT_COST),
        # The legacy settings' search runs on the deployment's account.
        "deployment_paid": True,
    }


def _serve_search_on_fake(
    monkeypatch: pytest.MonkeyPatch, config: GatewayConfig, search: dict[str, Any], fetch: dict[str, Any]
) -> None:
    """Serve in-loop search on a ``fake`` instance with ``search`` options, enriched by a ``fake`` fetch instance."""
    monkeypatch.delenv("OTARI_WEB_SEARCH_URL", raising=False)
    monkeypatch.setattr(
        config,
        "search_tools",
        {_FAKE_SEARCH: {"provider": "fake", "options": search, "fetch_tool": _FAKE_FETCH}},
    )
    monkeypatch.setattr(config, "fetch_tools", {_FAKE_FETCH: {"provider": "fake", "options": fetch}})
    monkeypatch.setattr(config, "web_search_default_tool", _FAKE_SEARCH)


def _search_once(client: TestClient, headers: dict[str, str]) -> None:
    """One chat request whose model searches once, then answers."""
    responses = [_completion(tool_call=True), _completion(tool_call=False)]
    with patch("gateway.services.mcp_loop.acompletion", new=AsyncMock(side_effect=responses)):
        response = client.post(
            f"{API_ROOT}/chat/completions",
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": "what is otari"}],
                "tools": [{"type": "otari_web_search"}],
            },
            headers=headers,
        )
    assert response.status_code == 200, response.text


@pytest.mark.asyncio
async def test_a_search_on_an_instance_is_priced_at_its_own_row_with_its_enrichment(
    client: TestClient,
    api_key_header: dict[str, str],
    api_key_obj: dict[str, Any],
    master_key_header: dict[str, str],
    model_pricing: dict[str, Any],
    search_pricing: dict[str, Any],
    db_session_factory: Callable[[], Session],
    monkeypatch: pytest.MonkeyPatch,
    test_config: GatewayConfig,
) -> None:
    """``fake:<instance>`` wins over ``otari:web_search``, and what the enrichment fetcher reported adds to it."""
    # Of the fake search's three canned hits, two come without page text and are enriched, at 0.002 each.
    _serve_search_on_fake(monkeypatch, test_config, search={}, fetch={"cost": 0.002})
    response = client.post(
        f"{API_ROOT}/pricing",
        json={
            "model_key": f"fake:{_FAKE_SEARCH}",
            "input_price_per_million": 4_000.0,
            "output_price_per_million": 0.0,
        },
        headers=master_key_header,
    )
    assert response.status_code == 200, response.text

    _search_once(client, api_key_header)

    row = _latest_row(db_session_factory, api_key_obj["user_id"])
    [line] = [entry for entry in (row.pricing_breakdown or []) if entry["meter"] == "web_search_calls"]
    assert (line["label"], line["units"]) == (f"fake:{_FAKE_SEARCH}", 1)
    assert line["cost"] == pytest.approx(0.004 + 0.004)
    assert set((row.billing_meters or {})[TOOL_METER_NAMESPACE]) == {"web_search"}, "enrichment records no fetch"


@pytest.mark.asyncio
async def test_a_search_on_an_instance_takes_its_search_row_and_an_operator_override_on_its_old_key(
    client: TestClient,
    api_key_header: dict[str, str],
    api_key_obj: dict[str, Any],
    master_key_header: dict[str, str],
    model_pricing: dict[str, Any],
    search_pricing: dict[str, Any],
    db_session_factory: Callable[[], Session],
    monkeypatch: pytest.MonkeyPatch,
    test_config: GatewayConfig,
) -> None:
    """The instance's search row wins over its old ``fake:<instance>`` row; an operator's override wins over both."""
    _serve_search_on_fake(monkeypatch, test_config, search={}, fetch={})
    for body in (
        {"model_key": f"fake:{_FAKE_SEARCH}", "input_price_per_million": 4_000.0},
        {"kind": "search", "model_key": _FAKE_SEARCH, "input_price_per_million": 2_500.0},
    ):
        response = client.post(
            f"{API_ROOT}/pricing", json={**body, "output_price_per_million": 0.0}, headers=master_key_header
        )
        assert response.status_code == 200, response.text

    def billed() -> tuple[str, float]:
        """The label and cost of the newest usage row's search line."""
        row = _latest_row(db_session_factory, api_key_obj["user_id"])
        [line] = [entry for entry in (row.pricing_breakdown or []) if entry["meter"] == "web_search_calls"]
        return line["label"], line["cost"]

    _search_once(client, api_key_header)
    label, cost = billed()
    # The label still names what served the call.
    assert label == f"fake:{_FAKE_SEARCH}"
    assert cost == pytest.approx(0.0025)

    # Every override is tried before any deployment price, the old key included.
    override = client.post(
        f"{API_ROOT}/organizations/me/pricing",
        json={"model_key": f"fake:{_FAKE_SEARCH}", "input_price_per_million": 1_000.0, "output_price_per_million": 0.0},
        headers=master_key_header,
    )
    assert override.status_code == 201, override.text
    _search_once(client, api_key_header)
    assert billed()[1] == pytest.approx(0.001)


@pytest.mark.asyncio
async def test_a_search_whose_provider_reports_its_cost_is_billed_that_cost(
    client: TestClient,
    api_key_header: dict[str, str],
    api_key_obj: dict[str, Any],
    model_pricing: dict[str, Any],
    search_pricing: dict[str, Any],
    db_session_factory: Callable[[], Session],
    monkeypatch: pytest.MonkeyPatch,
    test_config: GatewayConfig,
) -> None:
    """The reported cost wins over the ``otari:web_search`` fallback, and an enrichment reporting none adds none."""
    _serve_search_on_fake(monkeypatch, test_config, search={"cost": 0.005}, fetch={})

    _search_once(client, api_key_header)

    row = _latest_row(db_session_factory, api_key_obj["user_id"])
    [line] = [entry for entry in (row.pricing_breakdown or []) if entry["meter"] == "web_search_calls"]
    assert (line["label"], line["units"]) == (f"fake:{_FAKE_SEARCH}", 1)
    assert line["cost"] == pytest.approx(0.005)


class _Writer:
    def __init__(self) -> None:
        self.logs: list[UsageLog] = []

    async def put(self, log: UsageLog) -> bool:
        self.logs.append(log)
        return True

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass


@pytest.mark.asyncio
async def test_a_reported_zero_settles_at_zero_beside_an_unreported_call_at_the_rate(async_db: AsyncSession) -> None:
    """Both halves of a zero-hit search: a provider that reports 0 is billed 0, one that reports nothing the rate.

    The old backend never reports a cost, so the tally is filled here as an
    instance-backed search would fill it.
    """
    for model_key in ("exa:exa-search", "otari:web_search"):
        async_db.add(
            ModelPricing(
                model_key=model_key, input_price_per_million=_SEARCH_RATE_PER_MILLION, output_price_per_million=0.0
            )
        )
    await async_db.commit()
    tally = ToolUsageTally()
    no_hits = "No results for query: 'otari'"
    tally.record_result(
        "web_search", no_hits, label="exa:exa-search", rate_keys=("exa:exa-search", "otari:web_search"), cost=Decimal(0)
    )
    tally.record_result("web_search", no_hits)
    writer = _Writer()

    await log_usage(
        db=async_db,
        log_writer=writer,
        api_key_id=None,
        model=MODEL_NAME,
        provider=None,
        endpoint="/v1/chat/completions",
        usage_override=CompletionUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        tool_tally=tally,
    )

    [row] = writer.logs
    assert row.pricing_breakdown == [
        {
            "meter": "web_search_calls",
            "label": "exa:exa-search",
            "units": 1,
            "unit_rate": 0.0,
            "cost": 0.0,
            "deployment_paid": True,
        },
        {
            "meter": "web_search_calls",
            "label": "otari:web_search",
            "units": 1,
            "unit_rate": _SEARCH_UNIT_COST,
            "cost": _SEARCH_UNIT_COST,
            # Recorded with no payer, so read off its label.
            "deployment_paid": False,
        },
    ]
    meter = (row.billing_meters or {})[TOOL_METER_NAMESPACE]["web_search"]
    assert meter == {"billed": 2, "errors": 0, "unit_rate": pytest.approx(_SEARCH_UNIT_COST / 2)}
    assert row.cost == Decimal("0.01")


@pytest.mark.asyncio
async def test_a_pricing_lookup_that_times_out_keeps_the_counts_on_the_row(async_db: AsyncSession) -> None:
    """A connect timeout raises a bare ``TimeoutError``; settlement still writes the row with its tool counts."""
    tally = ToolUsageTally()
    tally.record_result("web_search", "results")
    writer = _Writer()

    with patch("gateway.api.routes._pipeline.price_tool_calls", new=AsyncMock(side_effect=TimeoutError)):
        await log_usage(
            db=async_db,
            log_writer=writer,
            api_key_id=None,
            model=MODEL_NAME,
            provider=None,
            endpoint="/v1/chat/completions",
            usage_override=CompletionUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
            tool_tally=tally,
        )

    [row] = writer.logs
    assert (row.billing_meters or {})[TOOL_METER_NAMESPACE] == {"web_search": {"billed": 1, "errors": 0}}
    assert not any(entry["meter"] == "web_search_calls" for entry in (row.pricing_breakdown or []))


@pytest.mark.asyncio
async def test_fetch_call_is_metered_priced_and_spent(
    client: TestClient,
    api_key_header: dict[str, str],
    api_key_obj: dict[str, Any],
    model_pricing: dict[str, Any],
    fetch_pricing: dict[str, Any],
    db_session_factory: Callable[[], Session],
    monkeypatch: pytest.MonkeyPatch,
    test_config: GatewayConfig,
) -> None:
    _serve_fetch_on_fake(monkeypatch, test_config)
    user_id = api_key_obj["user_id"]
    before = _spend(db_session_factory, user_id)
    responses = [
        _completion(tool_call=True, tool_name="web_fetch", arguments='{"url":"https://example.com"}'),
        _completion(tool_call=False),
    ]

    with patch("gateway.services.mcp_loop.acompletion", new=AsyncMock(side_effect=responses)):
        response = client.post(
            f"{API_ROOT}/chat/completions",
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": "read it"}],
                "tools": [{"type": "otari_web_fetch"}],
            },
            headers=api_key_header,
        )

    assert response.status_code == 200, response.text
    row = _latest_row(db_session_factory, user_id)
    tools = (row.billing_meters or {})[TOOL_METER_NAMESPACE]
    assert tools["web_fetch"]["billed"] == 1
    assert tools["web_fetch"]["errors"] == 0
    line = next(entry for entry in (row.pricing_breakdown or []) if entry["meter"] == "web_fetch_calls")
    # Labeled by the instance, which has no row of its own, so the tool's row prices it.
    assert line["label"] == f"fake:{_FAKE_FETCH}"
    assert line["units"] == 1
    assert line["cost"] == pytest.approx(_SEARCH_UNIT_COST)
    after = _spend(db_session_factory, user_id)
    assert after - before == pytest.approx(float(row.cost or 0), rel=1e-6)


def _fetch_once(client: TestClient, headers: dict[str, str]) -> None:
    """One chat request whose model fetches one page, then answers."""
    responses = [
        _completion(tool_call=True, tool_name="web_fetch", arguments='{"url":"https://example.com"}'),
        _completion(tool_call=False),
    ]
    with patch("gateway.services.mcp_loop.acompletion", new=AsyncMock(side_effect=responses)):
        response = client.post(
            f"{API_ROOT}/chat/completions",
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": "read it"}],
                "tools": [{"type": "otari_web_fetch"}],
            },
            headers=headers,
        )
    assert response.status_code == 200, response.text


def _fetch_line(db_session_factory: Callable[[], Session], user_id: str) -> dict[str, Any]:
    row = _latest_row(db_session_factory, user_id)
    [line] = [entry for entry in (row.pricing_breakdown or []) if entry["meter"] == "web_fetch_calls"]
    return dict(line)


@pytest.mark.asyncio
async def test_a_fetch_on_an_instance_is_priced_at_the_instances_own_row(
    client: TestClient,
    api_key_header: dict[str, str],
    api_key_obj: dict[str, Any],
    master_key_header: dict[str, str],
    model_pricing: dict[str, Any],
    fetch_pricing: dict[str, Any],
    db_session_factory: Callable[[], Session],
    monkeypatch: pytest.MonkeyPatch,
    test_config: GatewayConfig,
) -> None:
    """The instance's fetch row wins over its old ``fake:<instance>`` row and ``otari:web_fetch`` (design L)."""
    _serve_fetch_on_fake(monkeypatch, test_config)
    for body in (
        {"model_key": f"fake:{_FAKE_FETCH}", "input_price_per_million": 4_000.0},
        {"kind": "fetch", "model_key": _FAKE_FETCH, "input_price_per_million": 2_500.0},
    ):
        response = client.post(
            f"{API_ROOT}/pricing", json={**body, "output_price_per_million": 0.0}, headers=master_key_header
        )
        assert response.status_code == 200, response.text

    _fetch_once(client, api_key_header)

    line = _fetch_line(db_session_factory, api_key_obj["user_id"])
    # The label is unchanged: it names what served the call, which the breakdown's readers key on.
    assert (line["label"], line["units"]) == (f"fake:{_FAKE_FETCH}", 1)
    assert line["cost"] == pytest.approx(0.0025)


@pytest.mark.asyncio
async def test_a_fetch_whose_provider_reports_its_cost_is_billed_that_cost(
    client: TestClient,
    api_key_header: dict[str, str],
    api_key_obj: dict[str, Any],
    model_pricing: dict[str, Any],
    fetch_pricing: dict[str, Any],
    db_session_factory: Callable[[], Session],
    monkeypatch: pytest.MonkeyPatch,
    test_config: GatewayConfig,
) -> None:
    _serve_fetch_on_fake(monkeypatch, test_config, cost=0.002)

    _fetch_once(client, api_key_header)

    line = _fetch_line(db_session_factory, api_key_obj["user_id"])
    assert (line["label"], line["units"]) == (f"fake:{_FAKE_FETCH}", 1)
    assert line["cost"] == pytest.approx(0.002), "the reported cost, not the otari:web_fetch rate"


@pytest.mark.asyncio
async def test_a_built_in_fetch_is_labeled_builtin_fetch_and_priced_under_otari_web_fetch(
    client: TestClient,
    api_key_header: dict[str, str],
    api_key_obj: dict[str, Any],
    model_pricing: dict[str, Any],
    fetch_pricing: dict[str, Any],
    db_session_factory: Callable[[], Session],
    monkeypatch: pytest.MonkeyPatch,
    test_config: GatewayConfig,
) -> None:
    monkeypatch.setattr(test_config, "web_fetch_enabled", True)
    url = canonicalize_web_url("https://example.com")
    page = WebFetchResult(
        text="page", content_type="text/plain", content_kind="text", requested_url=url, final_url=url, redirect_count=0
    )

    with patch.object(WebFetchService, "fetch", new=AsyncMock(return_value=page)):
        _fetch_once(client, api_key_header)

    line = _fetch_line(db_session_factory, api_key_obj["user_id"])
    assert (line["label"], line["units"]) == ("builtin:builtin_fetch", 1)
    assert line["cost"] == pytest.approx(_SEARCH_UNIT_COST)


@pytest.fixture
def configured_fetch_price_client(test_config: GatewayConfig, clean_database: None) -> Iterator[TestClient]:
    """A gateway whose configuration file serves fetch on a ``fake`` instance and prices it per call.

    Built rather than copied from ``test_config``: a copy keeps the shared config's
    baseline of fetch instances, which the startup overlay restores.
    """
    config = GatewayConfig(
        database_url=test_config.database_url,
        master_key=test_config.master_key,
        auto_migrate=False,
        require_pricing=False,
        providers=declared_test_providers(),
        web_fetch_enabled=True,
        fetch_tools={_FAKE_FETCH: {"provider": "fake", "price_per_call": 0.003}},
        web_fetch_default_tool=_FAKE_FETCH,
        pricing={MODEL_NAME: PricingConfig(input_price_per_million=1.0, output_price_per_million=1.0)},
    )
    yield from build_test_client(config)


@pytest.mark.asyncio
async def test_a_configuration_file_price_prices_a_fetch_instance(
    configured_fetch_price_client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
) -> None:
    client = configured_fetch_price_client
    key = client.post(f"{API_ROOT}/keys", json={"key_name": "fetch-price"}, headers=master_key_header).json()

    # The instance serves the fetch, so the built-in fetcher, which would reach the network, is never called.
    with patch.object(WebFetchService, "fetch", new=AsyncMock(side_effect=AssertionError("builtin_fetch ran"))):
        _fetch_once(client, {API_KEY_HEADER: f"Bearer {key['key']}"})

    line = _fetch_line(db_session_factory, key["user_id"])
    assert (line["label"], line["units"]) == (f"fake:{_FAKE_FETCH}", 1)
    assert line["cost"] == pytest.approx(0.003)


@pytest.mark.asyncio
async def test_failed_fetch_is_counted_but_not_billed(
    client: TestClient,
    api_key_header: dict[str, str],
    api_key_obj: dict[str, Any],
    model_pricing: dict[str, Any],
    fetch_pricing: dict[str, Any],
    db_session_factory: Callable[[], Session],
    monkeypatch: pytest.MonkeyPatch,
    test_config: GatewayConfig,
) -> None:
    _serve_fetch_on_fake(monkeypatch, test_config, error="rate_limit", error_status=429)
    responses = [
        _completion(tool_call=True, tool_name="web_fetch", arguments='{"url":"https://example.com"}'),
        _completion(tool_call=False),
    ]
    with patch("gateway.services.mcp_loop.acompletion", new=AsyncMock(side_effect=responses)):
        response = client.post(
            f"{API_ROOT}/chat/completions",
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": "read it"}],
                "tools": [{"type": "otari_web_fetch"}],
            },
            headers=api_key_header,
        )

    assert response.status_code == 200, response.text
    row = _latest_row(db_session_factory, api_key_obj["user_id"])
    tools = (row.billing_meters or {})[TOOL_METER_NAMESPACE]
    assert tools["web_fetch"] == {"billed": 0, "errors": 1}
    assert not any(entry["meter"] == "web_fetch_calls" for entry in (row.pricing_breakdown or []))


@pytest.mark.asyncio
async def test_fetch_batch_stops_at_the_shared_ten_call_limit(
    client: TestClient,
    api_key_header: dict[str, str],
    api_key_obj: dict[str, Any],
    db_session_factory: Callable[[], Session],
    monkeypatch: pytest.MonkeyPatch,
    test_config: GatewayConfig,
) -> None:
    _serve_fetch_on_fake(monkeypatch, test_config)
    provider = AsyncMock(return_value=_fetch_batch_completion(11))
    with patch("gateway.services.mcp_loop.acompletion", new=provider):
        response = client.post(
            f"{API_ROOT}/chat/completions",
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": "read these"}],
                "tools": [{"type": "otari_web_fetch"}],
            },
            headers=api_key_header,
        )

    assert response.status_code == 422, response.text
    assert "10 calls per request" in response.json()["detail"]
    row = _latest_row(db_session_factory, api_key_obj["user_id"])
    fetches = (row.billing_meters or {})[TOOL_METER_NAMESPACE]["web_fetch"]
    assert (fetches["billed"], fetches["errors"]) == (10, 0), "the eleventh call never reached the fetch instance"


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
        f"{API_ROOT}/users", json={"user_id": "tool-user"}, headers={API_KEY_HEADER: "Bearer test-master-key"}
    )
    # Price the model so the model gate passes and the tool gate is what fires.
    strict_pricing_client.post(
        f"{API_ROOT}/pricing",
        json={"model_key": "openai:gpt-4o", "input_price_per_million": 1.0, "output_price_per_million": 1.0},
        headers={API_KEY_HEADER: "Bearer test-master-key"},
    )

    with (
        patch("gateway.services.mcp_loop.acompletion", new=AsyncMock()) as provider,
        patch.dict("os.environ", {"OTARI_WEB_SEARCH_URL": "http://web-search.invalid"}),
    ):
        response = strict_pricing_client.post(
            f"{API_ROOT}/chat/completions",
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


@pytest.mark.asyncio
async def test_unpriced_fetch_is_refused_before_the_provider_call(
    strict_pricing_client: TestClient,
) -> None:
    headers = {API_KEY_HEADER: "Bearer test-master-key"}
    strict_pricing_client.post(f"{API_ROOT}/users", json={"user_id": "fetch-user"}, headers=headers)
    strict_pricing_client.post(
        f"{API_ROOT}/pricing",
        json={"model_key": "openai:gpt-4o", "input_price_per_million": 1.0, "output_price_per_million": 1.0},
        headers=headers,
    )

    with patch("gateway.services.mcp_loop.acompletion", new=AsyncMock()) as provider:
        response = strict_pricing_client.post(
            f"{API_ROOT}/chat/completions",
            json={
                "model": "openai:gpt-4o",
                "messages": [{"role": "user", "content": "hi"}],
                "user": "fetch-user",
                "tools": [{"type": "otari_web_fetch"}],
            },
            headers=headers,
        )

    assert response.status_code == 402, response.text
    assert "otari:web_fetch" in response.json()["detail"]
    provider.assert_not_called()


def _searched_usage(searches: int) -> GatewayUsage:
    """What a Messages call reports when Anthropic ran its own web search ``searches`` times."""
    return GatewayUsage(
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        cache_tokens_in_prompt=False,
        provider_web_searches=searches,
    )


async def _settle_provider_searches(async_db: AsyncSession, searches: int) -> UsageLog:
    writer = _Writer()
    await log_usage(
        db=async_db,
        log_writer=writer,
        api_key_id=None,
        model="claude-haiku-4-5",
        provider="anthropic",
        provider_type="anthropic",
        endpoint="/v1/messages",
        usage_override=_searched_usage(searches),
    )
    [row] = writer.logs
    return row


def _search_line(row: UsageLog) -> dict[str, Any]:
    [line] = [entry for entry in row.pricing_breakdown or [] if entry.get("label") == "anthropic:web_search"]
    return line


@pytest.mark.asyncio
async def test_searches_the_provider_ran_are_billed_at_its_list_price(async_db: AsyncSession) -> None:
    """Anthropic bills its own web search per search; default pricing prices it at the published $10 per 1,000."""
    configure_default_pricing(True)

    row = await _settle_provider_searches(async_db, 2)

    assert _search_line(row) == {
        "meter": "web_search_calls",
        "label": "anthropic:web_search",
        "units": 2,
        "unit_rate": 0.01,
        "cost": 0.02,
        # The model call was not deployment-paid, so neither were its provider's searches.
        "deployment_paid": False,
    }
    assert (row.billing_meters or {})[TOOL_METER_NAMESPACE]["web_search"]["billed"] == 2
    assert row.cost is not None
    assert row.cost >= Decimal("0.02")


@pytest.mark.asyncio
async def test_an_operator_rate_wins_over_the_list_price(async_db: AsyncSession) -> None:
    configure_default_pricing(True)
    async_db.add(
        ModelPricing(model_key="anthropic:web_search", input_price_per_million=5_000.0, output_price_per_million=0.0)
    )
    await async_db.commit()

    row = await _settle_provider_searches(async_db, 3)

    assert _search_line(row)["unit_rate"] == pytest.approx(0.005)
    assert _search_line(row)["cost"] == pytest.approx(0.015)


@pytest.mark.asyncio
async def test_without_default_pricing_or_a_rate_the_searches_stay_on_the_row_unpriced(async_db: AsyncSession) -> None:
    row = await _settle_provider_searches(async_db, 1)

    assert _search_line(row)["cost"] == 0.0
    assert (row.billing_meters or {})[TOOL_METER_NAMESPACE]["web_search"]["billed"] == 1


@pytest.mark.asyncio
async def test_a_call_with_no_provider_search_adds_no_tool_line(async_db: AsyncSession) -> None:
    configure_default_pricing(True)

    row = await _settle_provider_searches(async_db, 0)

    assert TOOL_METER_NAMESPACE not in (row.billing_meters or {})
    assert not any(entry.get("label") == "anthropic:web_search" for entry in row.pricing_breakdown or [])
