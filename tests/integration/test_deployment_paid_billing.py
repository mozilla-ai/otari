"""A deployment-paid completion is gated on and charged to its organization's funds, end to end.

A request whose credential ``ModelProviderPort`` served costs the deployment its
upstream bill, so it is admitted only when ``BillingPort`` finds funds and its
settled cost is charged to the organization. A probe overlay binds both ports
here: the fleet serves ``openai`` and nothing else, and the billing probe keeps
a ledger of what landed. It writes on the request session as a port does, and
an entry reaches the ledger only when that session commits, so a charge that is
recorded but never committed shows up as missing. A request on the gateway's
own provider key never reaches the probe at all.
"""

import asyncio
import concurrent.futures
import itertools
import logging
import sys
from collections.abc import AsyncIterator, Callable, Generator, Iterator
from contextlib import contextmanager
from decimal import Decimal
from pathlib import Path
from types import ModuleType
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from any_llm.types.batch import Batch
from any_llm.types.completion import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    ChatCompletionMessageFunctionToolCall,
    Choice,
    ChoiceDelta,
    ChunkChoice,
    CompletionUsage,
    CreateEmbeddingResponse,
    Embedding,
    Function,
    Usage,
)
from fastapi import FastAPI, status
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from gateway.api.deps import get_web_provider_client
from gateway.api.routes import _pipeline as pipeline
from gateway.core.config import API_KEY_HEADER, API_ROOT, COST_USD_HEADER, PRICING_SOURCE_HEADER, GatewayConfig
from gateway.core.error_codes import ERROR_CODE_HEADER, INSUFFICIENT_FUNDS, RATE_LIMIT_RULE_HEADER, RATE_LIMITED
from gateway.log_config import logger as gateway_logger
from gateway.models.routing import RoutingConfig
from gateway.models.usage import UsageLog
from gateway.ports.billing_port import BilledModel, ToolUse
from gateway.ports.rate_limit_store_port import RateLimitStorePort
from gateway.services.secret_box import generate_secret_key

from .conftest import FAKE_SEARCH, build_test_client, price_fake_search

MODULE = "probe_deployment_paid"
PROBE_BOOTSTRAP = """
from decimal import Decimal

from gateway.container import Container
from gateway.ports.billing_port import BillingPort, InsufficientFundsError
from gateway.ports.model_provider_port import HostedCredential, ModelProviderPort

GATES = []  # Every gate call, landed or not.
BILLED = []  # The model each gate and charge was told it pays for.
TOOLS = []  # The gateway-run tool calls each charge was told it pays for.
GATE_TOOLS = []  # The gateway-run tools each gate was told a call may run.
LEDGER = []  # What landed: a session's entries move here when it commits.
PENDING = {}  # Entries written on a session, by its id, until it commits, rolls back or closes.
SESSIONS = []
FUNDS = {"available": Decimal("10")}
FAULTS = set()
LIMIT = {"concurrency": None}


class Fleet:
    def __init__(self, session):
        del session

    async def resolve_hosted_credential(self, *, organization_id, workspace_id, provider, model):
        if provider != "openai":
            return None
        return HostedCredential(api_key="fleet-key", api_base=None, response_provider="openai")

    async def get_hosted_models(self, *, organization_id):
        return {"openai": None}


class ProbeBilling:
    def __init__(self, session):
        self.session = session
        SESSIONS.append(session)

    async def apply_due_credit(self, *, organization_id):
        GATES.append(("apply_due_credit", str(organization_id)))

    async def require_funds_on_deposit(self, *, organization_id, billed, tools, actor_user_id):
        GATES.append(("require_funds_on_deposit", str(organization_id)))
        BILLED.append(("gate", billed))
        GATE_TOOLS.append(tools)
        if "funds" in FAULTS:
            raise RuntimeError("billing down")
        if FUNDS["available"] <= 0:
            raise InsufficientFundsError(FUNDS["available"])

    async def has_funds_on_deposit(self, *, organization_id, billed, tools, actor_user_id):
        GATES.append(("has_funds_on_deposit", str(organization_id)))
        GATE_TOOLS.append(tools)
        return FUNDS["available"] > 0

    async def concurrency_limit(self, *, organization_id, billed, actor_user_id):
        return LIMIT["concurrency"]

    async def require_unheld_funds(self, *, organization_id):
        raise AssertionError("no holds on the completion path")

    async def hold(self, *, organization_id, amount):
        raise AssertionError("no holds on the completion path")

    async def release_hold(self, *, organization_id, amount):
        raise AssertionError("no holds on the completion path")

    async def charge(
        self, *, organization_id, amount, billed, tools, description=None, actor_user_id=None, api_key_id=None
    ):
        if "charge" in FAULTS:
            raise RuntimeError("ledger down")
        BILLED.append(("charge", billed))
        TOOLS.append(dict(tools))
        entry = ("charge", str(organization_id), amount, description, api_key_id)
        PENDING.setdefault(id(self.session), []).append(entry)


def register(container: Container) -> None:
    container.bind(ModelProviderPort, Fleet)
    container.bind(BillingPort, ProbeBilling)
"""
MASTER = {API_KEY_HEADER: "Bearer test-master-key"}
# $1,000 per million tokens either way, so the 15 tokens below cost $0.015.
USAGE = CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
COST = Decimal("0.015")

_NAMES = (f"{MODULE}_{n}" for n in itertools.count())


@pytest.fixture(autouse=True)
def _no_ambient_provider_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """A developer's own shell key would credential the ladder and skip the fleet."""
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(name, raising=False)


@contextmanager
def _session_ledger(probe: ModuleType) -> Iterator[None]:
    """Land a probe session's pending entries when it commits, and drop them when it rolls back or closes."""
    commit, rollback, close = AsyncSession.commit, AsyncSession.rollback, AsyncSession.close

    async def committed(self: AsyncSession) -> None:
        await commit(self)
        if any(self is session for session in probe.SESSIONS):
            probe.LEDGER.extend(probe.PENDING.pop(id(self), []))

    async def rolled_back(self: AsyncSession) -> None:
        await rollback(self)
        probe.PENDING.pop(id(self), None)

    async def closed(self: AsyncSession) -> None:
        await close(self)
        probe.PENDING.pop(id(self), None)

    with (
        patch.object(AsyncSession, "commit", committed),
        patch.object(AsyncSession, "rollback", rolled_back),
        patch.object(AsyncSession, "close", closed),
    ):
        yield


@pytest.fixture
def billed(
    postgres_url: str, clean_database: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Generator[tuple[TestClient, ModuleType]]:
    yield from _billed(postgres_url, tmp_path, monkeypatch)


@pytest.fixture
def billed_search(
    postgres_url: str, clean_database: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Generator[tuple[TestClient, ModuleType]]:
    """``billed``, with a search tool on the deployment's account that reports what each search cost it."""
    options = {"hits": [{"url": "https://otari.example"}], "cost": 0.007}
    tools = {"fake-search": {"provider": "fake", "options": options}}
    yield from _billed(postgres_url, tmp_path, monkeypatch, search_tools=tools)


@pytest.fixture
def billed_in_loop_search(
    postgres_url: str, clean_database: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Generator[tuple[TestClient, ModuleType]]:
    """``billed``, with the in-loop search on a search tool on the deployment's account that its own row prices."""
    tools = {FAKE_SEARCH: {"provider": "fake", "options": {"hits": [{"url": "https://otari.example"}]}}}
    yield from _billed(postgres_url, tmp_path, monkeypatch, search_tools=tools, web_search_extract=False)


@pytest.fixture
def billed_legacy_search(
    postgres_url: str, clean_database: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Generator[tuple[TestClient, ModuleType]]:
    """``billed``, searching only through the legacy settings' Tavily key, which the deployment pays for."""
    monkeypatch.delenv("OTARI_WEB_SEARCH_URL", raising=False)
    yield from _billed(
        postgres_url, tmp_path, monkeypatch, web_search_provider="tavily", web_search_provider_api_key="tvly-ours"
    )


@pytest.fixture
def billed_sandbox(
    postgres_url: str, clean_database: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Generator[tuple[TestClient, ModuleType]]:
    """``billed``, with a sandbox on the deployment's account that no test reaches."""
    yield from _billed(postgres_url, tmp_path, monkeypatch, sandbox_url="http://sandbox.invalid")


def _billed(
    postgres_url: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **settings: Any
) -> Generator[tuple[TestClient, ModuleType]]:
    name = next(_NAMES)
    (tmp_path / f"{name}.py").write_text(PROBE_BOOTSTRAP)
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop(name, None)
    config = GatewayConfig(
        database_url=postgres_url,
        master_key="test-master-key",
        auto_migrate=False,
        require_pricing=False,
        model_discovery=False,
        # The gateway's own key for one provider, which the fleet must never displace.
        providers={"anthropic": {"api_key": "sk-ant"}},
        routing=RoutingConfig.model_validate(
            {
                "policies": {
                    # A fleet candidate first, then the gateway's own key.
                    "fleet_first": {
                        "select": [{"default": "openai:gpt-4o"}],
                        "on_failure": ["anthropic:claude-opus-4"],
                    },
                    # The gateway's own key first, then a fleet candidate.
                    "fleet_second": {
                        "select": [{"default": "anthropic:claude-opus-4"}],
                        "on_failure": ["openai:gpt-4o"],
                    },
                }
            }
        ),
        bootstrap=f"{name}:register",
        **settings,
    )
    try:
        for client in build_test_client(config):
            probe = sys.modules[name]
            with _session_ledger(probe):
                yield client, probe
    finally:
        sys.modules.pop(name, None)


@pytest.fixture
def gateway_logs(caplog: pytest.LogCaptureFixture) -> Iterator[pytest.LogCaptureFixture]:
    """The gateway logger does not propagate, so caplog sees it only with its handler attached."""
    gateway_logger.addHandler(caplog.handler)
    caplog.set_level(logging.WARNING, logger="gateway")
    try:
        yield caplog
    finally:
        gateway_logger.removeHandler(caplog.handler)


def _price(client: TestClient, model_key: str) -> None:
    response = client.post(
        f"{API_ROOT}/pricing",
        json={"model_key": model_key, "input_price_per_million": 1000.0, "output_price_per_million": 1000.0},
        headers=MASTER,
    )
    assert response.status_code == status.HTTP_200_OK, response.text


def _key(client: TestClient) -> dict[str, Any]:
    budget = client.post(f"{API_ROOT}/budgets", json={"max_budget": 100.0}, headers=MASTER)
    assert budget.status_code == status.HTTP_200_OK, budget.text
    user = client.post(
        f"{API_ROOT}/users", json={"user_id": "billed", "budget_id": budget.json()["budget_id"]}, headers=MASTER
    )
    assert user.status_code == status.HTTP_200_OK, user.text
    created = client.post(f"{API_ROOT}/keys", json={"key_name": "billed", "user_id": "billed"}, headers=MASTER)
    assert created.status_code == status.HTTP_200_OK, created.text
    key: dict[str, Any] = created.json()
    return key


def _auth(key: dict[str, Any]) -> dict[str, str]:
    return {API_KEY_HEADER: f"Bearer {key['key']}"}


def _user(client: TestClient) -> dict[str, Any]:
    user: dict[str, Any] = client.get(f"{API_ROOT}/users/billed", headers=MASTER).json()
    return user


def _deployment_paid_flags(make_session: Callable[[], Session]) -> list[bool]:
    with make_session() as db:
        return list(db.scalars(select(UsageLog.deployment_paid).where(UsageLog.user_id == "billed")))


def _completion(model: str) -> ChatCompletion:
    return ChatCompletion(
        id="chatcmpl-billed",
        object="chat.completion",
        created=0,
        model=model,
        choices=[Choice(index=0, message=ChatCompletionMessage(role="assistant", content="hi"), finish_reason="stop")],
        usage=USAGE,
    )


def _provider(model: str, *, stream: bool) -> AsyncMock:
    async def acompletion(**kwargs: Any) -> Any:
        if not stream:
            return _completion(model)

        async def chunks() -> AsyncIterator[ChatCompletionChunk]:
            yield ChatCompletionChunk(
                id="chunk-1",
                object="chat.completion.chunk",
                created=0,
                model=model,
                choices=[ChunkChoice(index=0, delta=ChoiceDelta(role="assistant", content="hi"), finish_reason="stop")],
            )
            yield ChatCompletionChunk(
                id="chunk-1", object="chat.completion.chunk", created=0, model=model, choices=[], usage=USAGE
            )

        return chunks()

    return AsyncMock(side_effect=acompletion)


def _chat(client: TestClient, key: dict[str, Any], model: str, *, stream: bool = False) -> tuple[Any, AsyncMock]:
    provider = _provider(model, stream=stream)
    with patch("gateway.api.routes.chat.acompletion", new=provider):
        response = client.post(
            f"{API_ROOT}/chat/completions",
            json={"model": model, "messages": [{"role": "user", "content": "Hi"}], "stream": stream},
            headers=_auth(key),
        )
        # A streamed body settles while it is read, inside the patch.
        response.read()
    return response, provider


def test_a_deployment_paid_completion_is_gated_then_charged_its_settled_cost(
    billed: tuple[TestClient, ModuleType], db_session_factory: Callable[[], Session]
) -> None:
    client, probe = billed
    _price(client, "openai:gpt-4o")
    key = _key(client)

    response, provider = _chat(client, key, "openai:gpt-4o")

    assert response.status_code == status.HTTP_200_OK, response.text
    assert provider.call_args.kwargs["api_key"] == "fleet-key"
    (organization,) = {org for _, org in probe.GATES}
    assert [gate for gate, _ in probe.GATES] == ["apply_due_credit", "require_funds_on_deposit"]
    # Landed with the settlement's commit, once, on the organization that was gated.
    assert probe.LEDGER == [("charge", organization, COST, "openai:gpt-4o", key["id"])]
    served = BilledModel(provider="openai", model="gpt-4o", open_weights=False)
    assert probe.BILLED == [("gate", served), ("charge", served)]
    user = _user(client)
    assert user["spend"] == pytest.approx(float(COST))
    assert user["reserved"] == pytest.approx(0.0)
    assert _deployment_paid_flags(db_session_factory) == [True]


def test_a_completion_over_its_organizations_concurrency_limit_waits_then_is_refused(
    billed: tuple[TestClient, ModuleType], monkeypatch: pytest.MonkeyPatch
) -> None:
    client, probe = billed
    _price(client, "openai:gpt-4o")
    key = _key(client)
    probe.LIMIT["concurrency"] = 1
    monkeypatch.setattr(pipeline, "_ORGANIZATION_SLOT_WAIT_SEC", 0.05)

    served, _ = _chat(client, key, "openai:gpt-4o")
    assert served.status_code == status.HTTP_200_OK, served.text
    (organization,) = {org for _, org in probe.GATES}
    store: RateLimitStorePort = cast(FastAPI, client.app).state.rate_limit_store
    slot = f"organization:{organization}:concurrent"
    # The served request gave its slot back when its response ended, so another request can take it.
    other = asyncio.run(store.acquire(slot, 1, 60.0))
    assert other is not None

    refused, provider = _chat(client, key, "openai:gpt-4o")

    assert refused.status_code == status.HTTP_429_TOO_MANY_REQUESTS, refused.text
    assert refused.headers[ERROR_CODE_HEADER] == RATE_LIMITED
    assert refused.headers["Retry-After"] == "1"
    assert refused.headers[RATE_LIMIT_RULE_HEADER] == "organization_concurrency"
    provider.assert_not_awaited()
    assert len(probe.LEDGER) == 1
    assert _user(client)["reserved"] == pytest.approx(0.0)
    asyncio.run(store.release(slot, other))


def test_a_completion_canceled_while_waiting_for_a_slot_gives_its_reservation_back(
    billed: tuple[TestClient, ModuleType], monkeypatch: pytest.MonkeyPatch
) -> None:
    client, _probe = billed
    _price(client, "openai:gpt-4o")
    key = _key(client)

    async def canceled(ctx: Any, charge: Any) -> None:
        raise asyncio.CancelledError

    monkeypatch.setattr(pipeline, "_hold_organization_slot", canceled)

    with pytest.raises((asyncio.CancelledError, concurrent.futures.CancelledError)):
        _chat(client, key, "openai:gpt-4o")

    assert _user(client)["reserved"] == pytest.approx(0.0)


def test_a_failover_canceled_while_waiting_for_a_slot_gives_its_reservation_back(
    billed: tuple[TestClient, ModuleType], monkeypatch: pytest.MonkeyPatch
) -> None:
    client, _probe = billed
    _price(client, "openai:gpt-4o")
    _price(client, "anthropic:claude-opus-4")
    key = _key(client)

    waits: list[Any] = []
    held: list[Decimal] = []
    hold = pipeline._hold_organization_slot

    async def canceled(ctx: Any, charge: Any) -> None:
        # The head (the gateway's own key) is admitted with nothing to charge; only the fleet fallback waits.
        if charge is None:
            await hold(ctx, charge)
            return
        waits.append(charge)
        held.append(ctx.reservation.estimate if ctx.reservation is not None else Decimal(0))
        raise asyncio.CancelledError

    monkeypatch.setattr(pipeline, "_hold_organization_slot", canceled)

    tried: list[str] = []

    async def acompletion(**kwargs: Any) -> Any:
        tried.append(kwargs["model"])
        request = httpx.Request("POST", "http://upstream")
        raise httpx.HTTPStatusError("down", request=request, response=httpx.Response(503, request=request))

    with (
        patch("gateway.api.routes.chat.acompletion", new=AsyncMock(side_effect=acompletion)),
        # The test client hands the server's cancellation back as concurrent.futures' own.
        pytest.raises((asyncio.CancelledError, concurrent.futures.CancelledError)),
    ):
        client.post(
            f"{API_ROOT}/chat/completions",
            json={"model": "fleet_second", "messages": [{"role": "user", "content": "Hi"}]},
            headers=_auth(key),
        )
    # The head was tried and failed over: the fallback's wait is what was canceled.

    assert len(waits) == 1
    assert tried == ["anthropic:claude-opus-4"]
    assert held[0] > 0
    assert _user(client)["reserved"] == pytest.approx(0.0)


def test_a_deployment_paid_stream_is_charged_once_after_it_ends(
    billed: tuple[TestClient, ModuleType], db_session_factory: Callable[[], Session]
) -> None:
    client, probe = billed
    _price(client, "openai:gpt-4o")
    key = _key(client)

    response, _ = _chat(client, key, "openai:gpt-4o", stream=True)

    assert response.status_code == status.HTTP_200_OK, response.text
    assert "[DONE]" in response.text
    assert [(entry[0], entry[2], entry[3]) for entry in probe.LEDGER] == [("charge", COST, "openai:gpt-4o")]
    assert _user(client)["reserved"] == pytest.approx(0.0)
    assert _deployment_paid_flags(db_session_factory) == [True]


@pytest.mark.parametrize(
    ("path", "body", "error_body"),
    [
        (
            "/chat/completions",
            {"messages": [{"role": "user", "content": "Hi"}]},
            lambda detail: {"detail": detail, "code": INSUFFICIENT_FUNDS},
        ),
        (
            "/messages",
            {"max_tokens": 16, "messages": [{"role": "user", "content": "Hi"}]},
            lambda detail: {
                "type": "error",
                "error": {"type": "invalid_request_error", "message": detail},
                "code": INSUFFICIENT_FUNDS,
            },
        ),
        ("/responses", {"input": "Hi"}, lambda detail: {"detail": detail, "code": INSUFFICIENT_FUNDS}),
    ],
    ids=["chat", "messages", "responses"],
)
def test_an_organization_without_funds_is_refused_before_dispatch(
    billed: tuple[TestClient, ModuleType],
    db_session_factory: Callable[[], Session],
    path: str,
    body: dict[str, Any],
    error_body: Any,
) -> None:
    client, probe = billed
    _price(client, "openai:gpt-4o")
    key = _key(client)
    probe.FUNDS["available"] = Decimal("0")

    with (
        patch("gateway.api.routes.chat.acompletion", new=AsyncMock()) as chat_call,
        patch("gateway.api.routes.responses.aresponses", new=AsyncMock()) as responses_call,
    ):
        response = client.post(f"{API_ROOT}{path}", json={"model": "openai:gpt-4o", **body}, headers=_auth(key))

    assert response.status_code == status.HTTP_402_PAYMENT_REQUIRED, response.text
    assert response.headers[ERROR_CODE_HEADER] == INSUFFICIENT_FUNDS
    detail = "Insufficient funds ($0.00) to serve this request."
    payload = response.json()
    expected = error_body(detail)
    assert {name: payload.get(name) for name in expected} == expected
    chat_call.assert_not_called()
    responses_call.assert_not_called()
    assert probe.LEDGER == []
    # The preamble's hold is given back, and the refusal is on the activity log.
    assert _user(client)["reserved"] == pytest.approx(0.0)
    rows = client.get(f"{API_ROOT}/usage", params={"user_id": "billed"}, headers=MASTER).json()
    assert [(row["status"], row["status_code"], row["provider"]) for row in rows] == [("error", 402, "openai")]
    # Refused before it was served, so nothing the deployment paid for.
    assert _deployment_paid_flags(db_session_factory) == [False]


def test_a_request_on_the_gateways_own_key_never_reaches_billing(
    billed: tuple[TestClient, ModuleType], db_session_factory: Callable[[], Session]
) -> None:
    client, probe = billed
    _price(client, "anthropic:claude-opus-4")
    key = _key(client)

    response, provider = _chat(client, key, "anthropic:claude-opus-4")

    assert response.status_code == status.HTTP_200_OK, response.text
    assert provider.call_args.kwargs["api_key"] == "sk-ant"
    assert probe.GATES == []
    assert probe.LEDGER == []
    assert _user(client)["spend"] == pytest.approx(float(COST))
    assert _deployment_paid_flags(db_session_factory) == [False]


def test_an_unpriced_deployment_paid_completion_is_not_charged_and_says_so(
    billed: tuple[TestClient, ModuleType], gateway_logs: pytest.LogCaptureFixture
) -> None:
    client, probe = billed
    key = _key(client)

    response, _ = _chat(client, key, "openai:gpt-4o-unpriced")

    assert response.status_code == status.HTTP_200_OK, response.text
    assert len(probe.GATES) == 2
    assert probe.LEDGER == []
    warnings = [r.getMessage() for r in gateway_logs.records if r.levelno == logging.WARNING]
    assert any("gpt-4o-unpriced" in message and "not charged" in message for message in warnings), warnings


def test_a_failed_charge_still_serves_and_settles_the_budget(
    billed: tuple[TestClient, ModuleType], gateway_logs: pytest.LogCaptureFixture
) -> None:
    client, probe = billed
    _price(client, "openai:gpt-4o")
    key = _key(client)
    probe.FAULTS.add("charge")

    response, _ = _chat(client, key, "openai:gpt-4o")

    assert response.status_code == status.HTTP_200_OK, response.text
    assert probe.LEDGER == []
    user = _user(client)
    assert user["spend"] == pytest.approx(float(COST))
    assert user["reserved"] == pytest.approx(0.0)
    errors = [r.getMessage() for r in gateway_logs.records if r.levelno == logging.ERROR]
    assert any("deployment-paid" in message and "request_id=" in message for message in errors), errors


def _routed(client: TestClient, key: dict[str, Any], policy: str, *, failing: str) -> tuple[Any, AsyncMock]:
    async def acompletion(**kwargs: Any) -> Any:
        if kwargs["model"] == failing:
            request = httpx.Request("POST", "http://upstream")
            raise httpx.HTTPStatusError("down", request=request, response=httpx.Response(503, request=request))
        return _completion(kwargs["model"].split(":", 1)[1])

    provider = AsyncMock(side_effect=acompletion)
    with patch("gateway.api.routes.chat.acompletion", new=provider):
        response = client.post(
            f"{API_ROOT}/chat/completions",
            json={"model": policy, "messages": [{"role": "user", "content": "Hi"}]},
            headers=_auth(key),
        )
    return response, provider


def test_a_policy_falls_over_onto_a_deployment_paid_candidate_and_charges_it(
    billed: tuple[TestClient, ModuleType], db_session_factory: Callable[[], Session]
) -> None:
    client, probe = billed
    _price(client, "openai:gpt-4o")
    _price(client, "anthropic:claude-opus-4")
    key = _key(client)

    response, provider = _routed(client, key, "fleet_second", failing="anthropic:claude-opus-4")

    assert response.status_code == status.HTTP_200_OK, response.text
    assert [call.kwargs["api_key"] for call in provider.call_args_list] == ["sk-ant", "fleet-key"]
    assert [(entry[0], entry[2], entry[3]) for entry in probe.LEDGER] == [("charge", COST, "openai:gpt-4o")]
    assert _user(client)["reserved"] == pytest.approx(0.0)
    with db_session_factory() as db:
        served = db.scalars(select(UsageLog.deployment_paid).where(UsageLog.status == "success")).all()
    assert served == [True]


def test_a_policy_skips_a_deployment_paid_candidate_its_organization_cannot_fund(
    billed: tuple[TestClient, ModuleType], db_session_factory: Callable[[], Session]
) -> None:
    client, probe = billed
    _price(client, "openai:gpt-4o")
    _price(client, "anthropic:claude-opus-4")
    key = _key(client)
    probe.FUNDS["available"] = Decimal("0")

    response, provider = _routed(client, key, "fleet_first", failing="none")

    assert response.status_code == status.HTTP_200_OK, response.text
    assert [call.kwargs["api_key"] for call in provider.call_args_list] == ["sk-ant"]
    assert probe.LEDGER == []
    with db_session_factory() as db:
        served = db.scalars(select(UsageLog.deployment_paid).where(UsageLog.status == "success")).all()
    assert served == [False]


def test_a_policy_skips_a_deployment_paid_candidate_with_no_free_slot(
    billed: tuple[TestClient, ModuleType], monkeypatch: pytest.MonkeyPatch
) -> None:
    client, probe = billed
    _price(client, "openai:gpt-4o")
    _price(client, "anthropic:claude-opus-4")
    key = _key(client)
    # One served request names the organization whose slot is then taken.
    assert _chat(client, key, "openai:gpt-4o")[0].status_code == status.HTTP_200_OK
    (organization,) = {org for _, org in probe.GATES}
    probe.LIMIT["concurrency"] = 1
    monkeypatch.setattr(pipeline, "_ORGANIZATION_SLOT_WAIT_SEC", 0.05)
    store: RateLimitStorePort = cast(FastAPI, client.app).state.rate_limit_store
    slot = f"organization:{organization}:concurrent"
    taken = asyncio.run(store.acquire(slot, 1, 60.0))
    assert taken is not None

    response, provider = _routed(client, key, "fleet_first", failing="none")

    assert response.status_code == status.HTTP_200_OK, response.text
    assert [call.kwargs["api_key"] for call in provider.call_args_list] == ["sk-ant"]
    assert len(probe.LEDGER) == 1
    asyncio.run(store.release(slot, taken))


def test_embeddings_on_a_deployment_paid_credential_are_gated_then_charged(
    billed: tuple[TestClient, ModuleType], db_session_factory: Callable[[], Session]
) -> None:
    client, probe = billed
    _price(client, "openai:text-embedding-3-small")
    key = _key(client)
    embedded = CreateEmbeddingResponse(
        data=[Embedding(embedding=[0.1], index=0, object="embedding")],
        model="text-embedding-3-small",
        object="list",
        usage=Usage(prompt_tokens=15, total_tokens=15),
    )

    with patch("gateway.api.routes.embeddings.aembedding", new=AsyncMock(return_value=embedded)) as call:
        response = client.post(
            f"{API_ROOT}/embeddings", json={"model": "openai:text-embedding-3-small", "input": "hi"}, headers=_auth(key)
        )

    assert response.status_code == status.HTTP_200_OK, response.text
    assert call.await_args is not None
    assert call.await_args.kwargs["api_key"] == "fleet-key"
    assert [gate for gate, _ in probe.GATES] == ["apply_due_credit", "require_funds_on_deposit"]
    assert [(entry[0], entry[3]) for entry in probe.LEDGER] == [("charge", "openai:text-embedding-3-small")]
    assert _deployment_paid_flags(db_session_factory) == [True]

    probe.FUNDS["available"] = Decimal("0")
    with patch("gateway.api.routes.embeddings.aembedding", new=AsyncMock(return_value=embedded)) as refused_call:
        refused = client.post(
            f"{API_ROOT}/embeddings", json={"model": "openai:text-embedding-3-small", "input": "hi"}, headers=_auth(key)
        )

    assert refused.status_code == status.HTTP_402_PAYMENT_REQUIRED, refused.text
    assert refused.headers[ERROR_CODE_HEADER] == INSUFFICIENT_FUNDS
    refused_call.assert_not_called()
    assert _user(client)["reserved"] == pytest.approx(0.0)


class _SupportsBatch:
    SUPPORTS_BATCH = True


def test_a_batch_on_a_deployment_paid_credential_is_refused_without_funds(
    billed: tuple[TestClient, ModuleType],
) -> None:
    client, probe = billed
    key = _key(client)
    probe.FUNDS["available"] = Decimal("0")
    body = {
        "model": "openai:gpt-4o",
        "requests": [{"custom_id": "r1", "body": {"messages": [{"role": "user", "content": "Hi"}]}}],
    }

    with (
        patch("gateway.api.routes.batches.AnyLLM.get_provider_class", return_value=_SupportsBatch),
        patch("gateway.api.routes.batches.acreate_batch", new=AsyncMock(spec=Batch)) as create,
    ):
        response = client.post(f"{API_ROOT}/batches", json=body, headers=_auth(key))

    assert response.status_code == status.HTTP_402_PAYMENT_REQUIRED, response.text
    assert response.headers[ERROR_CODE_HEADER] == INSUFFICIENT_FUNDS
    create.assert_not_called()


# A gateway-run tool is charged by whose account ran it, whoever paid for the model call.
# $1,000 per million requests: one search costs $0.001.
SEARCH_COST = Decimal("0.001")


def _search_round(model: str) -> ChatCompletion:
    """A model turn that asks the gateway to search."""
    call = ChatCompletionMessageFunctionToolCall(
        id="call_1", type="function", function=Function(name="web_search", arguments='{"query": "otari"}')
    )
    message = ChatCompletionMessage(role="assistant", content=None, tool_calls=[call])
    return ChatCompletion(
        id="chatcmpl-search",
        object="chat.completion",
        created=0,
        model=model,
        choices=[Choice(index=0, message=message, finish_reason="tool_calls")],
        # A copy: the tool loop adds each round's usage onto the first round's in place.
        usage=USAGE.model_copy(),
    )


def _chat_with_search(client: TestClient, key: dict[str, Any], model: str) -> tuple[Any, AsyncMock]:
    """A completion whose model searches once on the deployment's search tool, then answers."""
    bare = model.split(":", 1)[1]
    answer = _completion(bare)
    answer.usage = USAGE.model_copy()
    provider = AsyncMock(side_effect=[_search_round(bare), answer])
    with patch("gateway.services.mcp_loop.acompletion", new=provider):
        response = client.post(
            f"{API_ROOT}/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": "what is otari"}],
                "tools": [{"type": "otari_web_search"}],
            },
            headers=_auth(key),
        )
    return response, provider


def test_a_search_on_the_deployments_account_is_charged_beside_a_model_on_another_key(
    billed_in_loop_search: tuple[TestClient, ModuleType], db_session_factory: Callable[[], Session]
) -> None:
    client, probe = billed_in_loop_search
    _price(client, "anthropic:claude-opus-4")
    price_fake_search(client, MASTER, per_million=1000.0)
    key = _key(client)

    response, provider = _chat_with_search(client, key, "anthropic:claude-opus-4")

    assert response.status_code == status.HTTP_200_OK, response.text
    assert {call.kwargs["api_key"] for call in provider.call_args_list} == {"sk-ant"}
    assert [gate for gate, _ in probe.GATES] == ["apply_due_credit", "require_funds_on_deposit"]
    assert probe.GATE_TOOLS == [frozenset({"otari:web_search"})]
    # The search alone: the model ran on the gateway's own key, which the organization's funds do not pay for.
    assert [(entry[0], entry[2]) for entry in probe.LEDGER] == [("charge", SEARCH_COST)]
    assert probe.TOOLS == [{"otari:web_search": ToolUse(calls=1, amount=SEARCH_COST)}]
    user = _user(client)
    assert user["spend"] == pytest.approx(float(2 * COST + SEARCH_COST))
    assert user["reserved"] == pytest.approx(0.0)
    assert _deployment_paid_flags(db_session_factory) == [False]


def test_a_search_on_the_deployments_account_is_refused_without_funds_before_the_model_is_called(
    billed_in_loop_search: tuple[TestClient, ModuleType],
) -> None:
    client, probe = billed_in_loop_search
    _price(client, "anthropic:claude-opus-4")
    price_fake_search(client, MASTER, per_million=1000.0)
    key = _key(client)
    probe.FUNDS["available"] = Decimal("0")

    response, provider = _chat_with_search(client, key, "anthropic:claude-opus-4")

    assert response.status_code == status.HTTP_402_PAYMENT_REQUIRED, response.text
    assert response.headers[ERROR_CODE_HEADER] == INSUFFICIENT_FUNDS
    # Named, so it reads apart from the refusal of a model the deployment pays for.
    assert response.json()["detail"] == (
        "Insufficient funds ($0.00) to serve this request. The gateway tool 'otari_web_search' runs on this "
        "deployment's account, which this organization's funds pay for. Add funds, or leave it out of tools[]."
    )
    provider.assert_not_called()
    assert probe.LEDGER == []
    assert _user(client)["reserved"] == pytest.approx(0.0)
    rows = client.get(f"{API_ROOT}/usage", params={"user_id": "billed"}, headers=MASTER).json()
    assert [(row["status"], row["status_code"], row["model"]) for row in rows] == [("error", 402, "otari:web_search")]


def _tool_reasons(client: TestClient, headers: dict[str, str]) -> dict[str, tuple[bool, str | None]]:
    response = client.get(f"{API_ROOT}/tools", headers=headers)
    assert response.status_code == status.HTTP_200_OK, response.text
    return {tool["id"]: (tool["available"], tool["reason"]) for tool in response.json()["data"]}


def test_tools_lists_a_search_on_the_deployments_account_as_unavailable_without_funds(
    billed_in_loop_search: tuple[TestClient, ModuleType],
) -> None:
    client, probe = billed_in_loop_search
    key = _key(client)

    assert _tool_reasons(client, _auth(key))["otari_web_search"] == (True, None)

    probe.FUNDS["available"] = Decimal("0")
    assert _tool_reasons(client, _auth(key))["otari_web_search"] == (False, "insufficient_funds")
    # A master key belongs to no organization the gate would read, so it gets the deployment's answer.
    assert _tool_reasons(client, MASTER)["otari_web_search"] == (True, None)
    # Listing grants no credit and refuses nothing, so it starts none of what a refusal starts.
    assert {gate for gate, _ in probe.GATES} == {"has_funds_on_deposit"}


def test_tools_lists_a_search_on_the_organizations_key_as_available_without_funds(
    billed_in_loop_search: tuple[TestClient, ModuleType], monkeypatch: pytest.MonkeyPatch
) -> None:
    client, probe = billed_in_loop_search
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    key = _key(client)
    created = client.post(
        f"{API_ROOT}/organizations/me/web-search-keys",
        json={"provider": "tavily", "name": "ours", "api_key": "tvly-ours"},
        headers=MASTER,
    )
    assert created.status_code == status.HTTP_201_CREATED, created.text
    starred = client.post(f"{API_ROOT}/organizations/me/web-search-keys/{created.json()['id']}/default", headers=MASTER)
    assert starred.status_code == status.HTTP_200_OK, starred.text
    probe.FUNDS["available"] = Decimal("0")

    assert _tool_reasons(client, _auth(key))["otari_web_search"] == (True, None)


def test_code_execution_on_a_key_the_organization_brought_is_refused_by_name_without_funds(
    billed_sandbox: tuple[TestClient, ModuleType],
) -> None:
    client, probe = billed_sandbox
    _price(client, "anthropic:claude-opus-4")
    key = _key(client)
    probe.FUNDS["available"] = Decimal("0")

    assert _tool_reasons(client, _auth(key))["otari_code_execution"] == (False, "insufficient_funds")
    with patch("gateway.services.mcp_loop.acompletion", new=AsyncMock()) as provider:
        response = client.post(
            f"{API_ROOT}/chat/completions",
            json={
                "model": "anthropic:claude-opus-4",
                "messages": [{"role": "user", "content": "compute 17*23"}],
                "tools": [{"type": "otari_code_execution"}],
            },
            headers=_auth(key),
        )

    assert response.status_code == status.HTTP_402_PAYMENT_REQUIRED, response.text
    assert response.headers[ERROR_CODE_HEADER] == INSUFFICIENT_FUNDS
    assert "The gateway tool 'otari_code_execution' runs on this deployment's account" in response.json()["detail"]
    provider.assert_not_called()

    probe.FUNDS["available"] = Decimal("10")
    assert _tool_reasons(client, _auth(key))["otari_code_execution"] == (True, None)


def test_a_deployment_paid_model_and_its_search_are_charged_once_together(
    billed_in_loop_search: tuple[TestClient, ModuleType],
) -> None:
    client, probe = billed_in_loop_search
    _price(client, "openai:gpt-4o")
    price_fake_search(client, MASTER, per_million=1000.0)
    key = _key(client)

    response, provider = _chat_with_search(client, key, "openai:gpt-4o")

    assert response.status_code == status.HTTP_200_OK, response.text
    assert {call.kwargs["api_key"] for call in provider.call_args_list} == {"fleet-key"}
    assert [(entry[0], entry[2], entry[3]) for entry in probe.LEDGER] == [
        ("charge", 2 * COST + SEARCH_COST, "openai:gpt-4o")
    ]
    # The one charge names its search and that search's share, so the model's share is the rest.
    assert probe.TOOLS == [{"otari:web_search": ToolUse(calls=1, amount=SEARCH_COST)}]
    assert _user(client)["reserved"] == pytest.approx(0.0)


def _search(client: TestClient, key: dict[str, Any]) -> Any:
    return client.post(f"{API_ROOT}/search/fake-search", json={"query": "otari"}, headers=_auth(key))


def test_a_direct_search_on_the_deployments_account_is_gated_then_charged_what_it_cost(
    billed_search: tuple[TestClient, ModuleType], db_session_factory: Callable[[], Session]
) -> None:
    client, probe = billed_search
    key = _key(client)

    response = _search(client, key)

    assert response.status_code == status.HTTP_200_OK, response.text
    # What the organization was charged is what the answer says the call cost.
    assert (response.headers[COST_USD_HEADER], response.headers[PRICING_SOURCE_HEADER]) == ("0.007000", "reported")
    assert [gate for gate, _ in probe.GATES] == ["apply_due_credit", "require_funds_on_deposit"]
    assert [(entry[0], entry[2], entry[3]) for entry in probe.LEDGER] == [
        ("charge", Decimal("0.007"), "fake:fake-search")
    ]
    # A search is no model's call, so nothing funded per model may pay for it.
    assert probe.BILLED == [("gate", None), ("charge", None)]
    assert probe.TOOLS == [{"otari:web_search": ToolUse(calls=1, amount=Decimal("0.007"))}]
    user = _user(client)
    assert user["spend"] == pytest.approx(0.007)
    assert user["reserved"] == pytest.approx(0.0)
    assert _deployment_paid_flags(db_session_factory) == [True]


def test_a_direct_search_on_the_deployments_account_is_refused_without_funds(
    billed_search: tuple[TestClient, ModuleType], db_session_factory: Callable[[], Session]
) -> None:
    client, probe = billed_search
    key = _key(client)
    probe.FUNDS["available"] = Decimal("0")

    with patch("gateway.services.tools._direct_tool_call.run_direct_search", new=AsyncMock()) as searched:
        response = _search(client, key)

    assert response.status_code == status.HTTP_402_PAYMENT_REQUIRED, response.text
    assert response.headers[ERROR_CODE_HEADER] == INSUFFICIENT_FUNDS
    assert COST_USD_HEADER not in response.headers
    assert response.json()["detail"] == (
        "Insufficient funds ($0.00) to serve this request. The gateway tool 'fake-search' runs on this "
        "deployment's account, which this organization's funds pay for. Add funds to run it."
    )
    searched.assert_not_called()
    assert probe.LEDGER == []
    assert _user(client)["reserved"] == pytest.approx(0.0)
    rows = client.get(f"{API_ROOT}/usage", params={"user_id": "billed"}, headers=MASTER).json()
    assert [(row["status"], row["status_code"], row["model"]) for row in rows] == [("error", 402, "fake-search")]
    # Nothing was admitted, so nothing was paid for.
    assert _deployment_paid_flags(db_session_factory) == [False]


def test_a_direct_search_whose_funds_check_fails_is_refused_and_refunded(
    billed_search: tuple[TestClient, ModuleType], gateway_logs: pytest.LogCaptureFixture
) -> None:
    """A billing port that raises is not a refusal for want of funds: the search answers 503 and holds nothing."""
    client, probe = billed_search
    key = _key(client)
    probe.FAULTS.add("funds")

    with patch("gateway.services.tools._direct_tool_call.run_direct_search", new=AsyncMock()) as searched:
        response = _search(client, key)

    assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE, response.text
    assert response.json()["detail"] == "This organization's funds could not be checked. Retry later."
    searched.assert_not_called()
    assert probe.LEDGER == []
    assert _user(client)["reserved"] == pytest.approx(0.0)
    rows = client.get(f"{API_ROOT}/usage", params={"user_id": "billed"}, headers=MASTER).json()
    assert [(row["status"], row["status_code"], row["model"]) for row in rows] == [("error", 503, "fake-search")]
    errors = [r.getMessage() for r in gateway_logs.records if r.levelno == logging.ERROR]
    assert any("deployment-paid search" in message and "RuntimeError" in message for message in errors), errors


def _tavily_answers(client: TestClient) -> None:
    async def web_provider_client() -> AsyncIterator[httpx.AsyncClient]:
        answer = {"results": [{"url": "https://otari.example", "title": "Otari", "content": "A snippet."}]}
        transport = httpx.MockTransport(lambda _: httpx.Response(200, json=answer))
        async with httpx.AsyncClient(transport=transport) as answering:
            yield answering

    client.app.dependency_overrides[get_web_provider_client] = web_provider_client  # type: ignore[attr-defined]


def test_a_direct_search_on_the_legacy_settings_is_charged_at_the_web_search_rate(
    billed_legacy_search: tuple[TestClient, ModuleType], db_session_factory: Callable[[], Session]
) -> None:
    client, probe = billed_legacy_search
    _price(client, "otari:web_search")
    key = _key(client)
    _tavily_answers(client)

    response = client.post(f"{API_ROOT}/search", json={"query": "otari"}, headers=_auth(key))

    assert response.status_code == status.HTTP_200_OK, response.text
    assert (response.headers[COST_USD_HEADER], response.headers[PRICING_SOURCE_HEADER]) == ("0.001000", "deployment")
    assert [gate for gate, _ in probe.GATES] == ["apply_due_credit", "require_funds_on_deposit"]
    assert [(entry[0], entry[2], entry[3]) for entry in probe.LEDGER] == [("charge", Decimal("0.001"), "tavily:tavily")]
    assert probe.TOOLS == [{"otari:web_search": ToolUse(calls=1, amount=Decimal("0.001"))}]
    assert _user(client)["spend"] == pytest.approx(0.001)
    assert _deployment_paid_flags(db_session_factory) == [True]


def test_a_direct_search_on_the_legacy_settings_is_refused_without_funds(
    billed_legacy_search: tuple[TestClient, ModuleType], db_session_factory: Callable[[], Session]
) -> None:
    client, probe = billed_legacy_search
    _price(client, "otari:web_search")
    key = _key(client)
    probe.FUNDS["available"] = Decimal("0")

    with patch("gateway.services.tools._direct_tool_call.run_direct_search", new=AsyncMock()) as searched:
        response = client.post(f"{API_ROOT}/search/tavily", json={"query": "otari"}, headers=_auth(key))

    assert response.status_code == status.HTTP_402_PAYMENT_REQUIRED, response.text
    assert response.headers[ERROR_CODE_HEADER] == INSUFFICIENT_FUNDS
    searched.assert_not_called()
    assert probe.LEDGER == []
    assert _user(client)["reserved"] == pytest.approx(0.0)
    assert _deployment_paid_flags(db_session_factory) == [False]


def test_a_tool_admission_that_does_not_commit_is_not_served(
    billed_in_loop_search: tuple[TestClient, ModuleType],
) -> None:
    client, probe = billed_in_loop_search
    _price(client, "anthropic:claude-opus-4")
    price_fake_search(client, MASTER, per_million=1000.0)
    key = _key(client)

    with patch("gateway.api.routes._pipeline.release_session", new=AsyncMock(return_value=False)):
        response, provider = _chat_with_search(client, key, "anthropic:claude-opus-4")

    assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE, response.text
    provider.assert_not_called()
    assert probe.LEDGER == []
    assert _user(client)["reserved"] == pytest.approx(0.0)


def test_a_direct_search_whose_admission_does_not_commit_is_not_served(
    billed_search: tuple[TestClient, ModuleType],
) -> None:
    client, probe = billed_search
    key = _key(client)

    with (
        patch("gateway.api.deps.release_session", new=AsyncMock(return_value=False)),
        patch("gateway.services.tools._direct_tool_call.run_direct_search", new=AsyncMock()) as searched,
    ):
        response = _search(client, key)

    assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE, response.text
    assert response.json()["detail"] == "This organization's funds could not be checked. Retry later."
    searched.assert_not_called()
    assert probe.LEDGER == []
    assert _user(client)["reserved"] == pytest.approx(0.0)
