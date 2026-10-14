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

import itertools
import logging
import sys
from collections.abc import AsyncIterator, Generator, Iterator
from contextlib import contextmanager
from decimal import Decimal
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import AsyncMock, patch

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
from fastapi import status
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.config import API_KEY_HEADER, API_ROOT, GatewayConfig
from gateway.core.error_codes import ERROR_CODE_HEADER, INSUFFICIENT_FUNDS
from gateway.log_config import logger as gateway_logger

from .conftest import build_test_client

MODULE = "probe_deployment_paid"
PROBE_BOOTSTRAP = """
from decimal import Decimal

from gateway.container import Container
from gateway.ports.billing_port import BillingPort, InsufficientFundsError
from gateway.ports.model_provider_port import HostedCredential, ModelProviderPort

GATES = []  # Every gate call, landed or not.
LEDGER = []  # What landed: a session's entries move here when it commits.
PENDING = {}  # Entries written on a session, by its id, until it commits, rolls back or closes.
SESSIONS = []
FUNDS = {"available": Decimal("10")}
FAULTS = set()


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

    async def require_funds_on_deposit(self, *, organization_id):
        GATES.append(("require_funds_on_deposit", str(organization_id)))
        if FUNDS["available"] <= 0:
            raise InsufficientFundsError(FUNDS["available"])

    async def require_unheld_funds(self, *, organization_id):
        raise AssertionError("no holds on the completion path")

    async def hold(self, *, organization_id, amount):
        raise AssertionError("no holds on the completion path")

    async def release_hold(self, *, organization_id, amount):
        raise AssertionError("no holds on the completion path")

    async def charge(self, *, organization_id, amount, description=None, actor_user_id=None, api_key_id=None):
        if "charge" in FAULTS:
            raise RuntimeError("ledger down")
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
        bootstrap=f"{name}:register",
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
    billed: tuple[TestClient, ModuleType],
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
    user = _user(client)
    assert user["spend"] == pytest.approx(float(COST))
    assert user["reserved"] == pytest.approx(0.0)


def test_a_deployment_paid_stream_is_charged_once_after_it_ends(billed: tuple[TestClient, ModuleType]) -> None:
    client, probe = billed
    _price(client, "openai:gpt-4o")
    key = _key(client)

    response, _ = _chat(client, key, "openai:gpt-4o", stream=True)

    assert response.status_code == status.HTTP_200_OK, response.text
    assert "[DONE]" in response.text
    assert [(entry[0], entry[2], entry[3]) for entry in probe.LEDGER] == [("charge", COST, "openai:gpt-4o")]
    assert _user(client)["reserved"] == pytest.approx(0.0)


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
            lambda detail: {"detail": {"type": "error", "error": {"type": "invalid_request_error", "message": detail}}},
        ),
        ("/responses", {"input": "Hi"}, lambda detail: {"detail": detail, "code": INSUFFICIENT_FUNDS}),
    ],
    ids=["chat", "messages", "responses"],
)
def test_an_organization_without_funds_is_refused_before_dispatch(
    billed: tuple[TestClient, ModuleType], path: str, body: dict[str, Any], error_body: Any
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


def test_a_request_on_the_gateways_own_key_never_reaches_billing(billed: tuple[TestClient, ModuleType]) -> None:
    client, probe = billed
    _price(client, "anthropic:claude-opus-4")
    key = _key(client)

    response, provider = _chat(client, key, "anthropic:claude-opus-4")

    assert response.status_code == status.HTTP_200_OK, response.text
    assert provider.call_args.kwargs["api_key"] == "sk-ant"
    assert probe.GATES == []
    assert probe.LEDGER == []
    assert _user(client)["spend"] == pytest.approx(float(COST))


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
