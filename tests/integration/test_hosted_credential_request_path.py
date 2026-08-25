"""An overlay-bound ``ModelProviderPort`` serves a request, end to end.

``ModelProviderPort`` is only worth binding if the request path asks it
(otari#757), and only worth asking if it cannot displace a credential the
gateway already holds. Both are properties of a booted app rather than of a
function, so this exercises them against one: the same config is served by a
gateway with an overlay bootstrap and by a gateway without, and what reaches
any-llm is compared.
"""

import logging
import sys
from collections.abc import Generator
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from gateway.core.config import API_KEY_HEADER, GatewayConfig
from gateway.log_config import logger as gateway_logger
from gateway.models.entities import User

from .conftest import build_test_client

HEADERS = {API_KEY_HEADER: "Bearer test-master-key"}

# An overlay in miniature: one rebound port, a fleet that serves anything the
# gateway itself could not credential, and one upstream the organization is
# refused. Written to a file and imported by dotted path, because binding it
# through ``OTARI_BOOTSTRAP`` is half of what is under test.
OVERLAY_MODULE = '''
from gateway.container import Container
from gateway.ports.model_provider_port import HostedAccessDeniedError, HostedCredential, ModelProviderPort


class FleetModelProviderAdapter:
    """Serves a candidate from a deployment-owned fleet, and refuses one upstream."""

    def __init__(self, session):
        self.session = session

    async def resolve_hosted_credential(self, *, organization_id, workspace_id, provider, model):
        if provider == "groq":
            # Stands in for the ordinary failure of a network-backed adapter:
            # an httpx timeout, a reset connection, a database error.
            raise RuntimeError("fleet credential service is unreachable")
        if provider == "mistral":
            raise HostedAccessDeniedError(
                "FleetAdapter: this organization has no mistral entitlement",
                workspace_id=workspace_id,
            )
        return HostedCredential(
            api_key="fleet-key",
            api_base="https://fleet.test/v1",
            response_provider="together",
        )


def register(container: Container) -> None:
    container.bind(ModelProviderPort, FleetModelProviderAdapter)
'''


class _MockCompletionError(Exception):
    """Raised to short-circuit the mocked acompletion after capturing kwargs."""


@pytest.fixture(autouse=True)
def _no_ambient_provider_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """A developer's own shell keys would credential the ladder and skip the port."""
    for name in ("OPENAI_API_KEY", "MISTRAL_API_KEY", "TOGETHER_API_KEY", "VLLM_API_KEY"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def overlay_on_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[str]:
    (tmp_path / "otari_fleet_overlay.py").write_text(OVERLAY_MODULE)
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("otari_fleet_overlay", None)
    yield "otari_fleet_overlay"
    sys.modules.pop("otari_fleet_overlay", None)


def _config(postgres_url: str, *, bootstrap: str | None = None) -> GatewayConfig:
    return GatewayConfig(
        database_url=postgres_url,
        master_key="test-master-key",
        auto_migrate=False,
        require_pricing=False,
        model_discovery=False,
        # One provider the gateway credentials itself, so BYO precedence has
        # something concrete to be checked against.
        providers={"anthropic": {"api_key": "sk-ant"}},
        bootstrap=bootstrap,
    )


@pytest.fixture
def overlay_client(postgres_url: str, overlay_on_path: str) -> Generator[TestClient]:
    yield from build_test_client(_config(postgres_url, bootstrap=f"{overlay_on_path}:register"))


@pytest.fixture
def plain_client(postgres_url: str) -> Generator[TestClient]:
    """The same gateway with no bootstrap: the core adapter answers None for everything."""
    yield from build_test_client(_config(postgres_url))


def _create_user(client: TestClient, user_id: str = "u1") -> None:
    response = client.post("/v1/users", json={"user_id": user_id, "alias": user_id}, headers=HEADERS)
    assert response.status_code == 200, response.text


def _post_chat_capture(client: TestClient, model: str) -> tuple[dict[str, object], int]:
    """POST a completion with the provider call mocked; return its kwargs and the status."""
    captured: dict[str, object] = {}

    async def fake_acompletion(**kwargs: object) -> None:
        captured.update(kwargs)
        raise _MockCompletionError

    with patch("gateway.api.routes.chat.acompletion", new=AsyncMock(side_effect=fake_acompletion)):
        response = client.post(
            "/v1/chat/completions",
            json={"model": model, "messages": [{"role": "user", "content": "Hi"}], "user": "u1"},
            headers=HEADERS,
        )
    return captured, response.status_code


def test_the_overlay_fleet_serves_a_candidate_the_gateway_cannot(overlay_client: TestClient) -> None:
    _create_user(overlay_client)

    captured, _ = _post_chat_capture(overlay_client, "openai:gpt-4o")

    assert captured, "the provider call was never made"
    assert captured["api_key"] == "fleet-key"
    assert captured["api_base"] == "https://fleet.test/v1"
    # Dispatched against the upstream the credential named, not the one asked for.
    assert captured["model"] == "together:gpt-4o"


def test_the_plain_build_dispatches_the_same_candidate_uncredentialed(plain_client: TestClient) -> None:
    """The acceptance case for every deployment with no overlay: nothing changed."""
    _create_user(plain_client)

    captured, _ = _post_chat_capture(plain_client, "openai:gpt-4o")

    assert captured, "the provider call was never made"
    assert "api_key" not in captured
    assert "api_base" not in captured
    assert captured["model"] == "openai:gpt-4o"


def test_a_configured_provider_key_is_not_displaced_by_the_fleet(overlay_client: TestClient) -> None:
    """BYO stays upstream: the port is asked last, and here it is not asked at all."""
    _create_user(overlay_client)

    captured, _ = _post_chat_capture(overlay_client, "anthropic:claude-opus-4")

    assert captured["api_key"] == "sk-ant"
    assert captured["model"] == "anthropic:claude-opus-4"


def test_a_keyless_local_backend_is_not_claimed_by_the_fleet(overlay_client: TestClient) -> None:
    """Self-hosting stays a first-class path upstream of the port, even with an overlay bound.

    ``ollama:llama3`` needs no credential at all, so an empty kwargs is not a
    missing key and the fleet is never asked. Without this, a self-hoster's local
    traffic would silently start being served (and metered) from somewhere else.
    """
    _create_user(overlay_client)

    captured, _ = _post_chat_capture(overlay_client, "ollama:llama3")

    assert captured, "the provider call was never made"
    assert "api_key" not in captured
    assert captured["model"] == "ollama:llama3"


def test_a_self_hosted_backend_declaring_a_key_is_not_claimed_by_the_fleet(overlay_client: TestClient) -> None:
    """``vllm`` declares ``VLLM_API_KEY`` but any-llm never insists on it.

    The same invariant as the case above, for the half of the keyless set the
    declaration cannot identify: a bare ``vllm:my-model`` reaches a local vLLM at
    any-llm's default ``http://localhost:8000/v1`` with nothing configured here,
    so the fleet must not be asked about it.
    """
    _create_user(overlay_client)

    captured, _ = _post_chat_capture(overlay_client, "vllm:my-model")

    assert captured, "the provider call was never made"
    assert "api_key" not in captured
    assert "api_base" not in captured
    assert captured["model"] == "vllm:my-model"


def test_an_unpriced_re_keyed_upstream_is_logged(overlay_client: TestClient, caplog: pytest.LogCaptureFixture) -> None:
    """A request about to settle free says so in the log.

    The pricing gate ran against the name the caller asked for, and settlement
    reprices on the upstream that served, so an unpriced ``response_provider``
    settles at no cost and releases the whole hold. Refusing is deliberately left
    out (an overlay's own fleet would be unusable until priced), which is why the
    signal has to exist: without it the only trace is a generic unpriced-model
    warning at settlement, indistinguishable from an ordinary unpriced model.
    """
    _create_user(overlay_client)

    # The `gateway` logger does not propagate, so caplog only sees it once its
    # handler is attached directly (the idiom in `test_provider_instances.py`).
    gateway_logger.addHandler(caplog.handler)
    caplog.set_level(logging.WARNING, logger="gateway")
    try:
        captured, _ = _post_chat_capture(overlay_client, "openai:gpt-4o")
    finally:
        gateway_logger.removeHandler(caplog.handler)

    assert captured["model"] == "together:gpt-4o"
    assert any("re-keyed openai:gpt-4o onto together:gpt-4o" in record.getMessage() for record in caplog.records), (
        f"no re-key pricing warning in {[r.getMessage() for r in caplog.records]}"
    )


def test_an_adapter_failure_returns_the_budget_hold(overlay_client: TestClient, postgres_url: str) -> None:
    """The reservation invariant, end to end against a real budget.

    An overlay adapter that fetches credentials over the network fails the
    ordinary ways, and `resolve_dispatch_provider` is called outside any `try`
    that would catch one, with no global handler above it. Before the guard this
    left the preamble's hold on `users.reserved` until the budget reset, which for
    a budget with no period is forever. The unit tests assert the refund helper is
    called; this one asserts the column it is called for.
    """
    budget = overlay_client.post(
        "/v1/budgets", json={"max_budget": 100.0, "budget_duration_sec": 86400}, headers=HEADERS
    ).json()
    assert (
        overlay_client.post(
            "/v1/users", json={"user_id": "held", "budget_id": budget["budget_id"]}, headers=HEADERS
        ).status_code
        == 200
    )
    # Priced, so the preamble reserves a real estimate rather than zero. Without
    # this the assertion below would hold whether or not anything was refunded.
    assert (
        overlay_client.post(
            "/v1/pricing",
            json={
                "model_key": "groq:llama-3.3-70b",
                "input_price_per_million": 1000.0,
                "output_price_per_million": 1000.0,
            },
            headers=HEADERS,
        ).status_code
        == 200
    )
    key = overlay_client.post("/v1/keys", json={"user_id": "held"}, headers=HEADERS).json()["key"]

    response = overlay_client.post(
        "/v1/chat/completions",
        json={"model": "groq:llama-3.3-70b", "messages": [{"role": "user", "content": "Hi" * 500}]},
        headers={API_KEY_HEADER: f"Bearer {key}"},
    )

    assert response.status_code == 502
    assert "fleet credential service" not in response.text

    engine = create_engine(postgres_url)
    try:
        with sessionmaker(bind=engine)() as session:
            user = session.query(User).filter(User.user_id == "held").one()
            assert float(user.reserved) == pytest.approx(0.0), (
                f"the budget hold leaked: users.reserved is {user.reserved}"
            )
            assert float(user.spend) == pytest.approx(0.0)
    finally:
        engine.dispose()


def test_a_refused_upstream_answers_403_without_naming_the_adapter(overlay_client: TestClient) -> None:
    _create_user(overlay_client)

    captured, status_code = _post_chat_capture(overlay_client, "mistral:mistral-large-latest")

    assert status_code == 403
    assert captured == {}, "a refused request must never reach a provider"


def test_the_refusal_body_carries_no_adapter_wording(overlay_client: TestClient) -> None:
    _create_user(overlay_client)

    with patch("gateway.api.routes.chat.acompletion", new=AsyncMock(side_effect=_MockCompletionError)):
        response = overlay_client.post(
            "/v1/chat/completions",
            json={
                "model": "mistral:mistral-large-latest",
                "messages": [{"role": "user", "content": "Hi"}],
                "user": "u1",
            },
            headers=HEADERS,
        )

    assert response.status_code == 403
    assert "FleetAdapter" not in response.text
    assert "entitlement" not in response.text
