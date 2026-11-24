"""A hosted provider's routing locality, checked by a policy's eligibility at compile time and at dispatch."""

from collections.abc import Generator, Iterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from any_llm.types.model import Model
from fastapi.testclient import TestClient

from gateway.core.config import API_KEY_HEADER, API_ROOT
from gateway.models.routing import RoutingConfig
from gateway.services.model_discovery_service import ProviderDiscovery
from gateway.services.routing import compiler as compiler_module
from gateway.services.secret_box import generate_secret_key

from .conftest import build_test_client, shared_test_config
from .test_routing_policies import _completion

HEADERS = {API_KEY_HEADER: "Bearer test-master-key"}


@pytest.fixture(autouse=True)
def _environment(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """A shell key would credential the ladder and skip the hosted rung."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    monkeypatch.setattr(compiler_module, "_warned_eligibility", set())

    async def _stub(impl_name: str, **_: object) -> ProviderDiscovery:
        return ProviderDiscovery(
            provider=impl_name, models=[Model(id="gpt-4o", object="model", created=0, owned_by=impl_name)]
        )

    monkeypatch.setattr("gateway.services.providers._hosted_provider_service.test_provider_credentials", _stub)
    yield


@pytest.fixture
def client(postgres_url: str) -> Generator[TestClient]:
    policies = {
        "cloud-first": {
            "select": [{"default": "openai-eu:gpt-4o"}],
            "on_failure": ["openai:gpt-4o"],
            "eligibility": {"allowed_localities": ["cloud"]},
        },
        "eu-only": {"select": [{"default": "openai-eu:gpt-4o"}], "eligibility": {"allowed_localities": ["cloud"]}},
        "us-only": {"select": [{"default": "openai:gpt-4o"}], "eligibility": {"allowed_localities": ["cloud"]}},
    }
    config = shared_test_config(postgres_url, {}).model_copy(
        update={"routing": RoutingConfig.model_validate({"policies": policies})}
    )
    for client in build_test_client(config):
        assert client.post(f"{API_ROOT}/users", json={"user_id": "u1"}, headers=HEADERS).status_code == 200
        for model_key in ("openai:gpt-4o", "openai-eu:gpt-4o"):
            response = client.post(
                f"{API_ROOT}/pricing",
                json={"model_key": model_key, "input_price_per_million": 2.5, "output_price_per_million": 10},
                headers=HEADERS,
            )
            assert response.status_code == 200, response.text
        _hosted(client, "openai", routing_locality="cloud")
        _hosted(client, "openai-eu", provider="openai")
        yield client


def _hosted(client: TestClient, name: str, *, provider: str | None = None, **fields: Any) -> None:
    response = client.post(
        f"{API_ROOT}/hosted-providers",
        json={"provider": provider or name, "name": name, "api_key": f"sk-{name}", **fields},
        headers=HEADERS,
    )
    assert response.status_code == 201, response.text


def _chat(client: TestClient, model: str) -> Any:
    return client.post(
        f"{API_ROOT}/chat/completions",
        json={"model": model, "messages": [{"role": "user", "content": "hi"}], "user": "u1"},
        headers=HEADERS,
    )


def test_an_undeclared_hosted_provider_is_left_out_of_the_plan(client: TestClient) -> None:
    with patch("gateway.api.routes.chat.acompletion", new=AsyncMock(return_value=_completion("gpt-4o"))) as mock:
        resp = _chat(client, "cloud-first")

    assert resp.status_code == 200, resp.text
    assert mock.await_count == 1
    assert mock.await_args is not None
    assert mock.await_args.kwargs["api_key"] == "sk-openai"


def test_the_walker_holds_the_hosted_credential_to_the_policy(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # As if the compile-time index vouched for both: the credential the hosted
    # rung resolves for openai-eu still declares no locality, so the walker
    # passes it over and the backup serves.
    monkeypatch.setattr(compiler_module, "hosted_routing_locality", lambda _name: "cloud")
    with patch("gateway.api.routes.chat.acompletion", new=AsyncMock(return_value=_completion("gpt-4o"))) as mock:
        resp = _chat(client, "cloud-first")

    assert resp.status_code == 200, resp.text
    assert mock.await_count == 1
    assert mock.await_args is not None
    assert mock.await_args.kwargs["api_key"] == "sk-openai"


def test_nothing_is_sent_when_no_hosted_credential_qualifies(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(compiler_module, "hosted_routing_locality", lambda _name: "cloud")
    with patch("gateway.api.routes.chat.acompletion", new=AsyncMock()) as mock:
        resp = _chat(client, "eu-only")

    assert resp.status_code == 403, resp.text
    mock.assert_not_awaited()
    assert "gpt-4o" not in resp.text


@pytest.mark.parametrize(
    ("policy", "status", "uploaded_on"), [("us-only", 200, ["sk-openai"]), ("eu-only", 403, [None])]
)
def test_an_attachment_is_uploaded_only_on_a_credential_the_policy_admits(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, policy: str, status: int, uploaded_on: list[str | None]
) -> None:
    monkeypatch.setattr(compiler_module, "hosted_routing_locality", lambda _name: "cloud")
    credentials: list[Any] = []

    async def _normalize(messages: Any, **kwargs: Any) -> tuple[Any, SimpleNamespace]:
        # What moving an attachment past the provider's inline limit asks for first.
        credentials.append(await kwargs["hosted_credential"]())
        return messages, SimpleNamespace(sandbox_inputs=[], vision_usage=lambda: None)

    with (
        patch("gateway.api.routes.chat.normalize_request_messages", new=_normalize),
        patch("gateway.api.routes.chat.acompletion", new=AsyncMock(return_value=_completion("gpt-4o"))),
    ):
        resp = _chat(client, policy)

    assert resp.status_code == status, resp.text
    assert [credential.api_key if credential is not None else None for credential in credentials] == uploaded_on
