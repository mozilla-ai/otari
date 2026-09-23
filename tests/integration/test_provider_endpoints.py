"""Provider endpoints a workspace or a user owns: management, and what a request to one does.

Endpoint URLs are public IP literals, so saving one needs no DNS. The provider
call is mocked, so nothing is dialed.
"""

from collections.abc import Generator, Iterator
from typing import Any, cast
from unittest.mock import patch

import httpx
import pytest
from any_llm.types.completion import ChatCompletion, ChatCompletionMessage, Choice, CompletionUsage
from any_llm.types.messages import MessageResponse, MessageUsage, TextBlock
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from gateway.core.config import API_KEY_HEADER, API_ROOT, GatewayConfig
from gateway.models.usage import UsageLog
from gateway.models.users import User
from gateway.services.secret_box import generate_secret_key

from .conftest import build_test_client

ENDPOINTS = f"{API_ROOT}/provider-endpoints"
HEADERS = {API_KEY_HEADER: "Bearer test-master-key"}
PUBLIC_BASE = "https://1.1.1.1/v1"


def _config(postgres_url: str, *, enabled: bool = True) -> GatewayConfig:
    return GatewayConfig(
        database_url=postgres_url,
        master_key="test-master-key",
        auto_migrate=False,
        # Fail closed on pricing, so a request that reaches an unpriced owned
        # endpoint proves the budget exemption skips the pricing gate too.
        require_pricing=True,
        default_pricing=False,
        model_discovery=False,
        providers={"openai": {"api_key": "sk-deployment"}, "home_lab": {"provider_type": "openai"}},
        provider_endpoints_enabled=enabled,
    )


@pytest.fixture(autouse=True)
def _secret_key(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    yield


@pytest.fixture
def client(postgres_url: str, clean_database: None) -> Generator[TestClient]:
    yield from build_test_client(_config(postgres_url))


@pytest.fixture
def disabled_client(postgres_url: str, clean_database: None) -> Generator[TestClient]:
    yield from build_test_client(_config(postgres_url, enabled=False))


def _user_with_key(client: TestClient, user_id: str, *, max_budget: float | None = None) -> dict[str, str]:
    body: dict[str, Any] = {"user_id": user_id}
    if max_budget is not None:
        budget = client.post(
            f"{API_ROOT}/budgets", json={"max_budget": max_budget, "budget_duration_sec": 86400}, headers=HEADERS
        )
        body["budget_id"] = budget.json()["budget_id"]
    assert client.post(f"{API_ROOT}/users", json=body, headers=HEADERS).status_code in (200, 201)
    resp = client.post(f"{API_ROOT}/keys", json={"key_name": user_id, "user_id": user_id}, headers=HEADERS)
    assert resp.status_code == 200, resp.text
    return {API_KEY_HEADER: f"Bearer {resp.json()['key']}"}


def _create(client: TestClient, **fields: Any) -> Any:
    body = {"name": "my-vllm", "provider": "openai", "api_base": PUBLIC_BASE, **fields}
    return client.post(ENDPOINTS, json=body, headers=HEADERS)


def _completion() -> ChatCompletion:
    return ChatCompletion(
        id="chatcmpl-x",
        object="chat.completion",
        created=0,
        model="qwen3",
        choices=[Choice(index=0, message=ChatCompletionMessage(role="assistant", content="hi"), finish_reason="stop")],
        usage=CompletionUsage(prompt_tokens=1_000_000, completion_tokens=500_000, total_tokens=1_500_000),
    )


def _chat(
    client: TestClient, headers: dict[str, str], model: str, **fields: Any
) -> tuple[Any, list[dict[str, Any]]]:
    calls: list[dict[str, Any]] = []

    async def fake_acompletion(**kwargs: Any) -> ChatCompletion:
        calls.append(kwargs)
        return _completion()

    with patch("gateway.api.routes.chat.acompletion", side_effect=fake_acompletion):
        resp = client.post(
            f"{API_ROOT}/chat/completions",
            json={"model": model, "messages": [{"role": "user", "content": "hi"}], **fields},
            headers=headers,
        )
    return resp, calls


# ----------------------------------------------------------------------------
# Management
# ----------------------------------------------------------------------------


def test_every_route_is_refused_until_the_deployment_turns_endpoints_on(disabled_client: TestClient) -> None:
    assert disabled_client.get(ENDPOINTS, headers=HEADERS).status_code == 403
    assert _create(disabled_client).status_code == 403


def test_an_api_key_cannot_manage_endpoints(client: TestClient) -> None:
    headers = _user_with_key(client, "alice")
    assert client.get(ENDPOINTS, headers=headers).status_code in (401, 403)


def test_create_read_update_delete(client: TestClient) -> None:
    _user_with_key(client, "alice")
    created = _create(client, user_id="alice", api_key="sk-alice-1234", default_params={"top_k": 20})
    assert created.status_code == 201, created.text
    body = created.json()
    assert body["user_id"] == "alice"
    assert body["last4"] == "1234"
    assert "api_key" not in body and "encrypted_api_key" not in body
    assert body["default_params"] == {"top_k": 20}

    listed = client.get(ENDPOINTS, params={"user_id": "alice"}, headers=HEADERS).json()
    assert listed["count"] == 1
    assert listed["data"][0]["id"] == body["id"]

    updated = client.patch(f"{ENDPOINTS}/{body['id']}", json={"name": "renamed", "api_key": None}, headers=HEADERS)
    assert updated.status_code == 200, updated.text
    assert updated.json()["name"] == "renamed"
    assert updated.json()["last4"] is None

    assert client.delete(f"{ENDPOINTS}/{body['id']}", headers=HEADERS).status_code == 204
    assert client.get(f"{ENDPOINTS}/{body['id']}", headers=HEADERS).status_code == 404


def test_credential_shaped_default_params_are_masked_and_survive_a_resubmitted_mask(client: TestClient) -> None:
    created = _create(client, default_params={"x_token": "secret-value", "top_k": 20}).json()
    assert created["default_params"] == {"x_token": "***", "top_k": 20}

    resubmitted = client.patch(
        f"{ENDPOINTS}/{created['id']}", json={"default_params": created["default_params"]}, headers=HEADERS
    )
    assert resubmitted.status_code == 200, resubmitted.text
    assert resubmitted.json()["default_params"] == {"x_token": "***", "top_k": 20}


def test_one_name_per_owner(client: TestClient) -> None:
    _user_with_key(client, "alice")
    assert _create(client).status_code == 201
    assert _create(client).status_code == 409
    # A user's endpoint of the same name is a different owner, and shadows it for them.
    assert _create(client, user_id="alice").status_code == 201
    assert _create(client, user_id="alice").status_code == 409


@pytest.mark.parametrize(
    ("fields", "reason"),
    [
        ({"name": "openai"}, "names a provider"),
        ({"name": "home_lab"}, "configured provider instance"),
        ({"name": "hosted"}, "names a provider"),
        ({"name": "my:vllm"}, "letters, digits"),
        ({"provider": "bedrock"}, "must be one of"),
        ({"api_base": "http://127.0.0.1:8000/v1"}, "api_base refused"),
        ({"api_base": "http://169.254.169.254/latest"}, "api_base refused"),
        ({"api_base": "http://100.100.100.200/"}, "api_base refused"),
        ({"default_params": {"api_base": "http://10.0.0.1"}}, "cannot be endpoint defaults"),
        ({"default_params": {"model": "other"}}, "cannot be endpoint defaults"),
    ],
)
def test_refused_endpoints(client: TestClient, fields: dict[str, Any], reason: str) -> None:
    resp = _create(client, **fields)
    assert resp.status_code == 400, resp.text
    assert reason in resp.json()["detail"]


def test_an_unknown_owner_is_a_404(client: TestClient) -> None:
    assert _create(client, user_id="nobody").status_code == 404
    assert _create(client, workspace_id="00000000-0000-0000-0000-000000000001").status_code == 404


def test_an_update_is_held_to_the_same_rules(client: TestClient) -> None:
    created = _create(client).json()
    resp = client.patch(f"{ENDPOINTS}/{created['id']}", json={"api_base": "http://10.1.2.3/v1"}, headers=HEADERS)
    assert resp.status_code == 400


# ----------------------------------------------------------------------------
# Dispatch
# ----------------------------------------------------------------------------


def test_a_request_to_an_owned_endpoint_uses_its_key_and_skips_the_budget(
    client: TestClient, db_session: Session
) -> None:
    """A maxed-out, unpriced request succeeds, logs under the endpoint's name, and debits nothing."""
    headers = _user_with_key(client, "alice", max_budget=0.01)
    assert _create(client, user_id="alice", api_key="sk-alice-1234").status_code == 201

    resp, calls = _chat(client, headers, "my-vllm:qwen3")

    assert resp.status_code == 200, resp.text
    [call] = calls
    assert call["model"] == "openai:qwen3"
    assert call["api_base"] == PUBLIC_BASE
    assert call["api_key"] == "sk-alice-1234"
    assert isinstance(call["client_args"]["http_client"], httpx.AsyncClient)
    assert call["client_args"]["http_client"].follow_redirects is False

    user = db_session.query(User).filter(User.user_id == "alice").one()
    assert float(user.spend) == pytest.approx(0.0)
    assert float(user.reserved) == pytest.approx(0.0)
    row = db_session.query(UsageLog).filter(UsageLog.user_id == "alice").one()
    assert row.counts_toward_budget is False
    assert row.provider == "my-vllm"


def test_a_keyless_endpoint_never_falls_back_to_the_deployment_key(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-operator")
    headers = _user_with_key(client, "alice")
    assert _create(client, user_id="alice").status_code == 201

    resp, [call] = _chat(client, headers, "my-vllm:qwen3")

    assert resp.status_code == 200, resp.text
    assert call["api_key"] not in {"sk-operator", "sk-deployment"}


def test_default_params_go_beneath_the_callers_fields(client: TestClient) -> None:
    headers = _user_with_key(client, "alice")
    defaults = {"chat_template_kwargs": {"enable_thinking": False}, "temperature": 0.1}
    assert _create(client, user_id="alice", default_params=defaults).status_code == 201

    resp, [call] = _chat(client, headers, "my-vllm:qwen3", temperature=0.9)

    assert resp.status_code == 200, resp.text
    assert call["temperature"] == 0.9
    assert call["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}


def test_a_users_endpoint_shadows_the_workspaces_for_that_user_alone(client: TestClient) -> None:
    alice = _user_with_key(client, "alice")
    bob = _user_with_key(client, "bob")
    assert _create(client, api_base="https://1.1.1.1/shared").status_code == 201
    assert _create(client, user_id="alice", api_base="https://1.1.1.1/alice").status_code == 201

    _, [alice_call] = _chat(client, alice, "my-vllm:qwen3")
    _, [bob_call] = _chat(client, bob, "my-vllm:qwen3")

    assert alice_call["api_base"] == "https://1.1.1.1/alice"
    assert bob_call["api_base"] == "https://1.1.1.1/shared"


def test_another_users_endpoint_does_not_resolve(client: TestClient) -> None:
    _user_with_key(client, "alice")
    bob = _user_with_key(client, "bob")
    assert _create(client, user_id="alice").status_code == 201

    resp, calls = _chat(client, bob, "my-vllm:qwen3")

    # Unresolved, so unpriced, so refused at the pricing gate before any dispatch.
    assert resp.status_code == 402
    assert calls == []


def test_a_deleted_endpoint_stops_resolving(client: TestClient) -> None:
    headers = _user_with_key(client, "alice")
    created = _create(client, user_id="alice").json()
    client.delete(f"{ENDPOINTS}/{created['id']}", headers=HEADERS)

    resp, calls = _chat(client, headers, "my-vllm:qwen3")

    # Unresolved, so unpriced, so refused at the pricing gate before any dispatch.
    assert resp.status_code == 402
    assert calls == []


def test_an_anthropic_endpoint_serves_the_messages_route_with_its_defaults(client: TestClient) -> None:
    headers = _user_with_key(client, "alice")
    created = _create(
        client, user_id="alice", provider="anthropic", api_base="https://1.1.1.1", default_params={"top_k": 5}
    )
    assert created.status_code == 201, created.text
    calls: list[dict[str, Any]] = []

    async def fake_amessages(**kwargs: Any) -> MessageResponse:
        calls.append(kwargs)
        return MessageResponse(
            id="msg_test",
            type="message",
            role="assistant",
            model="claude",
            content=[TextBlock(type="text", text="hi", citations=None)],
            stop_reason=cast(Any, "end_turn"),
            stop_sequence=None,
            usage=MessageUsage(input_tokens=5, output_tokens=2),
            container=None,
        )

    with patch("gateway.api.routes.messages.amessages", side_effect=fake_amessages):
        resp = client.post(
            f"{API_ROOT}/messages",
            json={"model": "my-vllm:claude", "max_tokens": 16, "messages": [{"role": "user", "content": "hi"}]},
            headers=headers,
        )

    assert resp.status_code == 200, resp.text
    [call] = calls
    assert call["model"] == "anthropic:claude"
    assert call["api_base"] == "https://1.1.1.1"
    assert call["extra_body"] == {"top_k": 5}
