"""The end-user identifier a completion sends its provider."""

from collections.abc import Callable, Generator
from typing import Any

import httpx
import pytest
from any_llm.types.completion import ChatCompletion, ChatCompletionMessage, Choice, CompletionUsage
from fastapi.testclient import TestClient

from conftest import InstallControlPlane
from gateway.api.deps import reset_config
from gateway.api.routes._pipeline import NO_UPSTREAM_USER_DETAIL
from gateway.core.config import API_KEY_HEADER, API_ROOT, GatewayConfig
from gateway.core.database import reset_db
from gateway.services.inference import UPSTREAM_USER_FIELD, upstream_user_identifier

from .conftest import app_for, build_test_client

_MASTER_KEY = "test-master-key"
_MASTER = {API_KEY_HEADER: f"Bearer {_MASTER_KEY}"}
_KEY = "k" * 32
_PROVIDERS = {
    "openai": {"api_key": "sk-openai"},
    "anthropic": {"api_key": "sk-anthropic"},
    "gemini": {"api_key": "gemini-key"},
    "bedrock": {"api_key": "bedrock-key", "region_name": "us-east-1"},
}

ClientFactory = Callable[..., TestClient]


class _Stop(Exception):
    """Ends a provider call once its kwargs are captured."""


def _completion() -> ChatCompletion:
    return ChatCompletion(
        id="chatcmpl-1",
        object="chat.completion",
        created=1700000000,
        model="gpt-4o-mini",
        choices=[Choice(index=0, message=ChatCompletionMessage(role="assistant", content="hi"), finish_reason="stop")],
        usage=CompletionUsage(prompt_tokens=4, completion_tokens=2, total_tokens=6),
    )


@pytest.fixture
def make_client(postgres_url: str) -> Generator[ClientFactory]:
    clients: list[Generator[TestClient]] = []

    def make(**overrides: Any) -> TestClient:
        config = GatewayConfig(
            database_url=postgres_url,
            master_key=_MASTER_KEY,
            auto_migrate=False,
            require_pricing=False,
            providers=_PROVIDERS,
            **overrides,
        )
        generator = build_test_client(config)
        clients.append(generator)
        return next(generator)

    yield make
    for generator in clients:
        generator.close()


@pytest.fixture
def captured(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    async def fake_acompletion(**kwargs: Any) -> ChatCompletion:
        calls.append(kwargs)
        raise _Stop

    monkeypatch.setattr("gateway.api.routes.chat.acompletion", fake_acompletion)
    return calls


def _keyed_user(client: TestClient, user_id: str = "alice") -> tuple[dict[str, str], str]:
    """A key for ``user_id``, and the workspace it bills to."""
    assert client.post(f"{API_ROOT}/users", json={"user_id": user_id}, headers=_MASTER).status_code == 200
    created = client.post(f"{API_ROOT}/keys", json={"key_name": "k", "user_id": user_id}, headers=_MASTER)
    assert created.status_code == 200
    key = created.json()
    workspace = client.get(f"{API_ROOT}/keys/{key['id']}", headers=_MASTER).json()["workspace_id"]
    return {API_KEY_HEADER: f"Bearer {key['key']}"}, workspace


def _chat(client: TestClient, headers: dict[str, str], model: str = "openai:gpt-4o-mini", **body: Any) -> None:
    client.post(
        f"{API_ROOT}/chat/completions",
        json={"model": model, "messages": [{"role": "user", "content": "hi"}], **body},
        headers=headers,
    )


def test_openai_gets_a_digest_of_the_workspace_and_resolved_user(
    make_client: ClientFactory, captured: list[dict[str, Any]]
) -> None:
    client = make_client()
    headers, workspace = _keyed_user(client)

    _chat(client, headers)

    assert captured[0]["user"] == upstream_user_identifier(workspace, "alice", None)
    assert UPSTREAM_USER_FIELD not in captured[0]


def test_a_user_tag_replaces_the_resolved_user(make_client: ClientFactory, captured: list[dict[str, Any]]) -> None:
    client = make_client(reject_user_mismatch=False)
    headers, workspace = _keyed_user(client)

    _chat(client, headers, user="end-user-7")

    assert captured[0]["user"] == upstream_user_identifier(workspace, "end-user-7", None)


def test_echoing_your_own_user_gives_the_same_identifier(
    make_client: ClientFactory, captured: list[dict[str, Any]]
) -> None:
    client = make_client()
    headers, _ = _keyed_user(client)

    _chat(client, headers)
    _chat(client, headers, user="alice")

    assert captured[0]["user"] == captured[1]["user"]


def test_the_master_key_user_is_digested_too(make_client: ClientFactory, captured: list[dict[str, Any]]) -> None:
    client = make_client()
    headers, _ = _keyed_user(client)

    _chat(client, headers)
    _chat(client, _MASTER, user="alice")

    assert captured[1]["user"] == captured[0]["user"]
    assert captured[1]["user"] != "alice"


def test_a_key_makes_the_digest_keyed(make_client: ClientFactory, captured: list[dict[str, Any]]) -> None:
    client = make_client(provider_user_identifier_key=_KEY)
    headers, workspace = _keyed_user(client)

    _chat(client, headers)

    assert captured[0]["user"] == upstream_user_identifier(workspace, "alice", _KEY)


def test_anthropic_gets_metadata_user_id_and_no_user(
    make_client: ClientFactory, captured: list[dict[str, Any]]
) -> None:
    client = make_client()
    headers, workspace = _keyed_user(client)

    _chat(client, headers, model="anthropic:claude-sonnet-4-5", user="alice")

    assert captured[0]["metadata"] == {"user_id": upstream_user_identifier(workspace, "alice", None)}
    assert "user" not in captured[0]


def test_bedrock_gets_request_metadata(make_client: ClientFactory, captured: list[dict[str, Any]]) -> None:
    client = make_client()
    headers, workspace = _keyed_user(client)

    _chat(client, headers, model="bedrock:anthropic.claude-3-5-sonnet-20241022-v2:0")

    assert captured[0]["requestMetadata"] == {"otari_user": upstream_user_identifier(workspace, "alice", None)}
    assert "user" not in captured[0]


def test_a_provider_with_no_end_user_field_gets_nothing(
    make_client: ClientFactory, captured: list[dict[str, Any]]
) -> None:
    client = make_client()
    headers, _ = _keyed_user(client)

    _chat(client, headers, model="gemini:gemini-2.5-flash", user="alice")

    assert not {"user", "metadata", "requestMetadata", UPSTREAM_USER_FIELD} & set(captured[0])


def test_a_streamed_request_carries_it(make_client: ClientFactory, captured: list[dict[str, Any]]) -> None:
    client = make_client()
    headers, workspace = _keyed_user(client)

    _chat(client, headers, stream=True)

    assert captured[0]["user"] == upstream_user_identifier(workspace, "alice", None)


def test_turned_off_nothing_is_sent(make_client: ClientFactory, captured: list[dict[str, Any]]) -> None:
    client = make_client(provider_user_identifier_enabled=False)
    headers, _ = _keyed_user(client)

    _chat(client, headers, user="alice")

    assert not {"user", UPSTREAM_USER_FIELD} & set(captured[0])


# ---------------------------------------------------------------------------
# Hybrid mode: the identity comes from the platform's resolve response.
# ---------------------------------------------------------------------------


@pytest.fixture
def hybrid_client(monkeypatch: pytest.MonkeyPatch) -> Generator[Callable[..., TestClient]]:
    monkeypatch.setenv("OTARI_AI_TOKEN", "gw_test_token")
    entered: list[TestClient] = []

    def make(**overrides: Any) -> TestClient:
        app = app_for(GatewayConfig(mode="hybrid", platform={"base_url": "http://platform.test/api/v1"}, **overrides))
        client = TestClient(app)
        client.__enter__()
        entered.append(client)
        return client

    yield make
    for client in entered:
        client.__exit__(None, None, None)
    reset_config()
    reset_db()


def _platform(control_plane_transport: InstallControlPlane, **identity: str) -> None:
    async def fake_post_platform(
        url: str, headers: dict[str, str], body: dict[str, Any], timeout_seconds: float
    ) -> httpx.Response:
        if url.endswith("/gateway/provider-keys/resolve"):
            return httpx.Response(
                200,
                json={
                    "request_id": "req-1",
                    "fallback_enabled": False,
                    "attempts": [
                        {
                            "attempt_id": "att-1",
                            "position": 0,
                            "provider": "openai",
                            "model": "gpt-4o-mini",
                            "api_key": "sk-platform",
                            "managed": False,
                        }
                    ],
                    **identity,
                },
            )
        return httpx.Response(204)

    control_plane_transport(fake_post_platform)


@pytest.fixture
def hybrid_captured(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    async def fake_acompletion(**kwargs: Any) -> ChatCompletion:
        calls.append(kwargs)
        return _completion()

    monkeypatch.setattr("gateway.api.routes.chat.acompletion", fake_acompletion)
    return calls


_TOKEN = {"Authorization": "Bearer user_test_token"}


def test_hybrid_digests_the_platform_workspace_and_user(
    hybrid_client: Callable[..., TestClient],
    control_plane_transport: InstallControlPlane,
    hybrid_captured: list[dict[str, Any]],
) -> None:
    _platform(control_plane_transport, workspace_id="ws-1", user_id="platform-user")

    _chat(hybrid_client(), _TOKEN, model="gpt-4o-mini")

    assert hybrid_captured[0]["user"] == upstream_user_identifier("ws-1", "platform-user", None)


def test_hybrid_prefers_the_callers_tag(
    hybrid_client: Callable[..., TestClient],
    control_plane_transport: InstallControlPlane,
    hybrid_captured: list[dict[str, Any]],
) -> None:
    _platform(control_plane_transport, workspace_id="ws-1")

    _chat(hybrid_client(), _TOKEN, model="gpt-4o-mini", user="end-user-7")

    assert hybrid_captured[0]["user"] == upstream_user_identifier("ws-1", "end-user-7", None)


@pytest.mark.parametrize(
    ("identity", "body"),
    [({"user_id": "platform-user"}, {"user": "end-user-7"}), ({"workspace_id": "ws-1"}, {}), ({}, {})],
    ids=["no-workspace", "no-user-no-tag", "neither"],
)
def test_hybrid_refuses_a_request_it_cannot_identify(
    hybrid_client: Callable[..., TestClient],
    control_plane_transport: InstallControlPlane,
    hybrid_captured: list[dict[str, Any]],
    identity: dict[str, str],
    body: dict[str, Any],
) -> None:
    _platform(control_plane_transport, **identity)

    response = hybrid_client().post(
        f"{API_ROOT}/chat/completions",
        json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}], **body},
        headers=_TOKEN,
    )

    assert response.status_code == 502
    assert response.json() == {"detail": NO_UPSTREAM_USER_DETAIL}
    assert hybrid_captured == []


def test_hybrid_turned_off_serves_an_unidentified_request(
    hybrid_client: Callable[..., TestClient],
    control_plane_transport: InstallControlPlane,
    hybrid_captured: list[dict[str, Any]],
) -> None:
    _platform(control_plane_transport)

    _chat(hybrid_client(provider_user_identifier_enabled=False), _TOKEN, model="gpt-4o-mini")

    assert not {"user", UPSTREAM_USER_FIELD} & set(hybrid_captured[0])


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------


@pytest.fixture
def responses_captured(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """What a Responses request handed any-llm, natively or through the chat bridge."""
    calls: list[dict[str, Any]] = []

    async def fake(**kwargs: Any) -> None:
        calls.append(kwargs)
        raise _Stop

    monkeypatch.setattr("gateway.api.routes.responses.aresponses", fake)
    monkeypatch.setattr("gateway.services.inference._responses_bridge.acompletion", fake)
    return calls


def _respond(client: TestClient, headers: dict[str, str], model: str = "openai:gpt-4o-mini", **body: Any) -> None:
    client.post(f"{API_ROOT}/responses", json={"model": model, "input": "hi", **body}, headers=headers)


def test_responses_sends_openai_a_safety_identifier_and_no_user(
    make_client: ClientFactory, responses_captured: list[dict[str, Any]]
) -> None:
    client = make_client()
    headers, workspace = _keyed_user(client)

    _respond(client, headers, user="alice", safety_identifier="caller-chosen")

    assert responses_captured[0]["safety_identifier"] == upstream_user_identifier(workspace, "alice", None)
    assert not {"user", UPSTREAM_USER_FIELD} & set(responses_captured[0])


def test_responses_to_anthropic_sends_metadata_user_id_and_never_user(
    make_client: ClientFactory, responses_captured: list[dict[str, Any]]
) -> None:
    """Anthropic has no Responses API, so the bridge serves it as a chat completion, where ``user`` fails the call."""
    client = make_client()
    headers, workspace = _keyed_user(client)

    _respond(client, headers, model="anthropic:claude-sonnet-4-5")

    assert responses_captured[0]["metadata"] == {"user_id": upstream_user_identifier(workspace, "alice", None)}
    assert not {"user", "safety_identifier"} & set(responses_captured[0])


def test_responses_turned_off_sends_no_identifier_at_all(
    make_client: ClientFactory, responses_captured: list[dict[str, Any]]
) -> None:
    client = make_client(provider_user_identifier_enabled=False)
    headers, _ = _keyed_user(client)

    _respond(client, headers, user="alice", safety_identifier="caller-chosen")

    assert not {"user", "safety_identifier", UPSTREAM_USER_FIELD} & set(responses_captured[0])


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------


@pytest.fixture
def messages_captured(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    async def fake_amessages(**kwargs: Any) -> None:
        calls.append(kwargs)
        raise _Stop

    monkeypatch.setattr("gateway.api.routes.messages.amessages", fake_amessages)
    return calls


def _message(client: TestClient, headers: dict[str, str], **body: Any) -> None:
    client.post(
        f"{API_ROOT}/messages",
        json={
            "model": "anthropic:claude-sonnet-4-5",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hi"}],
            **body,
        },
        headers=headers,
    )


def test_messages_replaces_the_callers_user_id_with_the_identifier(
    make_client: ClientFactory, messages_captured: list[dict[str, Any]]
) -> None:
    client = make_client(reject_user_mismatch=False)
    headers, workspace = _keyed_user(client)

    _message(client, headers, metadata={"user_id": "end-user-7"})

    assert messages_captured[0]["metadata"] == {"user_id": upstream_user_identifier(workspace, "end-user-7", None)}
    assert UPSTREAM_USER_FIELD not in messages_captured[0]


def test_messages_without_a_tag_digests_the_resolved_user(
    make_client: ClientFactory, messages_captured: list[dict[str, Any]]
) -> None:
    client = make_client()
    headers, workspace = _keyed_user(client)

    _message(client, headers)

    assert messages_captured[0]["metadata"] == {"user_id": upstream_user_identifier(workspace, "alice", None)}


def test_messages_turned_off_drops_the_callers_user_id(
    make_client: ClientFactory, messages_captured: list[dict[str, Any]]
) -> None:
    client = make_client(provider_user_identifier_enabled=False)
    headers, _ = _keyed_user(client)

    _message(client, headers, metadata={"user_id": "alice"})

    assert not {"metadata", UPSTREAM_USER_FIELD} & set(messages_captured[0])


def test_messages_served_as_a_chat_completion_gets_the_providers_chat_field(
    make_client: ClientFactory, messages_captured: list[dict[str, Any]]
) -> None:
    client = make_client()
    headers, workspace = _keyed_user(client)

    _message(client, headers, model="openai:gpt-4o-mini")

    assert messages_captured[0]["user"] == upstream_user_identifier(workspace, "alice", None)
    assert not {"metadata", UPSTREAM_USER_FIELD} & set(messages_captured[0])
