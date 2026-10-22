"""The end-user identifier a completion sends its provider."""

import time
from collections.abc import AsyncIterator, Callable, Generator
from typing import Any

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
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from gateway.core.config import API_KEY_HEADER, API_ROOT, GatewayConfig
from gateway.models.usage import UsageLog
from gateway.services.inference import UPSTREAM_USER_FIELD, upstream_user_identifier

from .conftest import build_test_client

_MASTER_KEY = "test-master-key"
_MASTER = {API_KEY_HEADER: f"Bearer {_MASTER_KEY}"}
_KEY = "k" * 32
_PROVIDERS = {
    "openai": {"api_key": "sk-openai"},
    "azureopenai": {"api_key": "sk-azure", "api_base": "https://example.openai.azure.com"},
    "anthropic": {"api_key": "sk-anthropic"},
    "gemini": {"api_key": "gemini-key"},
    "meta": {"api_key": "meta-key"},
    "mistral": {"api_key": "mistral-key"},
    "ollama": {"api_base": "http://ollama:11434"},
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


def _keyed_user(client: TestClient, user_id: str = "alice", *, create_user: bool = True) -> tuple[dict[str, str], str]:
    """A key for ``user_id``, and the workspace it bills to."""
    if create_user:
        assert client.post(f"{API_ROOT}/users", json={"user_id": user_id}, headers=_MASTER).status_code == 200
    created = client.post(f"{API_ROOT}/keys", json={"key_name": "k", "user_id": user_id}, headers=_MASTER)
    assert created.status_code == 200
    key = created.json()
    workspace = client.get(f"{API_ROOT}/keys/{key['id']}", headers=_MASTER).json()["workspace_id"]
    return {API_KEY_HEADER: f"Bearer {key['key']}"}, workspace


def _chat(client: TestClient, headers: dict[str, str], model: str = "openai:gpt-4o-mini", **body: Any) -> None:
    client.post(
        f"{API_ROOT}/chat/completions",
        json={"model": model, "messages": [{"role": "user", "content": "hi"}]} | body,
        headers=headers,
    )


def test_openai_gets_a_digest_of_the_workspace_and_resolved_user(
    make_client: ClientFactory, captured: list[dict[str, Any]]
) -> None:
    client = make_client()
    headers, workspace = _keyed_user(client)

    _chat(client, headers)

    assert captured[0]["safety_identifier"] == upstream_user_identifier(workspace, "alice", None)
    assert not {"user", UPSTREAM_USER_FIELD} & set(captured[0])


def test_azure_openai_gets_a_safety_identifier(make_client: ClientFactory, captured: list[dict[str, Any]]) -> None:
    client = make_client()
    headers, workspace = _keyed_user(client)

    _chat(client, headers, model="azureopenai:gpt-4o-mini")

    assert captured[0]["safety_identifier"] == upstream_user_identifier(workspace, "alice", None)
    assert "user" not in captured[0]


def test_a_user_tag_replaces_the_resolved_user(make_client: ClientFactory, captured: list[dict[str, Any]]) -> None:
    client = make_client(reject_user_mismatch=False)
    headers, workspace = _keyed_user(client)

    _chat(client, headers, user="end-user-7")

    assert captured[0]["safety_identifier"] == upstream_user_identifier(workspace, "end-user-7", None)


def test_a_safety_identifier_is_the_tag(make_client: ClientFactory, captured: list[dict[str, Any]]) -> None:
    client = make_client()
    headers, workspace = _keyed_user(client)

    _chat(client, headers, safety_identifier="a")
    _chat(client, headers, safety_identifier="b")

    assert captured[0]["safety_identifier"] == upstream_user_identifier(workspace, "a", None)
    assert captured[1]["safety_identifier"] == upstream_user_identifier(workspace, "b", None)


def test_a_safety_identifier_wins_over_user(make_client: ClientFactory, captured: list[dict[str, Any]]) -> None:
    client = make_client()
    headers, workspace = _keyed_user(client)

    _chat(client, headers, user="alice", safety_identifier="end-user-7")

    assert captured[0]["safety_identifier"] == upstream_user_identifier(workspace, "end-user-7", None)


def test_the_callers_safety_identifier_never_reaches_the_provider(
    make_client: ClientFactory, captured: list[dict[str, Any]]
) -> None:
    client = make_client()
    headers, workspace = _keyed_user(client)

    _chat(client, headers, model="anthropic:claude-sonnet-4-5", safety_identifier="end-user-7")

    assert captured[0]["metadata"] == {"user_id": upstream_user_identifier(workspace, "end-user-7", None)}
    assert "safety_identifier" not in captured[0]


def test_echoing_your_own_user_gives_the_same_identifier(
    make_client: ClientFactory, captured: list[dict[str, Any]]
) -> None:
    client = make_client()
    headers, _ = _keyed_user(client)

    _chat(client, headers)
    _chat(client, headers, user="alice")

    assert captured[0]["safety_identifier"] == captured[1]["safety_identifier"]


def test_the_master_key_user_is_digested_too(make_client: ClientFactory, captured: list[dict[str, Any]]) -> None:
    client = make_client()
    headers, _ = _keyed_user(client)

    _chat(client, headers)
    _chat(client, _MASTER, user="alice")

    assert captured[1]["safety_identifier"] == captured[0]["safety_identifier"]
    assert captured[1]["safety_identifier"] != "alice"


def test_a_key_makes_the_digest_keyed(make_client: ClientFactory, captured: list[dict[str, Any]]) -> None:
    client = make_client(provider_user_identifier_key=_KEY)
    headers, workspace = _keyed_user(client)

    _chat(client, headers)

    assert captured[0]["safety_identifier"] == upstream_user_identifier(workspace, "alice", _KEY)


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


def test_gemini_gets_a_safety_identifier_label(make_client: ClientFactory, captured: list[dict[str, Any]]) -> None:
    client = make_client()
    headers, workspace = _keyed_user(client)

    _chat(client, headers, model="gemini:gemini-2.5-flash", user="alice")

    identifier = upstream_user_identifier(workspace, "alice", None)
    assert captured[0]["http_options"] == {"extra_body": {"labels": {"safety_identifier": identifier[:63]}}}
    assert "user" not in captured[0]


def test_meta_gets_a_safety_identifier(make_client: ClientFactory, captured: list[dict[str, Any]]) -> None:
    client = make_client()
    headers, workspace = _keyed_user(client)

    _chat(client, headers, model="meta:Llama-4-Maverick-17B-128E-Instruct-FP8", user="alice")

    assert captured[0]["safety_identifier"] == upstream_user_identifier(workspace, "alice", None)
    assert "user" not in captured[0]


def test_a_provider_with_no_end_user_field_gets_nothing(
    make_client: ClientFactory, captured: list[dict[str, Any]]
) -> None:
    client = make_client()
    headers, _ = _keyed_user(client)

    _chat(client, headers, model="mistral:mistral-small-latest", user="alice")

    assert not {"user", "metadata", "requestMetadata", "http_options", UPSTREAM_USER_FIELD} & set(captured[0])


def test_a_streamed_request_carries_it(make_client: ClientFactory, captured: list[dict[str, Any]]) -> None:
    client = make_client()
    headers, workspace = _keyed_user(client)

    _chat(client, headers, stream=True)

    assert captured[0]["safety_identifier"] == upstream_user_identifier(workspace, "alice", None)


_IMAGE = "data:image/png;base64,iVBORw0KGgo="


@pytest.mark.parametrize(
    ("vision_model", "fields"),
    [
        ("openai:gpt-4o-mini", lambda identifier: {"safety_identifier": identifier}),
        ("anthropic:claude-sonnet-4-5", lambda identifier: {"metadata": {"user_id": identifier}}),
    ],
    ids=["openai", "anthropic"],
)
def test_the_image_describe_call_carries_the_requests_identifier(
    make_client: ClientFactory,
    captured: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    vision_model: str,
    fields: Callable[[str], dict[str, Any]],
) -> None:
    """A text-only model's image is captioned by the vision model, which gets the main call's identifier."""
    described: list[dict[str, Any]] = []

    async def fake_describe(**kwargs: Any) -> ChatCompletion:
        described.append(kwargs)
        return _completion()

    monkeypatch.setattr("gateway.services.vision.acompletion", fake_describe)
    client = make_client(vision_strategy="describe", vision_describe_model=vision_model)
    headers, workspace = _keyed_user(client)

    _chat(
        client,
        headers,
        model="ollama:llama3",
        safety_identifier="end-user-7",
        messages=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": _IMAGE}}]}],
    )

    assert len(captured) == 1
    identifier = upstream_user_identifier(workspace, "end-user-7", None)
    end_user_fields = {"user", "safety_identifier", "metadata", UPSTREAM_USER_FIELD}
    assert {key: value for key, value in described[0].items() if key in end_user_fields} == fields(identifier)


# ---------------------------------------------------------------------------
# Usage rows
# ---------------------------------------------------------------------------


@pytest.fixture
def served(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every chat completion succeeds, streamed or not."""

    async def fake_acompletion(**kwargs: Any) -> Any:
        if not kwargs.get("stream"):
            return _completion()

        async def chunks() -> AsyncIterator[ChatCompletionChunk]:
            yield ChatCompletionChunk(
                id="chunk-1",
                object="chat.completion.chunk",
                created=0,
                model="gpt-4o-mini",
                choices=[ChunkChoice(index=0, delta=ChoiceDelta(role="assistant", content="hi"), finish_reason="stop")],
                usage=CompletionUsage(prompt_tokens=4, completion_tokens=2, total_tokens=6),
            )

        return chunks()

    monkeypatch.setattr("gateway.api.routes.chat.acompletion", fake_acompletion)


def _logged_identifiers(postgres_url: str, count: int, *, timeout: float = 3.0) -> list[str | None]:
    """The ``safety_identifier`` of each usage row, once ``count`` rows are written."""
    engine = create_engine(postgres_url)
    deadline = time.monotonic() + timeout
    try:
        while True:
            with Session(engine) as db:
                identifiers = list(db.scalars(select(UsageLog.safety_identifier)).all())
            if len(identifiers) >= count:
                return identifiers
            assert time.monotonic() < deadline, f"expected {count} usage rows, found {len(identifiers)}"
            time.sleep(0.1)
    finally:
        engine.dispose()


@pytest.mark.parametrize("stream", [False, True], ids=["non-streaming", "streaming"])
@pytest.mark.usefixtures("served")
def test_the_usage_row_stores_the_identifier_the_provider_got(
    make_client: ClientFactory, postgres_url: str, stream: bool
) -> None:
    client = make_client()
    headers, workspace = _keyed_user(client)

    _chat(client, headers, safety_identifier="end-user-7", stream=stream)

    assert _logged_identifiers(postgres_url, 1) == [upstream_user_identifier(workspace, "end-user-7", None)]


@pytest.mark.usefixtures("captured")
def test_a_failed_calls_usage_row_stores_the_identifier(make_client: ClientFactory, postgres_url: str) -> None:
    client = make_client()
    headers, workspace = _keyed_user(client)

    _chat(client, headers)

    assert _logged_identifiers(postgres_url, 1) == [upstream_user_identifier(workspace, "alice", None)]


@pytest.mark.usefixtures("captured")
def test_a_refusal_after_the_user_is_resolved_stores_the_identifier(
    make_client: ClientFactory, postgres_url: str
) -> None:
    client = make_client()
    created = client.post(
        f"{API_ROOT}/users", json={"user_id": "alice", "allowed_models": ["openai:gpt-4o"]}, headers=_MASTER
    )
    assert created.status_code == 200
    headers, workspace = _keyed_user(client, create_user=False)

    _chat(client, headers, safety_identifier="end-user-7")

    assert _logged_identifiers(postgres_url, 1) == [upstream_user_identifier(workspace, "end-user-7", None)]


@pytest.mark.usefixtures("served")
def test_the_image_describe_row_stores_the_requests_identifier(
    make_client: ClientFactory, postgres_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_describe(**_kwargs: Any) -> ChatCompletion:
        return _completion()

    monkeypatch.setattr("gateway.services.vision.acompletion", fake_describe)
    client = make_client(vision_strategy="describe", vision_describe_model="openai:gpt-4o-mini")
    headers, workspace = _keyed_user(client)

    _chat(
        client,
        headers,
        model="ollama:llama3",
        safety_identifier="end-user-7",
        messages=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": _IMAGE}}]}],
    )

    identifier = upstream_user_identifier(workspace, "end-user-7", None)
    assert _logged_identifiers(postgres_url, 2) == [identifier, identifier]


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

    _respond(client, headers, user="alice")

    assert responses_captured[0]["safety_identifier"] == upstream_user_identifier(workspace, "alice", None)
    assert not {"user", UPSTREAM_USER_FIELD} & set(responses_captured[0])


def test_responses_digests_the_callers_safety_identifier(
    make_client: ClientFactory, responses_captured: list[dict[str, Any]]
) -> None:
    client = make_client()
    headers, workspace = _keyed_user(client)

    _respond(client, headers, user="alice", safety_identifier="a")
    _respond(client, headers, user="alice", safety_identifier="b")

    assert responses_captured[0]["safety_identifier"] == upstream_user_identifier(workspace, "a", None)
    assert responses_captured[1]["safety_identifier"] == upstream_user_identifier(workspace, "b", None)


def test_responses_to_anthropic_sends_metadata_user_id_and_never_user(
    make_client: ClientFactory, responses_captured: list[dict[str, Any]]
) -> None:
    """Anthropic has no Responses API, so the bridge serves it as a chat completion, where ``user`` fails the call."""
    client = make_client()
    headers, workspace = _keyed_user(client)

    _respond(client, headers, model="anthropic:claude-sonnet-4-5")

    assert responses_captured[0]["metadata"] == {"user_id": upstream_user_identifier(workspace, "alice", None)}
    assert not {"user", "safety_identifier"} & set(responses_captured[0])


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


def test_messages_served_as_a_chat_completion_gets_the_providers_chat_field(
    make_client: ClientFactory, messages_captured: list[dict[str, Any]]
) -> None:
    client = make_client()
    headers, workspace = _keyed_user(client)

    _message(client, headers, model="openai:gpt-4o-mini")

    assert messages_captured[0]["safety_identifier"] == upstream_user_identifier(workspace, "alice", None)
    assert not {"metadata", UPSTREAM_USER_FIELD} & set(messages_captured[0])
