"""The end-user identifier sent to providers, and the field each provider reads it from."""

import hashlib
import hmac
import json
from typing import Any, cast
from unittest.mock import patch

import pytest
from any_llm import LLMProvider, acompletion

from gateway.api.routes import chat
from gateway.services.inference import (
    UPSTREAM_USER_FIELD,
    UnidentifiedEndUserError,
    chat_user_fields,
    end_user_identifier,
    messages_user_fields,
    responses_user_fields,
    upstream_user_identifier,
)

_KEY = "k" * 32

# Every any-llm provider, classified. A provider any-llm adds fails
# ``test_every_provider_is_classified`` until it is placed here, because sending
# ``user`` to an API that does not take it fails the call.
_FIELD_BY_PROVIDER: dict[LLMProvider, str | None] = {
    # Their chat APIs document ``user`` as an end-user id for abuse monitoring.
    LLMProvider.OPENAI: "user",
    LLMProvider.AZUREOPENAI: "user",
    LLMProvider.CEREBRAS: "user",
    LLMProvider.FIREWORKS: "user",
    LLMProvider.GROQ: "user",
    LLMProvider.OPENROUTER: "user",
    LLMProvider.XAI: "user",
    # any-llm sends it as DeepSeek's own ``user_id``.
    LLMProvider.DEEPSEEK: "user",
    LLMProvider.ANTHROPIC: "metadata",
    LLMProvider.AZUREANTHROPIC: "metadata",
    LLMProvider.VERTEXAIANTHROPIC: "metadata",
    LLMProvider.BEDROCK: "requestMetadata",
    # The SDK any-llm calls has no ``user`` argument, so the call would fail.
    LLMProvider.COHERE: None,
    LLMProvider.HUGGINGFACE: None,
    LLMProvider.OLLAMA: None,
    LLMProvider.TOGETHER: None,
    # No end-user field on the provider's API.
    LLMProvider.GEMINI: None,
    LLMProvider.LINER: None,
    LLMProvider.MISTRAL: None,
    LLMProvider.VERTEXAI: None,
    # The operator's own model server: nobody upstream enforces a policy.
    LLMProvider.LLAMACPP: None,
    LLMProvider.LLAMAFILE: None,
    LLMProvider.LMSTUDIO: None,
    LLMProvider.SAGEMAKER: None,
    LLMProvider.VLLM: None,
    # Not checked: whether the API accepts ``user`` is unknown.
    LLMProvider.ATLASCLOUD: None,
    LLMProvider.AZURE: None,
    LLMProvider.CASCADIA: None,
    LLMProvider.DASHSCOPE: None,
    LLMProvider.DATABRICKS: None,
    LLMProvider.DEEPINFRA: None,
    LLMProvider.EDENAI: None,
    LLMProvider.GITHUB: None,
    LLMProvider.GMI: None,
    LLMProvider.INCEPTION: None,
    LLMProvider.KENARI: None,
    LLMProvider.LLAMA: None,
    LLMProvider.META: None,
    LLMProvider.MINIMAX: None,
    LLMProvider.MOONSHOT: None,
    LLMProvider.MZAI: None,
    LLMProvider.NEBIUS: None,
    LLMProvider.NEOSANTARA: None,
    LLMProvider.PERPLEXITY: None,
    LLMProvider.PORTKEY: None,
    LLMProvider.QINIU: None,
    LLMProvider.REQUESTY: None,
    LLMProvider.SAMBANOVA: None,
    LLMProvider.TELNYX: None,
    LLMProvider.WATSONX: None,
    LLMProvider.ZAI: None,
    # Another Otari reads ``user`` as the billed user, and a strict key refuses one that is not its own.
    LLMProvider.OTARI: None,
    # Embeddings only.
    LLMProvider.VOYAGE: None,
    # No chat or Messages API.
    LLMProvider.TYPESAFE: None,
}


def test_the_identifier_is_a_keyed_digest_of_workspace_and_subject() -> None:
    payload = json.dumps(["otari-upstream-user-v1", "ws", "alice"], separators=(",", ":")).encode()

    assert upstream_user_identifier("ws", "alice", _KEY) == hmac.new(_KEY.encode(), payload, hashlib.sha256).hexdigest()


def test_without_a_key_the_identifier_is_an_unkeyed_digest() -> None:
    payload = json.dumps(["otari-upstream-user-v1", "ws", "alice"], separators=(",", ":")).encode()

    assert upstream_user_identifier("ws", "alice", None) == hashlib.sha256(payload).hexdigest()


def test_the_identifier_is_stable_and_never_the_raw_value() -> None:
    identifier = upstream_user_identifier("ws", "alice@example.com", _KEY)

    assert identifier == upstream_user_identifier("ws", "alice@example.com", _KEY)
    assert "alice" not in identifier
    assert len(identifier) == 64


@pytest.mark.parametrize(
    ("first", "second"),
    [
        (("ws", "alice", _KEY), ("ws", "bob", _KEY)),
        (("ws-1", "alice", _KEY), ("ws-2", "alice", _KEY)),
        (("ws", "alice", _KEY), ("ws", "alice", "j" * 32)),
        (("ws", "alice", _KEY), ("ws", "alice", None)),
    ],
)
def test_the_identifier_changes_with_each_input(
    first: tuple[str, str, str | None], second: tuple[str, str, str | None]
) -> None:
    assert upstream_user_identifier(*first) != upstream_user_identifier(*second)


def test_every_provider_is_classified() -> None:
    assert set(_FIELD_BY_PROVIDER) == set(LLMProvider)


@pytest.mark.parametrize(("provider", "field"), list(_FIELD_BY_PROVIDER.items()))
def test_each_provider_gets_its_own_field(provider: LLMProvider, field: str | None) -> None:
    expected = {
        "user": {"user": "id"},
        "metadata": {"metadata": {"user_id": "id"}},
        "requestMetadata": {"requestMetadata": {"otari_user": "id"}},
        None: {},
    }[field]

    assert chat_user_fields(provider, "id") == expected
    assert chat_user_fields(provider.value, "id") == expected


def test_an_unknown_provider_gets_nothing() -> None:
    assert chat_user_fields("not-a-provider", "id") == {}
    assert chat_user_fields(None, "id") == {}
    assert responses_user_fields("not-a-provider", "id") == {}
    assert responses_user_fields(None, "id") == {}


@pytest.mark.parametrize(
    ("provider", "expected"),
    [
        # A native Responses API that documents it.
        (LLMProvider.OPENAI, {"safety_identifier": "id"}),
        (LLMProvider.AZUREOPENAI, {"safety_identifier": "id"}),
        # Served as a chat completion, where the bridge puts it in the chat field.
        (LLMProvider.ANTHROPIC, {"safety_identifier": "id"}),
        (LLMProvider.BEDROCK, {"safety_identifier": "id"}),
        (LLMProvider.MISTRAL, {"safety_identifier": "id"}),
        # A native Responses API not known to take it.
        (LLMProvider.GROQ, {}),
        (LLMProvider.GEMINI, {}),
        (LLMProvider.OPENROUTER, {}),
    ],
)
def test_a_responses_call_carries_a_safety_identifier_where_it_is_read(
    provider: LLMProvider, expected: dict[str, Any]
) -> None:
    assert responses_user_fields(provider, "id") == expected


@pytest.mark.asyncio
async def test_the_chat_tool_loop_gets_the_providers_field(monkeypatch: pytest.MonkeyPatch) -> None:
    received: list[dict[str, Any]] = []

    async def fake_loop(*, completion_kwargs: dict[str, Any], **_: Any) -> None:
        received.append(completion_kwargs)

    monkeypatch.setattr(chat, "mcp_tool_loop", fake_loop)
    kwargs = {"model": "anthropic:claude-sonnet-4-5", UPSTREAM_USER_FIELD: "id"}

    await chat._ADAPTER.run_tool_loop(kwargs, cast(Any, None), 3)

    assert received == [{"model": "anthropic:claude-sonnet-4-5", "metadata": {"user_id": "id"}}]


def test_chat_kwargs_without_an_identifier_pass_through() -> None:
    kwargs = {"model": "openai:gpt-4o-mini"}

    assert chat._with_upstream_user(kwargs) is kwargs


class _Stop(Exception):
    """Ends a provider call once the SDK has been handed its arguments."""


@pytest.mark.asyncio
async def test_the_anthropic_sdk_receives_metadata_user_id() -> None:
    captured: dict[str, Any] = {}

    async def fake_create(*_: Any, **kwargs: Any) -> None:
        captured.update(kwargs)
        raise _Stop

    with patch("anthropic.resources.messages.AsyncMessages.create", new=fake_create), pytest.raises(_Stop):
        await acompletion(
            model="anthropic:claude-sonnet-4-5",
            messages=[{"role": "user", "content": "hi"}],
            api_key="sk-ant",
            **chat_user_fields(LLMProvider.ANTHROPIC, "id"),
        )

    assert captured["metadata"] == {"user_id": "id"}
    assert "user" not in captured


@pytest.mark.asyncio
async def test_the_bedrock_converse_call_receives_request_metadata() -> None:
    captured: dict[str, Any] = {}

    class _Client:
        def converse(self, **kwargs: Any) -> None:
            captured.update(kwargs)
            raise _Stop

    with pytest.raises(_Stop):
        await acompletion(
            model="bedrock:anthropic.claude-3-5-sonnet-20241022-v2:0",
            messages=[{"role": "user", "content": "hi"}],
            client_args={"client": _Client(), "region_name": "us-east-1"},
            **chat_user_fields(LLMProvider.BEDROCK, "id"),
        )

    assert captured["requestMetadata"] == {"otari_user": "id"}
    assert "user" not in captured


def _identifier(**overrides: Any) -> str | None:
    inputs: dict[str, Any] = {"enabled": True, "key": None, "workspace_id": "ws", "tag": None, "user_id": "alice"}
    return end_user_identifier(**{**inputs, **overrides})


def test_the_tag_wins_over_the_resolved_user() -> None:
    assert _identifier(tag="end-user-7") == upstream_user_identifier("ws", "end-user-7", None)
    assert _identifier() == upstream_user_identifier("ws", "alice", None)


def test_a_session_s_tag_is_ignored() -> None:
    assert _identifier(tag="end-user-7", session=True) == upstream_user_identifier("ws", "alice", None)


def test_turned_off_there_is_no_identifier_and_no_refusal() -> None:
    assert _identifier(enabled=False, workspace_id=None, user_id=None) is None


@pytest.mark.parametrize(
    "missing",
    [{"workspace_id": None}, {"user_id": None}, {"workspace_id": None, "tag": "end-user-7"}],
    ids=["no-workspace", "no-user-no-tag", "tag-but-no-workspace"],
)
def test_turned_on_a_request_with_nothing_to_derive_it_from_is_refused(missing: dict[str, Any]) -> None:
    with pytest.raises(UnidentifiedEndUserError):
        _identifier(**missing)


@pytest.mark.parametrize(
    ("provider", "native", "expected"),
    [
        (LLMProvider.ANTHROPIC, True, {"metadata": {"user_id": "id"}}),
        (LLMProvider.OPENAI, False, {"user": "id"}),
        (LLMProvider.BEDROCK, False, {"requestMetadata": {"otari_user": "id"}}),
        (LLMProvider.GEMINI, False, {}),
    ],
)
def test_messages_use_metadata_natively_and_the_chat_field_otherwise(
    provider: LLMProvider, native: bool, expected: dict[str, Any]
) -> None:
    assert messages_user_fields(provider, "id", native=native) == expected
