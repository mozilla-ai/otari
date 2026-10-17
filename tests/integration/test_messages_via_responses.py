"""``/api/v1/messages`` served through the Responses API of an instance that sets ``messages_api: responses``."""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator, Callable
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from gateway.core.config import API_ROOT, CONVERSATION_HEADER, GatewayConfig
from gateway.models.usage import UsageLog
from gateway.services.inference import reasoning_signature

from .conftest import declared_test_providers, shared_test_config

_ARESPONSES = "gateway.services.inference._messages_responses_bridge.aresponses"
_AMESSAGES = "gateway.api.routes.messages.amessages"
_USAGE = {
    "input_tokens": 100,
    "input_tokens_details": {"cached_tokens": 40},
    "output_tokens": 30,
    "output_tokens_details": {"reasoning_tokens": 12},
    "total_tokens": 130,
}
_REASONING = {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "enc-1"}
_MESSAGE = {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Hello"}]}


@pytest.fixture
def test_config(postgres_url: str) -> GatewayConfig:
    providers = {**declared_test_providers(), "openai": {"api_key": "sk-test", "messages_api": "responses"}}
    return shared_test_config(postgres_url, providers)


def _response() -> dict[str, Any]:
    return {"id": "resp_1", "model": "gpt-5", "status": "completed", "output": [_REASONING, _MESSAGE], "usage": _USAGE}


def _body(user_id: str, **fields: Any) -> dict[str, Any]:
    return {
        "model": "openai:gpt-5",
        "max_tokens": 256,
        "messages": [{"role": "user", "content": "hi"}],
        "metadata": {"user_id": user_id},
        **fields,
    }


def _usage_row(make_session: Callable[[], Session], user_id: str, *, timeout: float = 3.0) -> UsageLog | None:
    deadline = time.time() + timeout
    while True:
        db = make_session()
        try:
            row = db.query(UsageLog).filter(UsageLog.user_id == user_id).first()
            if row is not None or time.time() > deadline:
                return row
        finally:
            db.close()
        time.sleep(0.1)


def _create_user(client: TestClient, headers: dict[str, str], user_id: str) -> None:
    assert client.post(f"{API_ROOT}/users", json={"user_id": user_id}, headers=headers).status_code == 200


async def _no_native_call(**_: Any) -> Any:
    raise AssertionError("an opted-in instance must not reach amessages")


def test_messages_are_served_by_the_responses_api(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent: list[dict[str, Any]] = []

    async def fake_aresponses(**kwargs: Any) -> dict[str, Any]:
        sent.append(kwargs)
        return _response()

    monkeypatch.setattr(_ARESPONSES, fake_aresponses)
    monkeypatch.setattr(_AMESSAGES, _no_native_call)
    _create_user(client, master_key_header, "rsp-user")

    previous_turn = [
        {"type": "thinking", "thinking": "", "signature": reasoning_signature("openai", "enc-0")},
        {"type": "text", "text": "Earlier"},
    ]
    response = client.post(
        f"{API_ROOT}/messages",
        json=_body(
            "rsp-user",
            messages=[
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": previous_turn},
                {"role": "user", "content": "again"},
            ],
        ),
        headers=master_key_header,
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["content"] == [
        {"type": "thinking", "thinking": "", "signature": reasoning_signature("openai", "enc-1")},
        {"type": "text", "text": "Hello"},
    ]
    assert body["stop_reason"] == "end_turn"
    assert sent[0]["store"] is False
    assert sent[0]["max_output_tokens"] == 256
    assert sent[0]["input_data"][1] == {"type": "reasoning", "summary": [], "encrypted_content": "enc-0"}
    assert "_otari_responses_instance" not in sent[0]

    row = _usage_row(db_session_factory, "rsp-user")
    assert row is not None
    assert row.status == "success"
    assert row.provider == "openai"
    assert row.completion_tokens == 30


def test_streamed_messages_are_served_by_the_responses_api(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[dict[str, Any]] = [
        {"type": "response.created", "response": {"id": "resp_1", "model": "gpt-5"}},
        {"type": "response.output_item.added", "output_index": 0, "item": {"type": "message"}},
        {"type": "response.output_text.delta", "output_index": 0, "content_index": 0, "delta": "Hello"},
        {"type": "response.output_item.done", "output_index": 0, "item": _MESSAGE},
        {"type": "response.completed", "response": _response()},
    ]

    async def fake_aresponses(**kwargs: Any) -> AsyncIterator[dict[str, Any]]:
        assert kwargs["stream"] is True

        async def gen() -> AsyncIterator[dict[str, Any]]:
            for event in events:
                yield event

        return gen()

    monkeypatch.setattr(_ARESPONSES, fake_aresponses)
    monkeypatch.setattr(_AMESSAGES, _no_native_call)
    _create_user(client, master_key_header, "rsp-stream-user")

    response = client.post(
        f"{API_ROOT}/messages", json=_body("rsp-stream-user", stream=True), headers=master_key_header
    )

    assert response.status_code == 200, response.text
    frames = [json.loads(line[len("data: ") :]) for line in response.text.splitlines() if line.startswith("data: ")]
    # The gateway's own closing ``done`` frame carries no type.
    data = [frame for frame in frames if "type" in frame]
    assert [event["type"] for event in data] == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert data[-2]["usage"]["cache_read_input_tokens"] == 40

    row = _usage_row(db_session_factory, "rsp-stream-user")
    assert row is not None
    assert row.status == "success"
    assert row.completion_tokens == 30


def test_an_instance_that_did_not_opt_in_keeps_the_messages_call(
    client: TestClient,
    master_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent: list[dict[str, Any]] = []

    async def fake_amessages(**kwargs: Any) -> Any:
        sent.append(kwargs)
        raise RuntimeError("stop")

    async def no_responses_call(**_: Any) -> Any:
        raise AssertionError("an instance that did not opt in must not reach aresponses")

    monkeypatch.setattr(_AMESSAGES, fake_amessages)
    monkeypatch.setattr(_ARESPONSES, no_responses_call)
    _create_user(client, master_key_header, "native-user")

    signed_here = {"type": "thinking", "thinking": "", "signature": reasoning_signature("openai", "enc-0")}
    client.post(
        f"{API_ROOT}/messages",
        json=_body(
            "native-user",
            model="anthropic:claude-sonnet-4-5",
            messages=[
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": [signed_here, {"type": "text", "text": "Earlier"}]},
                {"role": "user", "content": "again"},
            ],
        ),
        headers=master_key_header,
    )

    assert sent, "the native Messages call must run"
    assert sent[0]["messages"][1]["content"] == [{"type": "text", "text": "Earlier"}]
    assert "_otari_responses_instance" not in sent[0]


def test_the_conversation_id_keys_the_prompt_cache(
    client: TestClient,
    master_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent: list[dict[str, Any]] = []

    async def fake_aresponses(**kwargs: Any) -> dict[str, Any]:
        sent.append(kwargs)
        return _response()

    monkeypatch.setattr(_ARESPONSES, fake_aresponses)
    monkeypatch.setattr(_AMESSAGES, _no_native_call)
    _create_user(client, master_key_header, "cache-user")

    for conversation in ("conv-a", "conv-a", "conv-b"):
        response = client.post(
            f"{API_ROOT}/messages",
            json=_body("cache-user"),
            headers={**master_key_header, CONVERSATION_HEADER: conversation},
        )
        assert response.status_code == 200, response.text
    harness = client.post(
        f"{API_ROOT}/messages",
        json=_body("cache-user"),
        headers={**master_key_header, "x-claude-code-session-id": "conv-a"},
    )
    assert harness.status_code == 200, harness.text
    explicit = client.post(
        f"{API_ROOT}/messages",
        json=_body("cache-user", prompt_cache_key="mine"),
        headers={**master_key_header, CONVERSATION_HEADER: "conv-a"},
    )
    assert explicit.status_code == 200, explicit.text

    keys = [call["prompt_cache_key"] for call in sent]
    assert keys[0] == keys[1]
    assert len(set(keys)) == 4
    # Scoped like any caller key: the raw conversation id never reaches the provider.
    assert not {"conv-a", "conv-b", "mine"} & set(keys)
    assert all("_otari_conversation_cache_key" not in call for call in sent)
