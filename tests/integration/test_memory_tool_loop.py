"""Agent memory as tools the gateway runs in a request's tool loop: asked for, offered unasked, or declined."""

import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, cast
from unittest.mock import patch

from any_llm.types.completion import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageFunctionToolCall,
    Choice,
    CompletionUsage,
    Function,
)
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from gateway.core.config import API_ROOT
from gateway.models.tenancy import Workspace
from gateway.services.memory._tools import UNTRUSTED_NOTICE

from .conftest import MODEL_NAME
from .test_memory_mcp import DEPLOY, _call, _consolidate, _key
from .test_memory_mcp import memory_client as memory_client  # noqa: F401 - the fixture, used by name

CHAT = f"{API_ROOT}/chat/completions"
MEMORY_TOOLS = {"memory_search", "memory_get", "memory_propose", "memory_confirm", "memory_flag", "memory_forget"}


def _completion(call: tuple[str, dict[str, Any]] | None) -> ChatCompletion:
    calls = (
        [
            ChatCompletionMessageFunctionToolCall(
                id="call_0", type="function", function=Function(name=call[0], arguments=json.dumps(call[1]))
            )
        ]
        if call
        else None
    )
    return ChatCompletion(
        id="chatcmpl-memory",
        object="chat.completion",
        created=0,
        model=MODEL_NAME,
        choices=[
            Choice(
                index=0,
                message=ChatCompletionMessage(
                    role="assistant", content=None if call else "done", tool_calls=cast(Any, calls)
                ),
                finish_reason="tool_calls" if call else "stop",
            )
        ],
        usage=CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


@contextmanager
def _provider(*responses: ChatCompletion) -> Iterator[list[dict[str, Any]]]:
    """The provider, answering ``responses`` in turn wherever the gateway calls it from; what it was sent."""
    sent: list[dict[str, Any]] = []
    remaining = iter(responses)

    async def fake(*args: Any, **kwargs: Any) -> ChatCompletion:
        sent.append(kwargs)
        return next(remaining)

    with (
        patch("gateway.services.mcp_loop.acompletion", new=fake),
        patch("gateway.api.routes.chat.acompletion", new=fake),
    ):
        yield sent


def _offered(sent: dict[str, Any]) -> set[str]:
    return {tool["function"]["name"] for tool in sent.get("tools") or [] if tool.get("type") == "function"}


def _switches(
    client: TestClient, db_session_factory: Callable[[], Session], master: dict[str, str], **values: Any
) -> None:
    with db_session_factory() as session:
        workspace = session.scalars(select(Workspace)).first()
        assert workspace is not None
        workspace_id = workspace.id
    body = {"enabled": True, "workspace_scope_enabled": True, "write_policy": "auto", **values}
    response = client.put(f"{API_ROOT}/workspaces/{workspace_id}/memory", json=body, headers=master)
    assert response.status_code == 200, response.text


def _chat(client: TestClient, headers: dict[str, str], **body: Any) -> Any:
    return client.post(
        CHAT,
        json={"model": MODEL_NAME, "messages": [{"role": "user", "content": "deploy it"}], **body},
        headers=headers,
    )


def test_a_request_that_declares_memory_runs_its_tools_in_the_loop(
    memory_client: TestClient, master_key_header: dict[str, str], db_session_factory: Callable[[], Session]
) -> None:
    headers = _key(memory_client, master_key_header, db_session_factory)
    propose = ("memory_propose", {"text": DEPLOY, "kind": "procedure"})
    with _provider(_completion(propose), _completion(None)) as sent:
        response = _chat(memory_client, headers, tools=[{"type": "otari_memory"}])

    assert response.status_code == 200, response.text
    assert response.json()["choices"][0]["message"]["content"] == "done"
    assert MEMORY_TOOLS <= _offered(sent[0])
    assert all(tool.get("type") != "otari_memory" for tool in sent[0]["tools"])
    [result] = [message for message in sent[1]["messages"] if message.get("role") == "tool"]
    assert result["content"].startswith(UNTRUSTED_NOTICE) and '"state": "pending"' in result["content"]
    assert _consolidate(memory_client) == 1
    [entry] = _call(memory_client, headers, "memory_search", query="deploy")["structuredContent"]["results"]
    assert entry["text"] == DEPLOY


def test_the_header_asks_for_memory_as_a_declaration_does(
    memory_client: TestClient, master_key_header: dict[str, str], db_session_factory: Callable[[], Session]
) -> None:
    headers = _key(memory_client, master_key_header, db_session_factory)
    with _provider(_completion(("memory_search", {"query": "deploy"})), _completion(None)) as sent:
        response = _chat(memory_client, {**headers, "Otari-Memory": "on"})

    assert response.status_code == 200, response.text
    assert MEMORY_TOOLS <= _offered(sent[0])


def test_a_request_that_does_not_ask_gets_no_memory_unless_its_workspace_offers_it(
    memory_client: TestClient, master_key_header: dict[str, str], db_session_factory: Callable[[], Session]
) -> None:
    headers = _key(memory_client, master_key_header, db_session_factory)
    with _provider(_completion(None)) as sent:
        assert _chat(memory_client, headers).status_code == 200
    assert not _offered(sent[0])

    _switches(memory_client, db_session_factory, master_key_header, offer_to_every_request=True)
    with _provider(_completion(None)) as sent:
        assert _chat(memory_client, headers).status_code == 200
    assert MEMORY_TOOLS <= _offered(sent[0])

    with _provider(_completion(None)) as sent:
        assert _chat(memory_client, {**headers, "Otari-Memory": "off"}).status_code == 200
    assert not _offered(sent[0])


def test_asking_where_memory_cannot_be_had_is_refused_before_the_provider_is_called(
    memory_client: TestClient, master_key_header: dict[str, str], db_session_factory: Callable[[], Session]
) -> None:
    off = _key(memory_client, master_key_header, db_session_factory, memory_on=False)
    billing = _key(memory_client, master_key_header, db_session_factory, member=False)

    with _provider() as sent:
        assert _chat(memory_client, off, tools=[{"type": "otari_memory"}]).status_code == 403
        assert _chat(memory_client, billing, tools=[{"type": "otari_memory"}]).status_code == 403
        assert _chat(memory_client, {**off, "Otari-Memory": "maybe"}).status_code == 400
        assert _chat(memory_client, {**off, "Otari-Memory": "off"}, tools=[{"type": "otari_memory"}]).status_code == 400
    assert sent == []


def test_where_the_deployment_does_not_offer_memory_asking_for_it_is_refused(
    client: TestClient, api_key_header: dict[str, str]
) -> None:
    with _provider() as sent:
        response = _chat(client, api_key_header, tools=[{"type": "otari_memory"}])

    assert response.status_code == 400 and "does not offer agent memory" in response.text
    assert sent == []


def test_a_request_with_no_memory_is_untouched(client: TestClient, api_key_header: dict[str, str]) -> None:
    with _provider(_completion(None)) as sent:
        assert _chat(client, api_key_header).status_code == 200

    assert not _offered(sent[0])
    assert len(sent) == 1
