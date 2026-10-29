"""A request routed through the gateway keeps its content only when it asks and its deployment and workspace allow it.

The trace writer is replaced by one that keeps what it is handed, so these tests
read the write the request produced rather than waiting for its rows.
"""

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import pytest
from any_llm.types.completion import ChatCompletion, ChatCompletionMessage, Choice, CompletionUsage
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from gateway.core.config import API_ROOT, TRACE_CONTENT_HEADER
from gateway.models.api_keys import APIKey
from gateway.models.tenancy import OrganizationMember, Workspace
from gateway.models.tenancy import User as TenancyUser
from gateway.types.traces import TraceWrite

from .conftest import MODEL_NAME

_KEY_NAME = "routed-content"


class _RecordingWriter:
    def __init__(self) -> None:
        self.writes: list[TraceWrite] = []

    def submit(self, trace: TraceWrite, *, truncated: int = 0) -> None:
        self.writes.append(trace)


def _completion() -> ChatCompletion:
    return ChatCompletion(
        id="chatcmpl-content",
        object="chat.completion",
        created=0,
        model=MODEL_NAME,
        choices=[Choice(index=0, message=ChatCompletionMessage(role="assistant", content="ok"), finish_reason="stop")],
        usage=CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


def _member_key(
    client: TestClient, master_key_header: dict[str, str], db_session_factory: Callable[[], Session]
) -> tuple[dict[str, str], Callable[[bool], None]]:
    """A key for a person who is an admin of the default workspace's organization, and a switch for its content."""
    with db_session_factory() as session:
        workspace = session.scalars(select(Workspace)).first()
        assert workspace is not None
        person = TenancyUser(
            email="dev@example.test", full_name="Dev", active_organization_id=workspace.organization_id
        )
        session.add(person)
        session.flush()
        session.add(
            OrganizationMember(
                organization_id=workspace.organization_id, user_id=person.id, role="admin", status="active"
            )
        )
        session.commit()
        person_id = str(person.id)
    client.post(f"{API_ROOT}/users", json={"user_id": person_id}, headers=master_key_header)
    resp = client.post(
        f"{API_ROOT}/keys",
        json={"key_name": _KEY_NAME, "user_id": person_id, "exclude_from_budget": True},
        headers=master_key_header,
    )
    assert resp.status_code == 200, resp.text

    def workspace_keeps(value: bool) -> None:
        with db_session_factory() as session:
            key = session.scalars(select(APIKey).where(APIKey.key_name == _KEY_NAME)).one()
            workspace = session.get(Workspace, key.workspace_id)
            assert workspace is not None
            workspace.trace_content_enabled = value
            session.commit()

    return {"Otari-Key": f"Bearer {resp.json()['key']}"}, workspace_keeps


@pytest.fixture
def routed(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[_RecordingWriter, dict[str, str], Callable[[bool], None], Any]:
    app = cast(FastAPI, client.app)
    writer = _RecordingWriter()
    monkeypatch.setattr(app.state, "trace_writer", writer)
    # Sealing itself is covered where content is stored; here only whether a key could seal matters.
    monkeypatch.setattr(app.state, "trace_content", SimpleNamespace(sealing=True, close=lambda: None))
    monkeypatch.setattr(app.state.config, "trace_capture_enabled", True)
    monkeypatch.setattr(app.state.config, "trace_content_enabled", True)
    headers, workspace_keeps = _member_key(client, master_key_header, db_session_factory)
    workspace_keeps(True)
    return writer, headers, workspace_keeps, app.state.config


def _chat(client: TestClient, headers: dict[str, str], messages: list[dict[str, Any]]) -> None:
    with patch("gateway.api.routes.chat.acompletion", return_value=_completion()):
        response = client.post(
            f"{API_ROOT}/chat/completions", json={"model": MODEL_NAME, "messages": messages}, headers=headers
        )
    assert response.status_code == 200, response.text


_PROMPT: list[dict[str, Any]] = [{"role": "user", "content": "fix the flaky test"}]
_ANSWER: list[dict[str, Any]] = [
    {"role": "user", "content": "go"},
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "ls", "arguments": '{"p":"."}'}}],
    },
    {"role": "tool", "tool_call_id": "call_1", "content": "a.txt"},
]


def test_a_request_that_asks_keeps_its_prompt(client: TestClient, routed: Any) -> None:
    writer, headers, _, _ = routed

    _chat(client, {**headers, TRACE_CONTENT_HEADER: "keep"}, _PROMPT)

    [write] = writer.writes
    assert write.content_generation is not None
    assert write.spans[0].content == {"prompt": "fix the flaky test"}


def test_a_request_that_does_not_ask_keeps_nothing(client: TestClient, routed: Any) -> None:
    writer, headers, _, _ = routed

    _chat(client, headers, _PROMPT)

    [write] = writer.writes
    assert write.content_generation is None
    assert all(not span.content for span in write.spans)


def test_a_workspace_that_keeps_no_content_keeps_nothing_however_the_request_asks(
    client: TestClient, routed: Any
) -> None:
    writer, headers, workspace_keeps, _ = routed
    workspace_keeps(False)

    _chat(client, {**headers, TRACE_CONTENT_HEADER: "keep"}, _PROMPT)

    [write] = writer.writes
    assert write.content_generation is None


def test_a_deployment_that_keeps_no_content_keeps_nothing(
    client: TestClient, routed: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    writer, headers, _, config = routed
    monkeypatch.setattr(config, "trace_content_enabled", False)

    _chat(client, {**headers, TRACE_CONTENT_HEADER: "keep"}, _PROMPT)

    [write] = writer.writes
    assert write.content_generation is None


def test_an_answered_call_keeps_its_result_only_where_the_deployment_keeps_tool_output(
    client: TestClient, routed: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    writer, headers, _, config = routed
    keep = {**headers, TRACE_CONTENT_HEADER: "keep"}

    _chat(client, keep, _ANSWER)
    monkeypatch.setattr(config, "trace_content_tool_output", True)
    _chat(client, keep, _ANSWER)

    without, with_output = (next(span for span in write.spans if span.tool_type == "client") for write in writer.writes)
    assert without.content == {"tool.input": '{"p":"."}'}
    assert with_output.content == {"tool.input": '{"p":"."}', "tool.output": "a.txt"}
