"""Captured content: off until a workspace admin turns it on, stored sealed, readable by an admin, purgeable."""

import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from any_llm.types.completion import ChatCompletion, ChatCompletionMessage, Choice, CompletionUsage
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from gateway.core.config import API_ROOT, REQUEST_ID_HEADER, GatewayConfig
from gateway.models.traces import TraceContentKey, TraceSpanContent
from gateway.ports.trace_storage_port import TraceStoragePort
from gateway.services.secret_box import generate_secret_key

from .conftest import MODEL_NAME

_SECRET_PROMPT = "the merger closes on friday"
_MESSAGES = [
    {"role": "user", "content": "x"},
    {
        "role": "assistant",
        "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "Bash", "arguments": '{"command": "cat notes"}'}}
        ],
    },
    {"role": "tool", "tool_call_id": "call_1", "content": _SECRET_PROMPT},
]


@pytest.fixture
def test_config(test_config: GatewayConfig, tmp_path: Path) -> GatewayConfig:
    """A deployment whose operator permits content: the ceiling is off by default."""
    return test_config.model_copy(update={"trace_content_capture_max": "full", "files_local_dir": str(tmp_path)})


@pytest.fixture(autouse=True)
def secret_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())


def _completion() -> ChatCompletion:
    return ChatCompletion(
        id="c",
        object="chat.completion",
        created=0,
        model=MODEL_NAME,
        choices=[
            Choice(index=0, message=ChatCompletionMessage(role="assistant", content="done"), finish_reason="stop")
        ],
        usage=CompletionUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )


def _send(client: TestClient, headers: dict[str, str]) -> str:
    with patch("gateway.api.routes.chat.acompletion", return_value=_completion()):
        response = client.post(
            f"{API_ROOT}/chat/completions", json={"model": MODEL_NAME, "messages": _MESSAGES}, headers=headers
        )
    assert response.status_code == 200, response.text
    return str(response.headers[REQUEST_ID_HEADER])


def _wait_for(factory: Callable[[], Session], model: Any, count: int) -> list[Any]:
    deadline = time.monotonic() + 10
    while True:
        with factory() as session:
            rows = list(session.scalars(select(model)))
        if len(rows) >= count or time.monotonic() > deadline:
            return rows
        time.sleep(0.1)


def _workspace(client: TestClient, headers: dict[str, str], api_key_obj: dict[str, Any]) -> str:
    response = client.get(f"{API_ROOT}/keys/{api_key_obj['id']}", headers=headers)
    assert response.status_code == 200, response.text
    return str(response.json()["workspace_id"])


def _settings(client: TestClient, headers: dict[str, str], workspace_id: str, level: str | None = None) -> Any:
    url = f"{API_ROOT}/workspaces/{workspace_id}/trace-settings"
    response = (
        client.put(url, json={"content_capture": level}, headers=headers) if level else client.get(url, headers=headers)
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_content_is_off_until_an_admin_turns_it_on(
    client: TestClient,
    master_key_header: dict[str, str],
    api_key_obj: dict[str, Any],
    api_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
) -> None:
    workspace_id = _workspace(client, master_key_header, api_key_obj)
    assert _settings(client, master_key_header, workspace_id)["content_capture"] == "off"

    _send(client, api_key_header)
    time.sleep(1.5)

    with db_session_factory() as session:
        assert list(session.scalars(select(TraceSpanContent))) == []


def test_captured_content_is_stored_sealed_read_by_an_admin_and_purged(
    tmp_path: Path,
    client: TestClient,
    master_key_header: dict[str, str],
    api_key_obj: dict[str, Any],
    api_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
) -> None:
    workspace_id = _workspace(client, master_key_header, api_key_obj)
    assert _settings(client, master_key_header, workspace_id, "tool_io")["effective"] == "tool_io"

    request_id = _send(client, api_key_header)
    [row] = _wait_for(db_session_factory, TraceSpanContent, 1)

    blob = (tmp_path / row.storage_ref).read_bytes()
    assert _SECRET_PROMPT.encode() not in blob, "content is stored sealed, never in the clear"
    detail = client.get(f"{API_ROOT}/traces/{row.trace_id}", headers=master_key_header).json()
    tool = next(span for span in detail["spans"] if span["kind"] == "tool")
    assert tool["has_content"] is True
    assert any(span["span_id"] == request_id for span in detail["spans"])

    read = client.post(
        f"{API_ROOT}/traces/{row.trace_id}/spans/{tool['span_id']}/content/break-glass",
        json={"reason": "Integration test of the content path"},
        headers=master_key_header,
    )
    assert read.status_code == 200, read.text
    assert read.json()["fields"] == {"arguments": '{"command": "cat notes"}', "result": _SECRET_PROMPT}

    purged = client.post(
        f"{API_ROOT}/workspaces/{workspace_id}/trace-settings/purge-content", headers=master_key_header
    )
    assert purged.json()["removed"] == 1
    gone = client.post(
        f"{API_ROOT}/traces/{row.trace_id}/spans/{tool['span_id']}/content/break-glass",
        json={"reason": "Integration test of the content path"},
        headers=master_key_header,
    )
    assert gone.status_code == 404
    assert not (tmp_path / row.storage_ref).exists(), "a purge deletes the stored blob"
    with db_session_factory() as session:
        assert list(session.scalars(select(TraceContentKey))) == []


def test_an_expired_session_takes_its_content_and_its_key_with_it(
    tmp_path: Path,
    client: TestClient,
    master_key_header: dict[str, Any],
    api_key_obj: dict[str, Any],
    api_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
) -> None:
    workspace_id = _workspace(client, master_key_header, api_key_obj)
    _settings(client, master_key_header, workspace_id, "tool_io")
    _send(client, api_key_header)
    [row] = _wait_for(db_session_factory, TraceSpanContent, 1)
    assert (tmp_path / row.storage_ref).exists()

    store = client.app.state.container.resolve(TraceStoragePort, None)  # type: ignore[attr-defined]
    removed = client.portal.call(store.expire, datetime.now(UTC) + timedelta(days=1))  # type: ignore[union-attr]

    assert removed == 1
    assert not (tmp_path / row.storage_ref).exists()
    with db_session_factory() as session:
        assert list(session.scalars(select(TraceContentKey))) == []
        assert list(session.scalars(select(TraceSpanContent))) == []


def test_a_workspace_cannot_keep_more_than_the_deployment_permits(
    client: TestClient, master_key_header: dict[str, str], api_key_obj: dict[str, Any]
) -> None:
    client.app.state.config.trace_content_capture_max = "off"  # type: ignore[attr-defined]
    try:
        response = client.put(
            f"{API_ROOT}/workspaces/{_workspace(client, master_key_header, api_key_obj)}/trace-settings",
            json={"content_capture": "full"},
            headers=master_key_header,
        )
    finally:
        client.app.state.config.trace_content_capture_max = "full"  # type: ignore[attr-defined]

    assert response.status_code == 400
