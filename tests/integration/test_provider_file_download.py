"""Integration tests for a file a provider's own sandbox produced.

Otari copies such a file into its store when the reply arrives, under the
provider's ID. Anthropic's Files API is stubbed here at the HTTP level.
"""

from __future__ import annotations

import base64
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import httpx
import pytest
from anthropic.types import CodeExecutionOutputBlock, CodeExecutionResultBlock, CodeExecutionToolResultBlock
from any_llm.types.messages import (
    ContentBlockStartEvent,
    MessageDelta,
    MessageDeltaEvent,
    MessageDeltaUsage,
    MessageResponse,
    MessageStartEvent,
    MessageStopEvent,
    MessageStreamEvent,
)
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from gateway.adapters.file_storage_adapter import LocalDirFileStore
from gateway.core.config import API_ROOT
from gateway.models.tools import FileObject

CHART = b"\x89PNG\r\n\x1a\nfake chart bytes"


@pytest.fixture
def tmp_file_store(client: TestClient, tmp_path: Path) -> None:
    cast(Any, client.app).state.file_store = LocalDirFileStore(str(tmp_path))


def _provider_run_block(file_id: str) -> CodeExecutionToolResultBlock:
    """Anthropic's own result block for a run that wrote one file."""
    return CodeExecutionToolResultBlock(
        type="code_execution_tool_result",
        tool_use_id="srvtoolu_01provider",
        content=CodeExecutionResultBlock(
            type="code_execution_result",
            stdout="",
            stderr="",
            return_code=0,
            content=[CodeExecutionOutputBlock(type="code_execution_output", file_id=file_id)],
        ),
    )


def _provider_reply(*blocks: Any) -> MessageResponse:
    return MessageResponse(
        id="msg_test",
        type="message",
        role="assistant",
        model="claude-sonnet-4-5",
        content=list(blocks),
        stop_reason=cast(Any, "end_turn"),
        stop_sequence=None,
        usage=cast(Any, {"input_tokens": 3, "output_tokens": 2}),
    )


async def _stream_of(*events: MessageStreamEvent) -> AsyncIterator[MessageStreamEvent]:
    for event in events:
        yield event


def _native_request(*, stream: bool = False) -> dict[str, Any]:
    return {
        "model": "anthropic:claude-sonnet-4-5",
        "messages": [{"role": "user", "content": "plot it"}],
        "max_tokens": 100,
        "tools": [{"type": "code_execution_20250825", "name": "code_execution"}],
        "stream": stream,
    }


class _StubAnthropic:
    """Anthropic's Files API, as far as copying a produced file needs it."""

    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.serving = True

    def handle(self, request: httpx.Request) -> httpx.Response:
        prefix = "/v1/files/"
        file_id, _, tail = request.url.path.removeprefix(prefix).partition("/")
        if not self.serving or not request.url.path.startswith(prefix) or file_id not in self.files:
            return httpx.Response(404)
        if tail == "content":
            return httpx.Response(200, content=self.files[file_id])
        return httpx.Response(
            200,
            json={
                "id": file_id,
                "type": "file",
                "filename": "bar_plot.png",
                "mime_type": "image/png",
                "size_bytes": len(self.files[file_id]),
                "created_at": "2026-01-01T00:00:00Z",
            },
        )


@pytest.fixture
def anthropic(monkeypatch: pytest.MonkeyPatch) -> _StubAnthropic:
    """A deployment credentialed for Anthropic by the SDK's own variable, whose Files API is stubbed."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    stub = _StubAnthropic()
    original_init = httpx.AsyncClient.__init__

    def patched_init(self: httpx.AsyncClient, *args: Any, **kwargs: Any) -> None:
        kwargs["transport"] = httpx.MockTransport(stub.handle)
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", patched_init)
    return stub


def _run_natively(client: TestClient, headers: dict[str, str], file_id: str) -> None:
    async def fake_amessages(**kwargs: Any) -> MessageResponse:
        return _provider_reply(_provider_run_block(file_id))

    with patch("gateway.api.routes.messages.amessages", new=fake_amessages):
        resp = client.post(f"{API_ROOT}/messages", json=_native_request(), headers=headers)
    assert resp.status_code == 200, resp.text


def test_a_produced_file_downloads_after_the_provider_stops_serving_it(
    client: TestClient,
    api_key_header: dict[str, str],
    db_session: Session,
    tmp_file_store: None,
    anthropic: _StubAnthropic,
) -> None:
    anthropic.files["file_01provider"] = CHART

    _run_natively(client, api_key_header, "file_01provider")
    # The provider discards the container, and the file with it.
    anthropic.serving = False

    download = client.get(f"{API_ROOT}/files/file_01provider/content", headers=api_key_header)
    assert download.status_code == 200, download.text
    assert download.content == CHART
    assert "bar_plot.png" in download.headers["content-disposition"]
    meta = client.get(f"{API_ROOT}/files/file_01provider", headers=api_key_header).json()
    assert (meta["filename"], meta["purpose"], meta["bytes"]) == ("bar_plot.png", "code_execution_output", len(CHART))
    row = db_session.get(FileObject, "file_01provider")
    assert row is not None
    assert row.storage_ref is not None
    assert (row.provider, row.provider_instance) == ("anthropic", "anthropic")


def test_a_streamed_provider_native_run_copies_its_files_too(
    client: TestClient,
    api_key_header: dict[str, str],
    tmp_file_store: None,
    anthropic: _StubAnthropic,
) -> None:
    """Anthropic's SDK streams by default, so the stream path owes the same copy."""
    anthropic.files["file_01streamed"] = CHART

    async def fake_amessages(**kwargs: Any) -> AsyncIterator[MessageStreamEvent]:
        return _stream_of(
            MessageStartEvent(type="message_start", message=cast(Any, _provider_reply())),
            ContentBlockStartEvent(
                type="content_block_start", index=0, content_block=_provider_run_block("file_01streamed")
            ),
            MessageDeltaEvent(
                type="message_delta",
                delta=MessageDelta(stop_reason=cast(Any, "end_turn"), stop_sequence=None),
                usage=MessageDeltaUsage(
                    input_tokens=None,
                    output_tokens=1,
                    cache_creation_input_tokens=None,
                    cache_read_input_tokens=None,
                    server_tool_use=None,
                ),
            ),
            MessageStopEvent(type="message_stop"),
        )

    with patch("gateway.api.routes.messages.amessages", new=fake_amessages):
        resp = client.post(f"{API_ROOT}/messages", json=_native_request(stream=True), headers=api_key_header)
    assert resp.status_code == 200, resp.text
    assert "file_01streamed" in resp.text
    anthropic.serving = False

    download = client.get(f"{API_ROOT}/files/file_01streamed/content", headers=api_key_header)
    assert download.status_code == 200, download.text
    assert download.content == CHART


def test_a_produced_file_referenced_in_a_later_request_reaches_the_model(
    client: TestClient,
    api_key_header: dict[str, str],
    tmp_file_store: None,
    anthropic: _StubAnthropic,
) -> None:
    anthropic.files["file_01provider"] = CHART
    _run_natively(client, api_key_header, "file_01provider")
    anthropic.serving = False
    sent: dict[str, Any] = {}

    async def fake_amessages(**kwargs: Any) -> MessageResponse:
        sent.update(kwargs)
        return _provider_reply()

    later = {
        "model": "anthropic:claude-sonnet-4-5",
        "max_tokens": 100,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "file", "file_id": "file_01provider"}},
                    {"type": "text", "text": "What does this chart show?"},
                ],
            }
        ],
    }
    with patch("gateway.api.routes.messages.amessages", new=fake_amessages):
        resp = client.post(f"{API_ROOT}/messages", json=later, headers=api_key_header)

    assert resp.status_code == 200, resp.text
    assert base64.b64encode(CHART).decode() in json.dumps(sent["messages"])


def test_a_file_the_provider_will_not_serve_is_not_recorded(
    client: TestClient,
    api_key_header: dict[str, str],
    db_session: Session,
    tmp_file_store: None,
    anthropic: _StubAnthropic,
) -> None:
    """The reply stands, and Otari does not claim a file it holds no bytes for."""
    _run_natively(client, api_key_header, "file_01missing")

    assert client.get(f"{API_ROOT}/files/file_01missing", headers=api_key_header).status_code == 404
    assert db_session.get(FileObject, "file_01missing") is None


def test_another_users_copied_file_is_not_found(
    client: TestClient,
    api_key_header: dict[str, str],
    master_key_header: dict[str, str],
    tmp_file_store: None,
    anthropic: _StubAnthropic,
) -> None:
    """A master-key request names the user it reads as, so asking as someone else is a 404."""
    anthropic.files["file_01provider"] = CHART
    _run_natively(client, api_key_header, "file_01provider")

    resp = client.get(
        f"{API_ROOT}/files/file_01provider/content",
        headers=master_key_header,
        params={"user": "somebody-else"},
    )

    assert resp.status_code == 404


def test_a_row_with_no_stored_bytes_is_not_found(
    client: TestClient,
    api_key_header: dict[str, str],
    db_session: Session,
    tmp_file_store: None,
) -> None:
    upload = client.post(
        f"{API_ROOT}/files",
        headers=api_key_header,
        files={"file": ("bar_plot.png", CHART, "image/png")},
        data={"purpose": "user_data"},
    )
    assert upload.status_code == 200, upload.text
    file_id = upload.json()["id"]
    db_session.query(FileObject).filter(FileObject.id == file_id).update({"storage_ref": None, "provider": "anthropic"})
    db_session.commit()

    resp = client.get(f"{API_ROOT}/files/{file_id}/content", headers=api_key_header)

    assert resp.status_code == 404
