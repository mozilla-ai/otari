"""Integration tests for downloading a file a provider's own sandbox produced.

Such a file is a ``file_objects`` row with no ``storage_ref``: Otari holds the
record saying whose it is, and streams the bytes from the provider on demand.
The provider call itself is faked here (the URL and headers it builds are unit
tested); what these cover is the route, the tenant predicate, and what a
listing says about a file whose size Otari does not know.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator
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

from gateway.core.config import API_ROOT
from gateway.models.tools import FileObject
from gateway.services.file_store import LocalDirFileStore

CHART = b"\x89PNG\r\n\x1a\nfake chart bytes"


@pytest.fixture
def tmp_file_store(client: TestClient, tmp_path: Path) -> None:
    cast(Any, client.app).state.file_store = LocalDirFileStore(str(tmp_path))


@pytest.fixture
def provider_file(
    client: TestClient,
    api_key_header: dict[str, str],
    db_session: Session,
    tmp_file_store: None,
) -> str:
    """A file id owned by the test key's user, held by Anthropic rather than locally.

    Uploaded first so the row carries the same user and workspace an upload
    does, then turned into a provider-held row, which is what a native code
    execution records.
    """
    upload = client.post(
        f"{API_ROOT}/files",
        headers=api_key_header,
        files={"file": ("bar_plot.png", b"placeholder", "image/png")},
        data={"purpose": "user_data"},
    )
    assert upload.status_code == 200, upload.text
    file_id = upload.json()["id"]
    db_session.query(FileObject).filter(FileObject.id == file_id).update(
        {
            "storage_ref": None,
            "provider": "anthropic",
            "provider_container_id": None,
            "purpose": "code_execution_output",
            "bytes": 0,
        }
    )
    db_session.commit()
    return str(file_id)


def _serving(payload: bytes) -> Any:
    async def _stream(record: FileObject, config: Any) -> AsyncGenerator[bytes, None]:
        del record, config
        yield payload

    return _stream


def _refusing(exc: BaseException) -> Any:
    async def _stream(record: FileObject, config: Any) -> AsyncGenerator[bytes, None]:
        del record, config
        raise exc
        yield b""  # pragma: no cover - unreachable, keeps this a generator

    return _stream


def test_a_provider_held_file_streams_through_the_gateway(
    client: TestClient,
    api_key_header: dict[str, str],
    provider_file: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("gateway.api.routes.files.stream_provider_file", _serving(CHART))

    resp = client.get(f"{API_ROOT}/files/{provider_file}/content", headers=api_key_header)

    assert resp.status_code == 200
    assert resp.content == CHART
    assert resp.headers["content-type"].startswith("image/png")
    assert "bar_plot.png" in resp.headers["content-disposition"]


def test_metadata_answers_for_a_file_otari_does_not_hold(
    client: TestClient,
    api_key_header: dict[str, str],
    provider_file: str,
) -> None:
    resp = client.get(f"{API_ROOT}/files/{provider_file}", headers=api_key_header)

    assert resp.status_code == 200
    body = resp.json()
    assert body["purpose"] == "code_execution_output"
    # The provider does not say how many bytes there are until they are read.
    assert body["bytes"] == 0
    assert body["filename"] == "bar_plot.png"


def test_a_provider_that_refuses_the_file_is_a_502(
    client: TestClient,
    api_key_header: dict[str, str],
    provider_file: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "gateway.api.routes.files.stream_provider_file",
        _refusing(
            httpx.HTTPStatusError("410", request=httpx.Request("GET", "https://x"), response=httpx.Response(410))
        ),
    )

    resp = client.get(f"{API_ROOT}/files/{provider_file}/content", headers=api_key_header)

    assert resp.status_code == 502
    # The provider's own message never reaches the caller.
    assert "410" not in resp.text


def test_a_deployment_with_no_credential_for_the_provider_is_a_500(
    client: TestClient,
    api_key_header: dict[str, str],
    provider_file: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "gateway.api.routes.files.stream_provider_file",
        _refusing(LookupError("no credential configured for provider 'anthropic'")),
    )

    resp = client.get(f"{API_ROOT}/files/{provider_file}/content", headers=api_key_header)

    assert resp.status_code == 500
    assert "anthropic" not in resp.text


def test_another_users_provider_file_is_not_found(
    client: TestClient,
    master_key_header: dict[str, str],
    provider_file: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The row is what makes the proxy safe: the provider would serve any id.

    A master-key request names the user it reads as, so asking as someone else
    is the same 404 a missing file gets, and the provider is never called.
    """
    called = False

    def _unexpected(record: FileObject, config: Any) -> AsyncGenerator[bytes, None]:
        nonlocal called
        called = True
        stream: AsyncGenerator[bytes, None] = _serving(CHART)(record, config)
        return stream

    monkeypatch.setattr("gateway.api.routes.files.stream_provider_file", _unexpected)

    resp = client.get(
        f"{API_ROOT}/files/{provider_file}/content",
        headers=master_key_header,
        params={"user": "somebody-else"},
    )

    assert resp.status_code == 404
    assert called is False


# --- recording what a provider-native run produced ------------------------------------


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


@pytest.fixture
def anthropic_credentialed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A deployment credentialed for Anthropic by the SDK's own variable, with the metadata call faked."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

    async def _named(provider: str, file_id: str, api_key: str, api_base: str | None) -> str | None:
        del provider, file_id, api_key, api_base
        return "bar_plot.png"

    monkeypatch.setattr("gateway.services.files.provider_files._fetch_filename", _named)


def test_a_file_a_provider_native_run_produced_is_recorded_and_served(
    client: TestClient,
    api_key_header: dict[str, str],
    db_session: Session,
    tmp_file_store: None,
    anthropic_credentialed: None,
) -> None:
    """The provider ran the code and kept the file; the row Otari records is what
    lets ``/v1/files`` answer for the id the caller was handed."""

    async def fake_amessages(**kwargs: Any) -> MessageResponse:
        return _provider_reply(_provider_run_block("file_01provider"))

    with patch("gateway.api.routes.messages.amessages", new=fake_amessages):
        resp = client.post(f"{API_ROOT}/messages", json=_native_request(), headers=api_key_header)
    assert resp.status_code == 200, resp.text

    meta = client.get(f"{API_ROOT}/files/file_01provider", headers=api_key_header)
    assert meta.status_code == 200, meta.text
    assert (meta.json()["filename"], meta.json()["purpose"]) == ("bar_plot.png", "code_execution_output")
    row = db_session.get(FileObject, "file_01provider")
    assert row is not None
    assert (row.storage_ref, row.provider, row.provider_instance) == (None, "anthropic", "anthropic")


def test_a_streamed_provider_native_run_records_its_files_too(
    client: TestClient,
    api_key_header: dict[str, str],
    tmp_file_store: None,
    anthropic_credentialed: None,
) -> None:
    """Anthropic's SDK streams by default, so the stream path owes the same row."""

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

    meta = client.get(f"{API_ROOT}/files/file_01streamed", headers=api_key_header)
    assert meta.status_code == 200, meta.text
    assert meta.json()["filename"] == "bar_plot.png"
