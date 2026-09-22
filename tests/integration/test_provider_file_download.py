"""Integration tests for a file a provider's own sandbox produced.

The bytes are copied into Otari's store as the run is recorded, so the file
outlives the provider's container. A row whose copy failed, and every row
written before copying existed, keeps the older shape: no ``storage_ref``, the
provider named, and the bytes streamed from it on demand. Both are covered
here, along with the route, the tenant predicate, and the output caps. The
provider call itself is faked (the URL and headers it builds are unit tested).
"""

from __future__ import annotations

import base64
import json
from collections.abc import AsyncGenerator, AsyncIterator, Generator
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import httpx
import pytest
from anthropic.types import CodeExecutionOutputBlock, CodeExecutionResultBlock, CodeExecutionToolResultBlock
from any_llm.types.completion import ChatCompletion, ChatCompletionMessage, Choice, CompletionUsage
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

from gateway.core.config import API_ROOT, GatewayConfig
from gateway.models.tools import FileObject
from gateway.services.file_store import LocalDirFileStore
from gateway.services.files import provider_files

from .conftest import build_test_client

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
    """A deployment credentialed for Anthropic, with the metadata and content calls faked.

    The content fake is what lets the copy succeed; without it every copy would
    fail the same way a real provider outage does, and a test asserting the
    stored bytes would pass for the wrong reason.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

    async def _named(provider: str, file_id: str, api_key: str, api_base: str | None) -> str | None:
        del provider, file_id, api_key, api_base
        return "bar_plot.png"

    async def _bytes(
        provider: str, file_id: str, container_id: str | None, api_key: str, api_base: str | None, budget: int
    ) -> AsyncGenerator[bytes, None]:
        del provider, file_id, container_id, api_key, api_base
        if len(CHART) > budget:
            raise provider_files.OutputOverBudget
        yield CHART

    monkeypatch.setattr("gateway.services.files.provider_files._fetch_filename", _named)
    monkeypatch.setattr("gateway.services.files.provider_files._provider_bytes", _bytes)


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
    assert (row.provider, row.provider_instance) == ("anthropic", "anthropic")
    # Copied, not merely named: the id outlives the provider's container.
    assert row.storage_ref is not None
    assert row.bytes == len(CHART)


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


def test_a_copied_file_still_downloads_once_the_provider_has_dropped_it(
    client: TestClient,
    api_key_header: dict[str, str],
    tmp_file_store: None,
    anthropic_credentialed: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The point of copying. OpenAI discards a container twenty minutes after its
    last use, and everything in it, so a caller returning to the conversation the
    next day was holding an id nobody could serve."""

    async def fake_amessages(**kwargs: Any) -> MessageResponse:
        return _provider_reply(_provider_run_block("file_01outlives"))

    with patch("gateway.api.routes.messages.amessages", new=fake_amessages):
        resp = client.post(f"{API_ROOT}/messages", json=_native_request(), headers=api_key_header)
    assert resp.status_code == 200, resp.text

    # The provider is now refusing, as it does once the container is reclaimed.
    monkeypatch.setattr(
        "gateway.api.routes.files.stream_provider_file", _refusing(httpx.HTTPError("container is gone"))
    )
    content = client.get(f"{API_ROOT}/files/file_01outlives/content", headers=api_key_header)

    assert content.status_code == 200, content.text
    assert content.content == CHART


def test_a_file_whose_copy_failed_still_serves_by_proxy(
    client: TestClient,
    api_key_header: dict[str, str],
    db_session: Session,
    tmp_file_store: None,
    anthropic_credentialed: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A copy is best effort: losing it must not cost the caller the row, which is
    what every provider file had before copying existed."""

    async def _unreachable(*args: Any, **kwargs: Any) -> AsyncGenerator[bytes, None]:
        raise httpx.HTTPError("provider unreachable")
        yield b""  # pragma: no cover - unreachable, keeps this a generator

    monkeypatch.setattr("gateway.services.files.provider_files._provider_bytes", _unreachable)

    async def fake_amessages(**kwargs: Any) -> MessageResponse:
        return _provider_reply(_provider_run_block("file_01nocopy"))

    with patch("gateway.api.routes.messages.amessages", new=fake_amessages):
        resp = client.post(f"{API_ROOT}/messages", json=_native_request(), headers=api_key_header)
    assert resp.status_code == 200, resp.text

    row = db_session.get(FileObject, "file_01nocopy")
    assert row is not None
    assert (row.storage_ref, row.bytes) == (None, 0), "the row stays, serving by proxy"

    monkeypatch.setattr("gateway.api.routes.files.stream_provider_file", _serving(CHART))
    content = client.get(f"{API_ROOT}/files/file_01nocopy/content", headers=api_key_header)
    assert content.status_code == 200
    assert content.content == CHART


@pytest.fixture
def one_output_client(test_config: GatewayConfig, clean_database: None, tmp_path: Path) -> Generator[TestClient]:
    """A deployment that will copy one produced file per response and no more."""
    updated = test_config.model_copy(update={"files_output_max_files": 1})
    for candidate in build_test_client(updated):
        cast(Any, candidate.app).state.file_store = LocalDirFileStore(str(tmp_path))
        yield candidate


def test_the_output_caps_apply_to_copied_files(
    one_output_client: TestClient,
    master_key_header: dict[str, str],
    db_session: Session,
    anthropic_credentialed: None,
) -> None:
    """The same two caps a sandbox run's outputs get. Past the count a file is
    still recorded, so its id resolves; it just serves by proxy as before."""
    key = one_output_client.post(f"{API_ROOT}/keys", json={"key_name": "k"}, headers=master_key_header)
    assert key.status_code == 200, key.text
    headers = {next(iter(master_key_header)): f"Bearer {key.json()['key']}"}

    async def fake_amessages(**kwargs: Any) -> MessageResponse:
        return _provider_reply(_provider_run_block("file_01first"), _provider_run_block("file_01second"))

    with patch("gateway.api.routes.messages.amessages", new=fake_amessages):
        resp = one_output_client.post(f"{API_ROOT}/messages", json=_native_request(), headers=headers)
    assert resp.status_code == 200, resp.text

    first = db_session.get(FileObject, "file_01first")
    second = db_session.get(FileObject, "file_01second")
    assert first is not None and second is not None
    assert first.storage_ref is not None, "the first is within the count"
    assert (second.storage_ref, second.bytes) == (None, 0), "past the count, recorded but not copied"


def test_a_produced_file_can_be_attached_to_a_later_request(
    client: TestClient,
    api_key_header: dict[str, str],
    tmp_file_store: None,
    anthropic_credentialed: None,
) -> None:
    """The other half of copying: a chart the provider drew is an ordinary Otari
    file, so the next turn can hand it back to the model. A row with no bytes is
    skipped by the content normalizer, which is what made this impossible."""

    async def fake_amessages(**kwargs: Any) -> MessageResponse:
        return _provider_reply(_provider_run_block("file_01reused"))

    with patch("gateway.api.routes.messages.amessages", new=fake_amessages):
        first = client.post(f"{API_ROOT}/messages", json=_native_request(), headers=api_key_header)
    assert first.status_code == 200, first.text

    captured: dict[str, Any] = {}

    async def capture_acompletion(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return ChatCompletion(
            id="chatcmpl-test",
            object="chat.completion",
            created=1700000000,
            model="llama3",
            choices=[
                Choice(index=0, message=ChatCompletionMessage(role="assistant", content="ok"), finish_reason="stop")
            ],
            usage=CompletionUsage(prompt_tokens=10, completion_tokens=2, total_tokens=12),
        )

    # A model that reads images natively, so the block passes through with the
    # bytes rather than being extracted to text for a text-only one.
    body = {
        "model": "anthropic:claude-sonnet-4-5",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what does this chart show?"},
                    {"type": "image_url", "image_url": {"url": ""}, "file_id": "file_01reused"},
                ],
            }
        ],
    }
    with patch("gateway.api.routes.chat.acompletion", new=capture_acompletion):
        second = client.post(f"{API_ROOT}/chat/completions", headers=api_key_header, json=body)

    assert second.status_code == 200, second.text
    sent = json.dumps(captured["messages"])
    assert base64.b64encode(CHART).decode() in sent, "the produced file's bytes reached the model"
