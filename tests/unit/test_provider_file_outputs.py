"""Structured stream blocks cannot expose an uncommitted provider file ID."""

import asyncio
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict, SecretStr

from gateway.services.provider_files import inference
from gateway.services.provider_files.anthropic_inference import AnthropicFileOutputBinder
from gateway.services.provider_files.client import PlatformFilesClient
from gateway.services.provider_files.contracts import FileAccount, FileMetadata, FilesError, Operation


class Event(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: str
    content_block: dict[str, Any] | None = None


@pytest.mark.asyncio
@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("output_type", ["code_execution_output", "bash_code_execution_output"])
async def test_file_block_is_held_until_registration(
    monkeypatch: pytest.MonkeyPatch, fail: bool, output_type: str
) -> None:
    entered, release = asyncio.Event(), asyncio.Event()
    operation = Operation(
        id=uuid.uuid4(),
        cleanup_token=SecretStr("cleanup"),
        deadline=datetime.now(UTC) + timedelta(minutes=1),
        account=FileAccount(generation_id=uuid.uuid4(), api_key=SecretStr("key")),
        max_bytes=100,
        expires_in_seconds=3600,
    )
    metadata = FileMetadata(
        id="file_generated",
        filename="output.csv",
        mime_type="text/csv",
        size_bytes=4,
        created_at=datetime.now(UTC),
        downloadable=True,
    )
    emitted: list[str] = []

    class Provider:
        async def aretrieve_file(self, *args: Any, **kwargs: Any) -> FileMetadata:
            return metadata

    @asynccontextmanager
    async def provider(*args: Any, **kwargs: Any) -> AsyncIterator[Provider]:
        yield Provider()

    async def retry(self: Any, path: str, body: dict[str, Any], result_type: type[Any]) -> Any:
        if path == "outputs/register":
            entered.set()
            await release.wait()
            if fail:
                raise FilesError(502, "unavailable")
            return metadata
        if path.endswith("/abandon"):
            raise FilesError(502, "unavailable")
        return result_type()

    monkeypatch.setattr(inference, "provider_client", provider)
    monkeypatch.setattr(PlatformFilesClient, "retry", retry)
    binder = AnthropicFileOutputBinder(PlatformFilesClient("https://authority", "gateway", "user"), operation, [])

    async def source() -> AsyncIterator[Event]:
        yield Event(type="message_start")
        yield Event(
            type="content_block_start",
            content_block={
                "type": "bash_code_execution_tool_result",
                "content": {
                    "type": "bash_code_execution_result",
                    "content": [{"type": output_type, "file_id": "file_generated"}],
                },
            },
        )
        yield Event(type="content_block_stop")
        yield Event(type="message_stop")

    async def consume() -> None:
        async for event in binder.stream(source()):
            emitted.append(event.type)  # noqa: PERF401 (observe emission before the stream finishes)

    task = asyncio.create_task(consume())
    registration = asyncio.create_task(entered.wait())
    try:
        done, _ = await asyncio.wait({task, registration}, timeout=1, return_when=asyncio.FIRST_COMPLETED)
        if task in done:
            await task
        assert registration in done, "Output registration did not start"
    finally:
        registration.cancel()
    assert emitted == ["message_start"]
    release.set()
    if fail:
        with pytest.raises(FilesError):
            await task
        assert emitted == ["message_start"]
    else:
        await task
        assert emitted == ["message_start", "content_block_start", "content_block_stop", "message_stop"]


@pytest.mark.asyncio
async def test_generic_output_registration_accepts_ids_without_anthropic_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operation = Operation(
        id=uuid.uuid4(),
        cleanup_token=SecretStr("cleanup"),
        deadline=datetime.now(UTC) + timedelta(minutes=1),
        account=FileAccount(generation_id=uuid.uuid4(), provider="openai", api_key=SecretStr("key")),
        max_bytes=100,
        expires_in_seconds=3600,
    )
    metadata = FileMetadata(id="file_generated", purpose="user_data")
    calls: list[str] = []

    class Provider:
        async def aretrieve_file(self, file_id: str, **kwargs: Any) -> FileMetadata:
            calls.append(file_id)
            return metadata

    @asynccontextmanager
    async def provider(account: FileAccount, **kwargs: Any) -> AsyncIterator[Provider]:
        assert account.provider == "openai"
        yield Provider()

    async def retry(self: Any, path: str, body: dict[str, Any], result_type: type[Any]) -> Any:
        assert path == "outputs/register"
        assert body["metadata"]["purpose"] == "user_data"
        return metadata

    monkeypatch.setattr(inference, "provider_client", provider)
    monkeypatch.setattr(PlatformFilesClient, "retry", retry)
    binder = inference.FileOutputBinder(PlatformFilesClient("https://authority", "gateway", "user"), operation, [])
    await binder.register_ids(["file_generated", "file_generated"])
    assert calls == ["file_generated"]
