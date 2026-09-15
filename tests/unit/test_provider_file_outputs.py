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
from gateway.services.provider_files.client import PlatformFilesClient
from gateway.services.provider_files.contracts import FileAccount, FileMetadata, FilesError, Operation


class Event(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: str
    content_block: dict[str, Any] | None = None


@pytest.mark.asyncio
@pytest.mark.parametrize("fail", [False, True])
async def test_file_block_is_held_until_registration(monkeypatch: pytest.MonkeyPatch, fail: bool) -> None:
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
    binder = inference.FileOutputBinder(PlatformFilesClient("https://authority", "gateway", "user"), operation, [])

    async def source() -> AsyncIterator[Event]:
        yield Event(type="message_start")
        yield Event(
            type="content_block_start",
            content_block={
                "type": "bash_code_execution_tool_result",
                "content": [{"type": "code_execution_output", "file_id": "file_generated"}],
            },
        )
        yield Event(type="content_block_stop")
        yield Event(type="message_stop")

    async def consume() -> None:
        async for event in binder.stream(source()):
            emitted.append(event.type)  # noqa: PERF401 (observe emission before the stream finishes)

    task = asyncio.create_task(consume())
    await asyncio.wait_for(entered.wait(), timeout=1)
    assert emitted == ["message_start"]
    release.set()
    if fail:
        with pytest.raises(FilesError):
            await task
        assert emitted == ["message_start"]
    else:
        await task
        assert emitted == ["message_start", "content_block_start", "content_block_stop", "message_stop"]
