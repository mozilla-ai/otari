"""Upload compensation survives handler cancellation but remains time-bounded."""

import asyncio
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from pydantic import SecretStr

from gateway.api.routes import hybrid_files
from gateway.services.provider_files.client import PlatformFilesClient
from gateway.services.provider_files.contracts import FileAccount, FileMetadata, FilesError, Operation


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_handler", [False, True])
@pytest.mark.parametrize("expire", [False, True])
async def test_upload_compensation_survives_cancellation_and_honors_timeout(
    monkeypatch: pytest.MonkeyPatch, cancel_handler: bool, expire: bool
) -> None:
    entered, release, finished = asyncio.Event(), asyncio.Event(), asyncio.Event()
    events: list[str] = []
    operation = Operation(
        id=uuid.uuid4(),
        cleanup_token=SecretStr("cleanup"),
        deadline=datetime.now(UTC) + timedelta(minutes=1),
        account=FileAccount(generation_id=uuid.uuid4(), api_key=SecretStr("key")),
        max_bytes=100,
        expires_in_seconds=3600,
    )
    metadata = FileMetadata(id="file_uploaded")

    class Provider:
        async def adelete_file(self, *args: Any, **kwargs: Any) -> None:
            entered.set()
            try:
                await release.wait()
                events.append("deleted")
            except asyncio.CancelledError:
                events.append("delete-cancelled")
                raise

    @asynccontextmanager
    async def provider(*args: Any, **kwargs: Any) -> AsyncIterator[Provider]:
        yield Provider()

    async def retry(self: Any, path: str, body: dict[str, Any], result_type: type[Any]) -> Any:
        assert path == f"uploads/{operation.id}/abandon"
        assert body["metadata"] == {"id": metadata.id}
        assert body["deleted"] is True
        events.append("reported")
        return result_type()

    real_wait_for = asyncio.wait_for
    timer = asyncio.timeout(None)

    async def controlled_wait_for(awaitable: Any, timeout: float) -> Any:
        assert timeout == 20
        try:
            async with timer:
                return await awaitable
        finally:
            finished.set()

    monkeypatch.setattr(hybrid_files, "provider_client", provider)
    monkeypatch.setattr(PlatformFilesClient, "retry", retry)
    monkeypatch.setattr(asyncio, "wait_for", controlled_wait_for)
    handler = asyncio.create_task(
        hybrid_files._compensate_upload(
            PlatformFilesClient("https://authority", "gateway", "user"),
            operation,
            metadata,
            {},
            True,
            FilesError(502, "unavailable"),
        )
    )
    try:
        await real_wait_for(entered.wait(), timeout=1)
        if cancel_handler:
            handler.cancel()
            with pytest.raises(asyncio.CancelledError):
                await handler
            assert events == []
        if expire:
            timer.reschedule(asyncio.get_running_loop().time())
        else:
            release.set()
        await real_wait_for(finished.wait(), timeout=1)
        if not cancel_handler:
            await handler
        assert events == (["delete-cancelled"] if expire else ["deleted", "reported"])
    finally:
        release.set()
        if not handler.done():
            await real_wait_for(handler, timeout=1)
