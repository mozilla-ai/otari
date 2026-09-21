"""Upload compensation survives cancellation and reserves time for abandonment."""

import asyncio
import uuid
from collections.abc import AsyncIterator, Coroutine
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
@pytest.mark.parametrize("expire", [None, "delete", "report"])
async def test_upload_compensation_survives_cancellation_and_honors_timeout(
    monkeypatch: pytest.MonkeyPatch, cancel_handler: bool, expire: str | None
) -> None:
    entered, release = asyncio.Event(), asyncio.Event()
    reporting, report_release = asyncio.Event(), asyncio.Event()
    events: list[str] = []
    timers: list[asyncio.Timeout] = []
    tasks: list[asyncio.Task[None]] = []
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
        assert body["deleted"] is (expire != "delete")
        assert body["outcome_unknown"] is False
        reporting.set()
        try:
            await report_release.wait()
            events.append("reported")
        except asyncio.CancelledError:
            events.append("report-cancelled")
            raise
        return result_type()

    real_timeout, real_create_task = asyncio.timeout, asyncio.create_task

    def controlled_timeout(delay: float | None) -> asyncio.Timeout:
        assert delay == 10
        timer = real_timeout(None)
        timers.append(timer)
        return timer

    def capture_task(coroutine: Coroutine[Any, Any, None]) -> asyncio.Task[None]:
        task = real_create_task(coroutine)
        tasks.append(task)
        return task

    monkeypatch.setattr(hybrid_files, "provider_client", provider)
    monkeypatch.setattr(PlatformFilesClient, "retry", retry)
    monkeypatch.setattr(asyncio, "timeout", controlled_timeout)
    monkeypatch.setattr(asyncio, "create_task", capture_task)
    handler = real_create_task(
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
        await asyncio.wait_for(entered.wait(), timeout=1)
        if cancel_handler:
            handler.cancel()
            with pytest.raises(asyncio.CancelledError):
                await handler
            assert events == []
        if expire == "delete":
            timers[0].reschedule(asyncio.get_running_loop().time())
        else:
            release.set()
        await asyncio.wait_for(reporting.wait(), timeout=1)
        assert len(timers) == 2
        if expire == "report":
            timers[1].reschedule(asyncio.get_running_loop().time())
        else:
            report_release.set()
        await asyncio.wait_for(tasks[0], timeout=1)
        if not cancel_handler:
            await handler
        assert events == [
            "delete-cancelled" if expire == "delete" else "deleted",
            "report-cancelled" if expire == "report" else "reported",
        ]
    finally:
        release.set()
        report_release.set()
        if not handler.done():
            await asyncio.wait_for(handler, timeout=1)
        for task in tasks:
            if not task.done():
                await asyncio.wait_for(task, timeout=1)
