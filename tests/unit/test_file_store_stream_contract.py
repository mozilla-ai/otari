"""Shared failure contract for every streamed file-storage adapter."""

from __future__ import annotations

import asyncio
import threading
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Generator
from dataclasses import dataclass
from pathlib import Path
from typing import IO

import pytest

from gateway.adapters.file_storage_adapter import FsspecFileStore, LocalDirFileStore, S3FileStore
from gateway.exceptions.files_exceptions import UploadTooLargeError
from gateway.ports.file_storage_port import FileStoragePort
from gateway.services.files._service import _capped


@dataclass(frozen=True)
class StoreCase:
    store: FileStoragePort
    assert_empty: Callable[[], Awaitable[None]]


@pytest.fixture(params=("local", "fsspec", "s3"))
def store_case(request: pytest.FixtureRequest, tmp_path: Path) -> Generator[StoreCase, None, None]:
    store: FileStoragePort
    if request.param == "local":
        store = LocalDirFileStore(str(tmp_path))

        async def assert_empty() -> None:
            assert not [path for path in tmp_path.rglob("*") if path.is_file()]

        yield StoreCase(store, assert_empty)
        return

    if request.param == "fsspec":
        fsspec = pytest.importorskip("fsspec")
        filesystem = fsspec.filesystem("memory")
        root = f"otari-stream-contract-{uuid.uuid4().hex}"
        store = FsspecFileStore(f"memory://{root}")

        async def assert_empty() -> None:
            assert filesystem.find(root) == []

        try:
            yield StoreCase(store, assert_empty)
        finally:
            filesystem.rm(root, recursive=True)
        return

    boto3 = pytest.importorskip("boto3")
    moto = pytest.importorskip("moto")
    bucket = f"otari-stream-contract-{uuid.uuid4().hex}"
    with moto.mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=bucket)
        store = S3FileStore(bucket=bucket, endpoint_url=None, region="us-east-1")

        async def assert_empty() -> None:
            listing = await asyncio.to_thread(client.list_objects_v2, Bucket=bucket)
            assert listing.get("KeyCount", 0) == 0
            uploads = await asyncio.to_thread(client.list_multipart_uploads, Bucket=bucket)
            assert uploads.get("Uploads", []) == []

        yield StoreCase(store, assert_empty)


@pytest.mark.asyncio
async def test_failed_source_leaves_no_object(store_case: StoreCase) -> None:
    async def failing_chunks() -> AsyncIterator[bytes]:
        yield b"partial bytes"
        raise RuntimeError("source failed")

    with pytest.raises(RuntimeError, match="source failed"):
        await store_case.store.put_stream(f"file-{uuid.uuid4().hex}", failing_chunks())

    await store_case.assert_empty()


@pytest.mark.asyncio
async def test_size_refusal_leaves_no_object(store_case: StoreCase) -> None:
    async def oversized_chunks() -> AsyncIterator[bytes]:
        yield b"1234"
        yield b"5678"

    with pytest.raises(UploadTooLargeError):
        await store_case.store.put_stream(f"file-{uuid.uuid4().hex}", _capped(oversized_chunks(), 5))

    await store_case.assert_empty()


@pytest.mark.asyncio
async def test_cancellation_mid_stream_leaves_no_object(store_case: StoreCase) -> None:
    started = asyncio.Event()

    async def blocked_chunks() -> AsyncIterator[bytes]:
        yield b"partial bytes"
        started.set()
        await asyncio.Event().wait()

    task = asyncio.create_task(store_case.store.put_stream(f"file-{uuid.uuid4().hex}", blocked_chunks()))
    await asyncio.wait_for(started.wait(), timeout=5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    await store_case.assert_empty()


@pytest.mark.asyncio
async def test_cancellation_during_open_closes_handle_and_removes_temporary_object(
    store_case: StoreCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    if isinstance(store_case.store, S3FileStore):
        pytest.skip("S3 uploads use a different open path")

    open_started = threading.Event()
    release_open = threading.Event()
    opened_handles: list[IO[bytes]] = []
    closed_handles: list[IO[bytes]] = []

    def hold_open(handle: IO[bytes]) -> IO[bytes]:
        close = handle.close

        def tracked_close() -> None:
            close()
            closed_handles.append(handle)

        setattr(handle, "close", tracked_close)
        opened_handles.append(handle)
        open_started.set()
        if not release_open.wait(timeout=5):
            raise TimeoutError("test did not release the open operation")
        return handle

    if isinstance(store_case.store, LocalDirFileStore):
        original_open = Path.open

        def blocked_local_open(path: Path, mode: str = "r") -> IO[bytes]:
            return hold_open(original_open(path, mode))

        monkeypatch.setattr(Path, "open", blocked_local_open)
    else:
        assert isinstance(store_case.store, FsspecFileStore)
        original_open = store_case.store._fs.open

        def blocked_fsspec_open(path: str, mode: str = "rb") -> IO[bytes]:
            return hold_open(original_open(path, mode))

        monkeypatch.setattr(store_case.store._fs, "open", blocked_fsspec_open)

    async def chunks() -> AsyncIterator[bytes]:
        yield b"unreached"

    task = asyncio.create_task(store_case.store.put_stream(f"file-{uuid.uuid4().hex}", chunks()))
    try:
        assert await asyncio.wait_for(asyncio.to_thread(open_started.wait), timeout=5)
        assert task.cancel()
    finally:
        release_open.set()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert len(opened_handles) == 1
    assert closed_handles == opened_handles
    if isinstance(store_case.store, LocalDirFileStore):
        assert opened_handles[0].closed
    await store_case.assert_empty()


@pytest.mark.asyncio
async def test_cancellation_during_publication_removes_published_object(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = LocalDirFileStore(str(tmp_path))
    publication_started = threading.Event()
    release_publication = threading.Event()
    original_replace = Path.replace

    def blocked_replace(source: Path, target: Path) -> Path:
        published = original_replace(source, target)
        publication_started.set()
        if not release_publication.wait(timeout=5):
            raise TimeoutError("test did not release the publication operation")
        return published

    monkeypatch.setattr(Path, "replace", blocked_replace)

    async def chunks() -> AsyncIterator[bytes]:
        yield b"complete upload"

    task = asyncio.create_task(store.put_stream("file-cancel-publish", chunks()))
    try:
        assert await asyncio.wait_for(asyncio.to_thread(publication_started.wait), timeout=5)
        assert task.cancel()
    finally:
        release_publication.set()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert not [path for path in tmp_path.rglob("*") if path.is_file()]


@pytest.mark.asyncio
async def test_cancellation_cleanup_preserves_a_later_same_key_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = LocalDirFileStore(str(tmp_path))
    first_publication_started = threading.Event()
    release_first_publication = threading.Event()
    second_publication_finished = threading.Event()
    second_handle_closed = threading.Event()
    original_open = Path.open
    original_replace = Path.replace

    def tracked_open(
        path: Path,
        mode: str = "r",
        buffering: int = -1,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
    ) -> IO[bytes]:
        handle = original_open(path, mode, buffering, encoding, errors, newline)
        if mode == "xb":
            write = handle.write
            close = handle.close
            wrote_replacement = False

            def tracked_write(chunk: bytes) -> int:
                nonlocal wrote_replacement
                result = write(chunk)
                if chunk == b"replacement":
                    wrote_replacement = True
                return result

            def tracked_close() -> None:
                close()
                if wrote_replacement:
                    second_handle_closed.set()

            setattr(handle, "write", tracked_write)
            setattr(handle, "close", tracked_close)
        return handle

    def block_first_replace(source: Path, target: Path) -> Path:
        content = source.read_bytes()
        published = original_replace(source, target)
        if content == b"cancelled":
            first_publication_started.set()
            if not release_first_publication.wait(timeout=5):
                raise TimeoutError("test did not release the first publication")
        elif content == b"replacement":
            second_publication_finished.set()
        return published

    monkeypatch.setattr(Path, "open", tracked_open)
    monkeypatch.setattr(Path, "replace", block_first_replace)

    async def chunks(content: bytes) -> AsyncIterator[bytes]:
        yield content

    first_task = asyncio.create_task(store.put_stream("file-cancel-race", chunks(b"cancelled")))
    second_task: asyncio.Task[tuple[str, int]] | None = None
    try:
        assert await asyncio.wait_for(asyncio.to_thread(first_publication_started.wait), timeout=5)
        assert first_task.cancel()
        second_task = asyncio.create_task(store.put_stream("file-cancel-race", chunks(b"replacement")))
        assert await asyncio.wait_for(asyncio.to_thread(second_handle_closed.wait), timeout=5)
        await asyncio.sleep(0.1)
        assert not second_publication_finished.is_set()
    finally:
        release_first_publication.set()

    with pytest.raises(asyncio.CancelledError):
        await first_task

    assert second_task is not None
    ref, total = await second_task
    assert total == len(b"replacement")
    assert second_publication_finished.is_set()
    assert await store.get(ref) == b"replacement"
