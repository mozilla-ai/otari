"""Unit tests for the fsspec-backed file store.

Runs on fsspec's built-in ``memory://`` and ``file://`` filesystems, so the
suite needs no cloud implementation package and no network. What it proves is
the adapter's own contract (refs, streaming, cleanup, error translation); the
cloud implementations are fsspec's to keep working.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import fsspec
import pytest

from gateway.core.config import GatewayConfig
from gateway.services.file_store import FsspecFileStore, build_file_store


async def _iter(chunks: list[bytes]) -> AsyncIterator[bytes]:
    for chunk in chunks:
        yield chunk


@pytest.fixture
def memory_root() -> str:
    # The memory filesystem is process-global; give each test its own prefix
    # and clear it afterwards so one test's blobs never show up in another.
    root = "memory://otari-test"
    fs = fsspec.filesystem("memory")
    if fs.exists("otari-test"):
        fs.rm("otari-test", recursive=True)
    return root


@pytest.mark.asyncio
async def test_put_get_roundtrip(memory_root: str) -> None:
    store = FsspecFileStore(memory_root)
    ref = await store.put("file-abcdef0123", b"hello bytes")
    assert ref == "ab/file-abcdef0123"
    assert await store.get(ref) == b"hello bytes"


@pytest.mark.asyncio
async def test_put_stream_and_get_stream_roundtrip(memory_root: str) -> None:
    store = FsspecFileStore(memory_root)
    payload = b"x" * (2 * 1024 * 1024 + 5)
    ref, size = await store.put_stream("file-streamtest01", _iter([payload[:1000], payload[1000:]]))
    assert size == len(payload)
    collected = bytearray()
    async for chunk in store.get_stream(ref):
        collected.extend(chunk)
    assert bytes(collected) == payload


@pytest.mark.asyncio
async def test_put_stream_removes_partial_blob_on_failure(memory_root: str) -> None:
    store = FsspecFileStore(memory_root)

    async def _failing() -> AsyncIterator[bytes]:
        yield b"partial"
        raise RuntimeError("client went away")

    with pytest.raises(RuntimeError):
        await store.put_stream("file-partial00001", _failing())
    assert not fsspec.filesystem("memory").exists("otari-test/pa/file-partial00001")


@pytest.mark.asyncio
async def test_put_stream_removes_partial_blob_on_cancellation(memory_root: str) -> None:
    store = FsspecFileStore(memory_root)
    started = asyncio.Event()

    async def _slow() -> AsyncIterator[bytes]:
        yield b"first"
        started.set()
        await asyncio.sleep(30)
        yield b"never"

    task = asyncio.create_task(store.put_stream("file-cancel000001", _slow()))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not fsspec.filesystem("memory").exists("otari-test/ca/file-cancel000001")


@pytest.mark.asyncio
async def test_missing_blob_is_file_not_found(memory_root: str) -> None:
    store = FsspecFileStore(memory_root)
    with pytest.raises(FileNotFoundError):
        await store.get("no/file-nope")
    with pytest.raises(FileNotFoundError):
        async for _ in store.get_stream("no/file-nope"):
            pass


@pytest.mark.asyncio
async def test_delete_is_idempotent(memory_root: str) -> None:
    store = FsspecFileStore(memory_root)
    ref = await store.put("file-deleteme0001", b"x")
    await store.delete(ref)
    await store.delete(ref)
    with pytest.raises(FileNotFoundError):
        await store.get(ref)


@pytest.mark.asyncio
async def test_rejects_refs_that_could_leave_the_root(memory_root: str) -> None:
    store = FsspecFileStore(memory_root)
    for bad in ("../escape", "/absolute", "a//b", "a/./b", ""):
        with pytest.raises(ValueError):
            await store.get(bad)


@pytest.mark.asyncio
async def test_backend_client_errors_become_oserror(memory_root: str) -> None:
    store = FsspecFileStore(memory_root)

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("some client's own exception class")

    store._fs.cat_file = _boom
    with pytest.raises(OSError, match="fsspec operation failed"):
        await store.get("ab/file-abcdef0123")


@pytest.mark.asyncio
async def test_local_file_protocol_writes_under_the_root(tmp_path: Path) -> None:
    store = FsspecFileStore(f"file://{tmp_path}")
    ref = await store.put("file-abcdef0123", b"on disk")
    assert (tmp_path / "ab" / "file-abcdef0123").read_bytes() == b"on disk"
    assert await store.get(ref) == b"on disk"


def test_build_file_store_fsspec_requires_url() -> None:
    cfg = GatewayConfig(files_backend="fsspec")
    with pytest.raises(ValueError, match="files_url"):
        build_file_store(cfg)


def test_build_file_store_fsspec(tmp_path: Path) -> None:
    cfg = GatewayConfig(
        files_backend="fsspec", files_url=f"file://{tmp_path}", files_storage_options={"auto_mkdir": True}
    )
    assert isinstance(build_file_store(cfg), FsspecFileStore)
