"""Unit tests for the local-filesystem file store."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from gateway.core.config import GatewayConfig
from gateway.services.file_store import LocalDirFileStore, build_file_store


async def _iter(chunks: list[bytes]) -> AsyncIterator[bytes]:
    for chunk in chunks:
        yield chunk


@pytest.mark.asyncio
async def test_put_get_roundtrip(tmp_path: Path) -> None:
    store = LocalDirFileStore(str(tmp_path))
    ref = await store.put("file-abcdef0123", b"hello bytes")
    assert await store.get(ref) == b"hello bytes"


@pytest.mark.asyncio
async def test_put_stream_get_roundtrip(tmp_path: Path) -> None:
    store = LocalDirFileStore(str(tmp_path))
    ref, size = await store.put_stream("file-streamtest01", _iter([b"hello ", b"stream", b"ed bytes"]))
    assert size == len(b"hello streamed bytes")
    assert await store.get(ref) == b"hello streamed bytes"


@pytest.mark.asyncio
async def test_put_stream_handles_empty_chunks(tmp_path: Path) -> None:
    store = LocalDirFileStore(str(tmp_path))
    ref, size = await store.put_stream("file-emptystream1", _iter([]))
    assert size == 0
    assert await store.get(ref) == b""


@pytest.mark.asyncio
async def test_get_stream_yields_all_bytes(tmp_path: Path) -> None:
    store = LocalDirFileStore(str(tmp_path))
    payload = b"x" * (3 * 1024 * 1024 + 17)  # spans multiple 1 MiB read chunks
    ref = await store.put("file-bigstream0001", payload)

    collected = bytearray()
    async for chunk in store.get_stream(ref):
        collected.extend(chunk)
    assert bytes(collected) == payload


@pytest.mark.asyncio
async def test_put_stream_cleans_up_partial_file_on_failure(tmp_path: Path) -> None:
    store = LocalDirFileStore(str(tmp_path))

    async def _failing_chunks() -> AsyncIterator[bytes]:
        yield b"partial data that should not survive"
        raise RuntimeError("simulated upstream failure mid-stream")

    with pytest.raises(RuntimeError, match="simulated upstream failure"):
        await store.put_stream("file-failmidstream", _failing_chunks())

    # No orphaned blob: the ref was never returned, so nothing else could
    # clean this up, which is why put_stream must do it itself.
    shard_dir = tmp_path / "fa"
    leftover = list(shard_dir.glob("*")) if shard_dir.exists() else []
    assert leftover == []


@pytest.mark.asyncio
async def test_put_stream_cleans_up_partial_file_on_cancellation(tmp_path: Path) -> None:
    """A client disconnect mid-upload raises CancelledError, not a plain Exception.

    put_stream's cleanup must run under asyncio.shield so a second cancel()
    can't cut off the unlink before it completes (see PR #380 review).
    """
    store = LocalDirFileStore(str(tmp_path))

    async def _cancelled_chunks() -> AsyncIterator[bytes]:
        yield b"partial data that should not survive cancellation"
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await store.put_stream("file-cancelmidstream", _cancelled_chunks())

    shard_dir = tmp_path / "ca"
    leftover = list(shard_dir.glob("*")) if shard_dir.exists() else []
    assert leftover == []


@pytest.mark.asyncio
async def test_put_stream_cleans_up_partial_file_on_http_exception(tmp_path: Path) -> None:
    """The real production trigger, not just a generic RuntimeError/cancellation.

    _capped_chunks (the route's size-cap wrapper) raises HTTPException(413)
    mid-stream once the upload exceeds the configured limit. Mirrors that
    shape directly here (without importing the route, to keep this a pure
    file_store test) to confirm cleanup covers the actual production path,
    not just the generic failure modes above.
    """
    from fastapi import HTTPException

    store = LocalDirFileStore(str(tmp_path))

    async def _oversized_chunks() -> AsyncIterator[bytes]:
        yield b"x" * 10
        raise HTTPException(status_code=413, detail="File exceeds maximum upload size of 1 MB")

    with pytest.raises(HTTPException):
        await store.put_stream("file-httpexcmidstream", _oversized_chunks())

    shard_dir = tmp_path / "ht"
    leftover = list(shard_dir.glob("*")) if shard_dir.exists() else []
    assert leftover == []


@pytest.mark.asyncio
async def test_delete_removes_empty_shard_dir(tmp_path: Path) -> None:
    """Deleting the only file in a shard dir also removes the now-empty dir.

    Otherwise a zero-byte upload that gets rejected (see the create_file route)
    leaves an empty shard directory behind for every attempt.
    """
    store = LocalDirFileStore(str(tmp_path))
    ref = await store.put("file-lonelyshard1", b"")
    shard_dir = (tmp_path / ref).parent
    assert shard_dir.exists()

    await store.delete(ref)
    assert not shard_dir.exists()


@pytest.mark.asyncio
async def test_delete_keeps_shard_dir_with_other_files(tmp_path: Path) -> None:
    """rmdir must be harmless when the shard still has siblings in it."""
    store = LocalDirFileStore(str(tmp_path))
    # Both ids share the "ab" shard prefix.
    ref1 = await store.put("file-ab111111", b"one")
    ref2 = await store.put("file-ab222222", b"two")
    shard_dir = (tmp_path / ref1).parent

    await store.delete(ref1)
    assert shard_dir.exists()  # ref2 still lives here
    assert await store.get(ref2) == b"two"


@pytest.mark.asyncio
async def test_put_shards_by_prefix(tmp_path: Path) -> None:
    store = LocalDirFileStore(str(tmp_path))
    ref = await store.put("file-ab12cd34", b"x")
    # Sharded under the first two hex chars of the id token.
    assert ref == "ab/file-ab12cd34"
    assert (tmp_path / "ab" / "file-ab12cd34").exists()


@pytest.mark.asyncio
async def test_delete_is_idempotent(tmp_path: Path) -> None:
    store = LocalDirFileStore(str(tmp_path))
    ref = await store.put("file-deadbeef", b"data")
    await store.delete(ref)
    assert not (tmp_path / ref).exists()
    # Deleting again must not raise.
    await store.delete(ref)


@pytest.mark.asyncio
async def test_rejects_path_traversal(tmp_path: Path) -> None:
    store = LocalDirFileStore(str(tmp_path))
    for ref in ("../escape", "../../etc/passwd", "ab/../../escape"):
        with pytest.raises(ValueError, match="escapes the file store root"):
            await store.get(ref)
        with pytest.raises(ValueError, match="escapes the file store root"):
            await store.delete(ref)


def test_build_file_store_local(tmp_path: Path) -> None:
    config = GatewayConfig(files_backend="local", files_local_dir=str(tmp_path))
    assert isinstance(build_file_store(config), LocalDirFileStore)


def test_build_file_store_rejects_unknown_backend() -> None:
    config = GatewayConfig(files_backend="ceph")
    with pytest.raises(ValueError, match="Unsupported files_backend"):
        build_file_store(config)


def test_build_file_store_s3_requires_bucket() -> None:
    config = GatewayConfig(files_backend="s3", files_s3_bucket=None)
    with pytest.raises(ValueError, match="files_s3_bucket is required"):
        build_file_store(config)


def test_build_file_store_s3() -> None:
    from gateway.services.file_store import S3FileStore

    config = GatewayConfig(files_backend="s3", files_s3_bucket="my-bucket")
    assert isinstance(build_file_store(config), S3FileStore)
