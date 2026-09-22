"""Tests ``FsspecFileStore`` against a real S3-compatible store, SeaweedFS, through s3fs."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any

import fsspec
import pytest
from testcontainers.core.container import DockerContainer
from testcontainers.core.wait_strategies import HttpWaitStrategy

from gateway.services.file_store import FsspecFileStore

_SEAWEEDFS_IMAGE = "docker.io/chrislusf/seaweedfs:4.47"
_S3_PORT = 8333
_ACCESS_KEY = "otari-test"
_SECRET_KEY = "otari-test-secret"
_BUCKET = "otari-files"
_ROOT = f"{_BUCKET}/uploads"
# This is S3's minimum part size, so a payload of a few blocks uploads in several parts.
_BLOCK_SIZE = 5 * 1024 * 1024


async def _iter(chunks: list[bytes]) -> AsyncIterator[bytes]:
    for chunk in chunks:
        yield chunk


@pytest.fixture(scope="module")
def s3_options() -> Iterator[dict[str, Any]]:
    container = (
        DockerContainer(_SEAWEEDFS_IMAGE)
        .with_command("mini -dir=/data -admin.ui=false -webdav=false -s3.port.iceberg=0 -s3.port.lance=0")
        .with_env("AWS_ACCESS_KEY_ID", _ACCESS_KEY)
        .with_env("AWS_SECRET_ACCESS_KEY", _SECRET_KEY)
        .with_exposed_ports(_S3_PORT)
        .waiting_for(HttpWaitStrategy(_S3_PORT, "/healthz"))
    )
    with container:
        endpoint = f"http://{container.get_container_host_ip()}:{container.get_exposed_port(_S3_PORT)}"
        options: dict[str, Any] = {
            "key": _ACCESS_KEY,
            "secret": _SECRET_KEY,
            "endpoint_url": endpoint,
            "default_block_size": _BLOCK_SIZE,
        }
        fsspec.filesystem("s3", **options).mkdir(_BUCKET)
        yield options


@pytest.fixture
def store(s3_options: dict[str, Any]) -> FsspecFileStore:
    return FsspecFileStore(f"s3://{_ROOT}", s3_options)


@pytest.mark.asyncio
async def test_put_writes_the_object_under_the_root(store: FsspecFileStore, s3_options: dict[str, Any]) -> None:
    ref = await store.put("file-abcdef0123", b"hello bytes")

    assert ref == "ab/file-abcdef0123"
    assert await store.get(ref) == b"hello bytes"
    assert fsspec.filesystem("s3", **s3_options).cat_file(f"{_ROOT}/{ref}") == b"hello bytes"


@pytest.mark.asyncio
async def test_put_stream_and_get_stream_roundtrip_across_parts(store: FsspecFileStore) -> None:
    payload = bytes(range(256)) * (2 * _BLOCK_SIZE // 256) + b"tail"
    chunks = [payload[i : i + 1024 * 1024] for i in range(0, len(payload), 1024 * 1024)]

    ref, size = await store.put_stream("file-streamtest01", _iter(chunks))

    assert size == len(payload)
    collected = bytearray()
    async for chunk in store.get_stream(ref):
        collected.extend(chunk)
    assert bytes(collected) == payload


@pytest.mark.asyncio
async def test_delete_removes_the_object_and_is_idempotent(store: FsspecFileStore) -> None:
    ref = await store.put("file-deleteme0001", b"x")

    await store.delete(ref)
    await store.delete(ref)

    with pytest.raises(FileNotFoundError):
        await store.get(ref)


@pytest.mark.asyncio
async def test_missing_object_is_file_not_found(store: FsspecFileStore) -> None:
    with pytest.raises(FileNotFoundError):
        await store.get("no/file-nope")
    with pytest.raises(FileNotFoundError):
        async for _ in store.get_stream("no/file-nope"):
            pass


@pytest.mark.asyncio
async def test_failed_put_stream_leaves_no_object(store: FsspecFileStore) -> None:
    async def _failing() -> AsyncIterator[bytes]:
        yield b"x" * _BLOCK_SIZE
        raise RuntimeError("client went away")

    with pytest.raises(RuntimeError, match="client went away"):
        await store.put_stream("file-partial00001", _failing())

    with pytest.raises(FileNotFoundError):
        await store.get("pa/file-partial00001")
