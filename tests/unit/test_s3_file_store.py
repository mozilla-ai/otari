"""Unit tests for the S3-compatible file store, mocked with moto (no Docker/MinIO needed).

Real MinIO-via-testcontainers integration coverage is a separate concern (see
tests/integration/), same split as LocalDirFileStore's unit vs integration tests.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Generator

import boto3
import pytest
from moto import mock_aws

from gateway.services.file_store import S3FileStore

_BUCKET = "otari-test-bucket"


async def _iter(chunks: list[bytes]) -> AsyncIterator[bytes]:
    for chunk in chunks:
        yield chunk


@pytest.fixture
def s3_store() -> Generator[S3FileStore, None, None]:
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=_BUCKET)
        yield S3FileStore(bucket=_BUCKET, endpoint_url=None, region="us-east-1")


@pytest.mark.asyncio
async def test_put_get_roundtrip(s3_store: S3FileStore) -> None:
    ref = await s3_store.put("file-abcdef0123", b"hello bytes")
    assert await s3_store.get(ref) == b"hello bytes"


@pytest.mark.asyncio
async def test_put_shards_by_prefix(s3_store: S3FileStore) -> None:
    ref = await s3_store.put("file-ab12cd34", b"x")
    assert ref == "ab/file-ab12cd34"


@pytest.mark.asyncio
async def test_put_stream_get_roundtrip(s3_store: S3FileStore) -> None:
    ref, size = await s3_store.put_stream("file-streamtest01", _iter([b"hello ", b"stream", b"ed bytes"]))
    assert size == len(b"hello streamed bytes")
    assert await s3_store.get(ref) == b"hello streamed bytes"


@pytest.mark.asyncio
async def test_put_stream_handles_empty_chunks(s3_store: S3FileStore) -> None:
    ref, size = await s3_store.put_stream("file-emptystream1", _iter([]))
    assert size == 0
    assert await s3_store.get(ref) == b""


@pytest.mark.asyncio
async def test_get_stream_yields_all_bytes(s3_store: S3FileStore) -> None:
    payload = b"x" * (3 * 1024 * 1024 + 17)  # spans multiple 1 MiB read chunks
    ref = await s3_store.put("file-bigstream0001", payload)

    collected = bytearray()
    async for chunk in s3_store.get_stream(ref):
        collected.extend(chunk)
    assert bytes(collected) == payload


@pytest.mark.asyncio
async def test_delete_removes_object(s3_store: S3FileStore) -> None:
    ref = await s3_store.put("file-deadbeef", b"data")
    await s3_store.delete(ref)
    with pytest.raises(Exception, match="Not Found|NoSuchKey|404"):
        await s3_store.get(ref)


@pytest.mark.asyncio
async def test_delete_is_idempotent(s3_store: S3FileStore) -> None:
    ref = await s3_store.put("file-deadbeef", b"data")
    await s3_store.delete(ref)
    # S3 DeleteObject is idempotent by design: deleting an absent key is not an error.
    await s3_store.delete(ref)


@pytest.mark.asyncio
async def test_put_stream_does_not_upload_on_failure_before_upload_starts(s3_store: S3FileStore) -> None:
    """If the chunk source fails while we're still spooling, nothing has been
    sent to S3 yet (upload_fileobj hasn't been called), so there's nothing to
    abort, just the local spool file to close (covered by the shielded finally).
    """

    async def _failing_chunks() -> AsyncIterator[bytes]:
        yield b"partial data that should not survive"
        msg = "simulated upstream failure mid-stream"
        raise RuntimeError(msg)

    with pytest.raises(RuntimeError, match="simulated upstream failure"):
        await s3_store.put_stream("file-failmidstream", _failing_chunks())

    client = s3_store._client  # noqa: SLF001 - inspecting internal state to assert nothing was uploaded
    listing = await asyncio.to_thread(client.list_objects_v2, Bucket=_BUCKET, Prefix="fa/")
    assert listing.get("KeyCount", 0) == 0


@pytest.mark.asyncio
async def test_put_stream_aborts_multipart_upload_on_mid_upload_failure(
    s3_store: S3FileStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies the claim made in the #156 design comment: upload_fileobj's
    multipart handling aborts an incomplete multipart upload on failure, so
    S3FileStore doesn't need to call abort_multipart_upload itself.

    Forces multipart (default threshold is 8 MiB) and injects a failure on the
    second part upload to simulate a network error partway through.
    """
    client = s3_store._client  # noqa: SLF001 - need the raw client to patch upload_part
    original_upload_part = client.upload_part

    call_count = {"value": 0}

    def _flaky_upload_part(**kwargs: object) -> object:
        call_count["value"] += 1
        if call_count["value"] == 2:
            msg = "simulated network failure on part 2"
            raise RuntimeError(msg)
        return original_upload_part(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(client, "upload_part", _flaky_upload_part)

    payload = b"y" * (9 * 1024 * 1024)  # > 8 MiB default multipart_threshold
    with pytest.raises(RuntimeError, match="simulated network failure"):
        await s3_store.put_stream("file-multipartfail1", _iter([payload]))

    assert call_count["value"] >= 2  # confirms multipart (and our failure point) actually engaged

    lingering = await asyncio.to_thread(client.list_multipart_uploads, Bucket=_BUCKET)
    assert lingering.get("Uploads", []) == []
