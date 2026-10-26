"""Unit tests for the S3-compatible file store, mocked with moto."""

from __future__ import annotations

import asyncio
import sys
import threading
from collections.abc import AsyncIterator, Generator
from typing import IO

import boto3
import pytest
from moto import mock_aws

from gateway.adapters.file_storage_adapter import S3FileStore

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
    ref = await s3_store.allocate("file-abcdef0123")
    await s3_store.put(ref, b"hello bytes")
    assert await s3_store.get(ref) == b"hello bytes"


@pytest.mark.asyncio
async def test_put_shards_by_prefix(s3_store: S3FileStore) -> None:
    ref = await s3_store.allocate("file-ab12cd34")
    await s3_store.put(ref, b"x")
    assert ref == "ab/file-ab12cd34"


@pytest.mark.asyncio
async def test_put_stream_get_roundtrip(s3_store: S3FileStore) -> None:
    ref = await s3_store.allocate("file-streamtest01")
    size = await s3_store.put_stream(ref, _iter([b"hello ", b"stream", b"ed bytes"]))
    assert size == len(b"hello streamed bytes")
    assert await s3_store.get(ref) == b"hello streamed bytes"


@pytest.mark.asyncio
async def test_put_stream_handles_empty_chunks(s3_store: S3FileStore) -> None:
    ref = await s3_store.allocate("file-emptystream1")
    size = await s3_store.put_stream(ref, _iter([]))
    assert size == 0
    assert await s3_store.get(ref) == b""


@pytest.mark.asyncio
async def test_get_stream_yields_all_bytes(s3_store: S3FileStore) -> None:
    payload = b"x" * (3 * 1024 * 1024 + 17)  # spans multiple 1 MiB read chunks
    ref = await s3_store.allocate("file-bigstream0001")
    await s3_store.put(ref, payload)

    collected = bytearray()
    async for chunk in s3_store.get_stream(ref):
        collected.extend(chunk)
    assert bytes(collected) == payload


@pytest.mark.asyncio
async def test_delete_removes_object(s3_store: S3FileStore) -> None:
    ref = await s3_store.allocate("file-deadbeef")
    await s3_store.put(ref, b"data")
    await s3_store.delete(ref)
    # S3FileStore translates a missing object into FileNotFoundError, mirroring
    # the local backend so route callers can keep catching OSError.
    with pytest.raises(FileNotFoundError):
        await s3_store.get(ref)


@pytest.mark.asyncio
async def test_get_stream_missing_key_raises_file_not_found(s3_store: S3FileStore) -> None:
    # get_stream is primed by the download route before the OSError handler;
    # a missing object must surface as FileNotFoundError (an OSError), not a
    # raw botocore ClientError that would escape that handler.
    with pytest.raises(FileNotFoundError):
        async for _ in s3_store.get_stream("no/such-key"):
            pass


@pytest.mark.asyncio
async def test_delete_failure_raises_oserror(s3_store: S3FileStore, monkeypatch: pytest.MonkeyPatch) -> None:
    # A non-404 backend failure during delete must map to OSError so the delete
    # route's best-effort `except OSError` catches it and a committed
    # soft-delete never turns into a 500.
    from botocore.exceptions import ClientError

    def _boom(**kwargs: object) -> object:
        raise ClientError({"Error": {"Code": "AccessDenied", "Message": "nope"}}, "DeleteObject")

    monkeypatch.setattr(s3_store._client, "delete_object", _boom)  # noqa: SLF001 - inject a backend failure
    with pytest.raises(OSError):  # noqa: PT011 - translated botocore error, message not asserted
        await s3_store.delete("ab/file-whatever")


@pytest.mark.asyncio
async def test_delete_is_idempotent(s3_store: S3FileStore) -> None:
    ref = await s3_store.allocate("file-deadbeef")
    await s3_store.put(ref, b"data")
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

    ref = await s3_store.allocate("file-failmidstream")
    with pytest.raises(RuntimeError, match="simulated upstream failure"):
        await s3_store.put_stream(ref, _failing_chunks())

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
    ref = await s3_store.allocate("file-multipartfail1")
    with pytest.raises(RuntimeError, match="simulated network failure"):
        await s3_store.put_stream(ref, _iter([payload]))

    assert call_count["value"] >= 2  # confirms multipart (and our failure point) actually engaged

    lingering = await asyncio.to_thread(client.list_multipart_uploads, Bucket=_BUCKET)
    assert lingering.get("Uploads", []) == []


@pytest.mark.asyncio
async def test_put_stream_removes_orphaned_upload_when_cancelled_after_success(
    s3_store: S3FileStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """upload_fileobj runs in a worker thread that to_thread cannot interrupt:
    if *our* await gets cancelled while that thread is still working, the
    upload can still complete successfully afterward. Confirms put_stream
    notices that and deletes the now-orphaned object instead of silently
    leaving it in the bucket with nothing that will ever reference it.

    Uses threading.Event gates instead of sleep-based timing so the sequence
    (upload thread actually started -> cancel -> release it to finish) is
    deterministic rather than a timing guess.
    """
    client = s3_store._client  # noqa: SLF001 - need the raw client to patch upload_fileobj
    original_upload_fileobj = client.upload_fileobj

    upload_started = threading.Event()
    release_upload = threading.Event()

    def _gated_upload_fileobj(fileobj: IO[bytes], bucket: str, key: str) -> None:
        upload_started.set()
        release_upload.wait(timeout=5)
        original_upload_fileobj(fileobj, bucket, key)

    monkeypatch.setattr(client, "upload_fileobj", _gated_upload_fileobj)

    ref = await s3_store.allocate("file-orphancheck01")
    task = asyncio.ensure_future(s3_store.put_stream(ref, _iter([b"data"])))
    # Don't cancel until the upload thread is truly running: an ignored
    # timeout here would let the test proceed anyway, cancelling before the
    # race it's meant to reproduce even started, silently.
    assert await asyncio.to_thread(upload_started.wait, 5), "upload thread never started"
    task.cancel()
    release_upload.set()  # let the (already-committed-to) background upload finish

    with pytest.raises(asyncio.CancelledError):
        await task

    listing = await asyncio.to_thread(client.list_objects_v2, Bucket=_BUCKET, Prefix="or/")
    assert listing.get("KeyCount", 0) == 0


def _client_error(operation: str) -> Exception:
    from botocore.exceptions import ClientError

    return ClientError({"Error": {"Code": "AccessDenied", "Message": "nope"}}, operation)


@pytest.mark.asyncio
async def test_put_failure_raises_oserror(s3_store: S3FileStore, monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(**kwargs: object) -> object:
        raise _client_error("PutObject")

    monkeypatch.setattr(s3_store._client, "put_object", _boom)  # noqa: SLF001 - inject a backend failure
    ref = await s3_store.allocate("file-putfail01")
    with pytest.raises(OSError):  # noqa: PT011 - translated botocore error, message not asserted
        await s3_store.put(ref, b"data")


@pytest.mark.asyncio
async def test_put_stream_upload_failure_raises_oserror(s3_store: S3FileStore, monkeypatch: pytest.MonkeyPatch) -> None:
    from boto3.exceptions import S3UploadFailedError

    def _boom(*args: object, **kwargs: object) -> None:
        msg = "Failed to upload"
        raise S3UploadFailedError(msg)

    monkeypatch.setattr(s3_store._client, "upload_fileobj", _boom)  # noqa: SLF001 - inject a backend failure
    ref = await s3_store.allocate("file-uploadfail01")
    with pytest.raises(OSError):  # noqa: PT011 - translated boto3 error, message not asserted
        await s3_store.put_stream(ref, _iter([b"data"]))


@pytest.mark.asyncio
async def test_get_stream_read_failure_raises_oserror(s3_store: S3FileStore, monkeypatch: pytest.MonkeyPatch) -> None:
    from botocore.exceptions import ResponseStreamingError

    ref = await s3_store.allocate("file-readfail01")
    await s3_store.put(ref, b"data")

    class _BrokenBody:
        def read(self, amt: int | None = None) -> bytes:
            raise ResponseStreamingError(error="connection reset")

        def close(self) -> None:
            pass

    monkeypatch.setattr(s3_store._client, "get_object", lambda **kw: {"Body": _BrokenBody()})  # noqa: SLF001
    with pytest.raises(OSError):  # noqa: PT011 - translated botocore error, message not asserted
        async for _ in s3_store.get_stream(ref):
            pass


@pytest.mark.asyncio
async def test_get_stream_tls_failure_raises_oserror(s3_store: S3FileStore, monkeypatch: pytest.MonkeyPatch) -> None:
    """botocore's StreamingBody.read lets urllib3's SSLError through unconverted."""
    from urllib3.exceptions import SSLError

    ref = await s3_store.allocate("file-tlsfail01")
    await s3_store.put(ref, b"data")

    class _BrokenBody:
        def read(self, amt: int | None = None) -> bytes:
            raise SSLError("bad record mac")

        def close(self) -> None:
            pass

    monkeypatch.setattr(s3_store._client, "get_object", lambda **kw: {"Body": _BrokenBody()})  # noqa: SLF001
    with pytest.raises(OSError):  # noqa: PT011 - translated urllib3 error, message not asserted
        async for _ in s3_store.get_stream(ref):
            pass


@pytest.mark.asyncio
async def test_put_stream_cleanup_survives_a_second_cancellation(
    s3_store: S3FileStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second cancel while put_stream waits for the in-flight upload must not skip the compensating delete."""
    client = s3_store._client  # noqa: SLF001 - need the raw client to patch upload_fileobj
    original_upload_fileobj = client.upload_fileobj
    upload_started = threading.Event()
    release_upload = threading.Event()
    upload_done = threading.Event()

    def _gated_upload_fileobj(fileobj: IO[bytes], bucket: str, key: str) -> None:
        upload_started.set()
        release_upload.wait(timeout=5)
        original_upload_fileobj(fileobj, bucket, key)
        upload_done.set()

    monkeypatch.setattr(client, "upload_fileobj", _gated_upload_fileobj)

    ref = await s3_store.allocate("file-twocancel01")
    task = asyncio.ensure_future(s3_store.put_stream(ref, _iter([b"data"])))
    assert await asyncio.to_thread(upload_started.wait, 5), "upload thread never started"
    task.cancel()
    await asyncio.sleep(0)  # let put_stream enter its cleanup wait
    task.cancel()  # the second cancellation, while cleanup waits on the upload
    release_upload.set()

    with pytest.raises(asyncio.CancelledError):
        await task

    # Without the fix the task ends before the upload lands; list only once it has.
    assert await asyncio.to_thread(upload_done.wait, 5), "upload never finished"
    listing = await asyncio.to_thread(client.list_objects_v2, Bucket=_BUCKET, Prefix="tw/")
    assert listing.get("KeyCount", 0) == 0


@pytest.mark.asyncio
async def test_put_stream_logs_when_removing_the_orphan_fails(
    s3_store: S3FileStore, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    client = s3_store._client  # noqa: SLF001 - need the raw client to patch upload_fileobj
    original_upload_fileobj = client.upload_fileobj
    upload_started = threading.Event()
    release_upload = threading.Event()

    def _gated_upload_fileobj(fileobj: IO[bytes], bucket: str, key: str) -> None:
        upload_started.set()
        release_upload.wait(timeout=5)
        original_upload_fileobj(fileobj, bucket, key)

    def _boom(**kwargs: object) -> object:
        raise _client_error("DeleteObject")

    monkeypatch.setattr(client, "upload_fileobj", _gated_upload_fileobj)
    monkeypatch.setattr(client, "delete_object", _boom)

    ref = await s3_store.allocate("file-deletefail01")
    task = asyncio.ensure_future(s3_store.put_stream(ref, _iter([b"data"])))
    assert await asyncio.to_thread(upload_started.wait, 5), "upload thread never started"
    task.cancel()
    release_upload.set()

    with caplog.at_level("WARNING"), pytest.raises(asyncio.CancelledError):
        await task

    assert "failed to remove orphaned upload" in caplog.text


@pytest.mark.asyncio
async def test_put_stream_cleanup_keeps_an_outer_timeout_intact(
    s3_store: S3FileStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """asyncio.timeout() must still see its deadline: the cancel it sent is not swallowed by the cleanup."""
    client = s3_store._client  # noqa: SLF001 - need the raw client to patch upload_fileobj
    original_upload_fileobj = client.upload_fileobj
    upload_started = threading.Event()
    release_upload = threading.Event()

    def _gated_upload_fileobj(fileobj: IO[bytes], bucket: str, key: str) -> None:
        upload_started.set()
        release_upload.wait(timeout=5)
        original_upload_fileobj(fileobj, bucket, key)

    monkeypatch.setattr(client, "upload_fileobj", _gated_upload_fileobj)
    ref = await s3_store.allocate("file-timeout01")

    async def _release_after_deadline() -> None:
        await asyncio.to_thread(upload_started.wait, 5)
        await asyncio.sleep(0.3)
        release_upload.set()

    releaser = asyncio.ensure_future(_release_after_deadline())
    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.1):
            await s3_store.put_stream(ref, _iter([b"data"]))
    await releaser

    listing = await asyncio.to_thread(client.list_objects_v2, Bucket=_BUCKET, Prefix="ti/")
    assert listing.get("KeyCount", 0) == 0


@pytest.mark.asyncio
async def test_put_stream_cleanup_gives_up_on_a_stuck_upload(
    s3_store: S3FileStore, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A thread that never finishes must not keep a cancelled request alive forever."""
    monkeypatch.setattr("gateway.adapters.file_storage_adapter._ORPHAN_CLEANUP_TIMEOUT_SECONDS", 0.2)
    client = s3_store._client  # noqa: SLF001 - need the raw client to patch upload_fileobj
    upload_started = threading.Event()
    release_upload = threading.Event()

    def _stuck_upload_fileobj(fileobj: IO[bytes], bucket: str, key: str) -> None:
        upload_started.set()
        release_upload.wait(timeout=5)

    monkeypatch.setattr(client, "upload_fileobj", _stuck_upload_fileobj)
    ref = await s3_store.allocate("file-stuck01")
    task = asyncio.ensure_future(s3_store.put_stream(ref, _iter([b"data"])))
    assert await asyncio.to_thread(upload_started.wait, 5), "upload thread never started"
    task.cancel()

    try:
        with caplog.at_level("WARNING"), pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=3)
    finally:
        release_upload.set()

    assert "gave up waiting for upload" in caplog.text


def test_missing_boto3_names_the_extra_to_install(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "boto3", None)

    with pytest.raises(ImportError, match=r"uv sync --extra s3"):
        S3FileStore(bucket=_BUCKET, endpoint_url=None, region=None)
