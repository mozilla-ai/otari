"""Unit tests for the local-filesystem file store."""

from __future__ import annotations

from pathlib import Path

import pytest

from gateway.core.config import GatewayConfig
from gateway.services.file_store import LocalDirFileStore, build_file_store


@pytest.mark.asyncio
async def test_put_get_roundtrip(tmp_path: Path) -> None:
    store = LocalDirFileStore(str(tmp_path))
    ref = await store.put("file-abcdef0123", b"hello bytes")
    assert await store.get(ref) == b"hello bytes"


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
