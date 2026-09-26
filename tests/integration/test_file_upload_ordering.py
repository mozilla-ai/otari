"""An upload whose bytes never complete stays reachable, and the sweep reclaims it.

A refused upload gives its row and its bytes back at once, but both halves of
that are best effort. What covers a cleanup the store refuses is the row, which
is written before the write begins and so names whatever the write left. The
sweep walks those rows once they are past their grace, and leaves alone the ones
that may still be receiving their bytes.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.adapters.file_storage_adapter import LocalDirFileStore
from gateway.core.config import GatewayConfig
from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions.files_exceptions import FileNotServedError, FileStorageError, UploadTooLargeError
from gateway.models.files import FileObject
from gateway.models.users import User as SpendUser
from gateway.repositories.files import FileRepositories
from gateway.services.files import FileScope, FileService, NewFile, SweepBatch

from .tenancy_helpers import create_member, create_organization, create_workspace

pytestmark = pytest.mark.asyncio

_USER_ID = "user-file-ordering"
_CEILING = 8


class _LeakyStore(LocalDirFileStore):
    """A local store that keeps what a failed write left, and refuses to delete it.

    Removing a partial write is best effort on both sides of the port, so this
    is the case a row has to cover: the bytes are reachable only if something
    names them. Set ``deletable`` once the refusal has passed.
    """

    def __init__(self, root: Path) -> None:
        super().__init__(str(root))
        self.root = root
        self.deletable = False
        # Runs once, between the sweep selecting a row and its bytes going.
        self.on_delete: Callable[[], Awaitable[None]] | None = None

    async def delete(self, storage_ref: str) -> None:
        if not self.deletable:
            msg = f"refusing to delete {storage_ref}"
            raise PermissionError(msg)
        if self.on_delete is not None:
            hook, self.on_delete = self.on_delete, None
            await hook()
        await super().delete(storage_ref)

    async def put_stream(self, storage_ref: str, chunks: AsyncIterator[bytes]) -> int:
        path = self.root / storage_ref
        path.parent.mkdir(parents=True, exist_ok=True)
        total = 0
        with path.open("wb") as handle:
            async for chunk in chunks:
                total += len(chunk)
                handle.write(chunk)
        return total


async def _chunks(*payloads: bytes) -> AsyncIterator[bytes]:
    for payload in payloads:
        yield payload


async def _tenant(db: AsyncSession) -> uuid.UUID:
    """A workspace and a spend identity an upload can be owned by."""
    organization = await create_organization(db, slug="files")
    owner = await create_member(db, organization, role="owner", full_name="Files Owner")
    workspace = await create_workspace(db, organization, name="Files", owner=owner)
    db.add(SpendUser(user_id=_USER_ID))
    await db.flush()
    return workspace.id


def _service(db: AsyncSession, store: _LeakyStore, config: GatewayConfig) -> FileService:
    uow = UnitOfWork(db)

    async def no_default_workspace() -> uuid.UUID:
        raise AssertionError("This upload names its own workspace")

    return FileService(uow, FileRepositories.on(uow), store, config, no_default_workspace)


def _oversized_upload(workspace_id: uuid.UUID) -> NewFile:
    return NewFile(
        user_id=_USER_ID,
        workspace_id=workspace_id,
        filename="big.txt",
        content_type="text/plain",
        purpose="user_data",
        # Two chunks, so the first lands in the store before the second trips the ceiling.
        chunks=_chunks(b"x" * _CEILING, b"x" * _CEILING),
    )


async def _refuse_an_upload(db: AsyncSession, store: _LeakyStore) -> FileService:
    """Run one upload past the ceiling, whose cleanup the store then refuses."""
    workspace_id = await _tenant(db)
    service = _service(db, store, GatewayConfig(files_max_bytes=_CEILING))
    with pytest.raises(UploadTooLargeError):
        await service.store(_oversized_upload(workspace_id))
    return service


async def _store_an_upload(db: AsyncSession, store: _LeakyStore) -> tuple[FileService, FileObject]:
    """Run one upload that completes, returning the service and the row it handed back."""
    workspace_id = await _tenant(db)
    service = _service(db, store, GatewayConfig())
    record = await service.store(
        NewFile(
            user_id=_USER_ID,
            workspace_id=workspace_id,
            filename="a.txt",
            content_type="text/plain",
            purpose="user_data",
            chunks=_chunks(b"a,b\n"),
        )
    )
    return service, record


async def _rows(db: AsyncSession) -> list[FileObject]:
    return list((await db.execute(select(FileObject))).scalars().all())


async def _age(db: AsyncSession, by: timedelta) -> None:
    """Backdate every pending row, which is how a test reaches past the sweep's grace."""
    stale = datetime.now(UTC) - by
    await db.execute(update(FileObject).where(FileObject.pending_since.is_not(None)).values(pending_since=stale))
    await db.commit()


async def test_a_refused_upload_whose_cleanup_fails_keeps_a_row(async_db: AsyncSession, tmp_path: Path) -> None:
    """The store will not drop the partial bytes, so the row is what keeps them reachable."""
    store = _LeakyStore(tmp_path)
    service = await _refuse_an_upload(async_db, store)

    rows = await _rows(async_db)
    assert len(rows) == 1
    assert rows[0].storage_ref is not None
    assert (tmp_path / rows[0].storage_ref).exists()
    with pytest.raises(FileNotServedError):
        await service.stored_file(rows[0].id, FileScope(user_id=_USER_ID, workspace_id=rows[0].workspace_id))


async def test_the_sweep_reclaims_an_upload_that_never_completed(async_db: AsyncSession, tmp_path: Path) -> None:
    """Once the row is past its grace, the sweep gives its bytes and its row back."""
    store = _LeakyStore(tmp_path)
    service = await _refuse_an_upload(async_db, store)

    store.deletable = True
    await _age(async_db, timedelta(days=1))

    batch = await service.sweep(batch_size=10)

    assert batch.reclaimed == 1
    assert await _rows(async_db) == []
    assert not list(tmp_path.rglob("file-*"))


async def test_the_sweep_leaves_an_upload_still_within_its_grace(async_db: AsyncSession, tmp_path: Path) -> None:
    """A row young enough to still be receiving its bytes is not reclaimed under it."""
    store = _LeakyStore(tmp_path)
    service = await _refuse_an_upload(async_db, store)

    store.deletable = True
    batch = await service.sweep(batch_size=10)

    assert batch == SweepBatch(reclaimed=0, seen=0, cursor=None)
    assert len(await _rows(async_db)) == 1



async def test_a_pending_row_past_its_expiry_waits_for_its_grace(async_db: AsyncSession, tmp_path: Path) -> None:
    """Expiry does not reclaim a row still within its grace, because it may still be receiving its bytes."""
    store = _LeakyStore(tmp_path)
    service = await _refuse_an_upload(async_db, store)
    store.deletable = True
    await async_db.execute(update(FileObject).values(expires_at=datetime.now(UTC) - timedelta(days=1)))
    await async_db.commit()

    batch = await service.sweep(batch_size=10)

    assert batch == SweepBatch(reclaimed=0, seen=0, cursor=None)
    assert len(await _rows(async_db)) == 1

async def test_a_stamp_that_lands_under_the_sweep_is_refused(async_db: AsyncSession, tmp_path: Path) -> None:
    """An upload that completes while the sweep is taking its row is told so, rather than told it succeeded."""
    store = _LeakyStore(tmp_path)
    service = await _refuse_an_upload(async_db, store)
    store.deletable = True
    await _age(async_db, timedelta(days=1))
    (reserved,) = await _rows(async_db)
    stamped: list[bool] = []

    async def _stamp_under_the_sweep() -> None:
        try:
            await service.mark_stored(reserved, _CEILING)
        except FileStorageError:
            stamped.append(False)
        else:
            stamped.append(True)

    store.on_delete = _stamp_under_the_sweep
    await service.sweep(batch_size=10)

    assert stamped == [False]
    assert await _rows(async_db) == []


async def test_a_stored_upload_is_handed_back_as_stored(async_db: AsyncSession, tmp_path: Path) -> None:
    """The row a caller receives matches the row in the database, rather than still reading as pending."""
    _, record = await _store_an_upload(async_db, _LeakyStore(tmp_path))

    (row,) = await _rows(async_db)
    assert (record.bytes, record.pending_since) == (row.bytes, row.pending_since)
    assert record.pending_since is None
