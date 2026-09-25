"""The files use cases: storing an upload, paging a caller's files, serving one back and discarding it."""

from __future__ import annotations

import uuid
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Collection, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum

from gateway.core.config import GatewayConfig
from gateway.core.database import DATABASE_ERRORS
from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions.files_exceptions import (
    EmptyUploadError,
    FileNotServedError,
    FileStorageError,
    UnknownPageCursorError,
    UploadTooLargeError,
)
from gateway.log_config import logger
from gateway.models.files import FileObject
from gateway.ports.file_storage_port import FileStoragePort
from gateway.repositories.files import FilePageQuery, FileRepositories, OutputFileRow
from gateway.services.files._cleanup import discard_output_bytes
from gateway.services.files._file_ids import file_id_in, page_token
from gateway.services.files._metadata import expiry_for, guess_mime_type
from gateway.services.files._staging import StagedFile

# Resolves the workspace a deployment-wide write lands in. It belongs to the
# organizations domain, so files receives it rather than looking it up.
DefaultWorkspace = Callable[[], Awaitable[uuid.UUID]]

# Page bounds. The default is OpenAI's; the ceiling is well under OpenAI's 10000
# because a page is one query and one JSON body.
DEFAULT_LIST_LIMIT = 100
MAX_LIST_LIMIT = 1000


class FileDialect(StrEnum):
    """Which SDK's Files API a request speaks.

    The two share their paths and verbs, and the gateway serves both from one
    set of rows. They differ in how a page is resumed, so a listing says which
    it is and gets its own cursor back.
    """

    OPENAI = "openai"
    ANTHROPIC = "anthropic"


@dataclass(frozen=True)
class FileScope:
    """The rows one files request may reach.

    ``workspace_id`` is the workspace the authenticating key belongs to, and is
    None for the master key, which is the operator acting deployment-wide and
    sees every workspace.
    """

    user_id: str
    workspace_id: uuid.UUID | None = None


@dataclass(frozen=True)
class NewFile:
    """An upload as the request carried it, before the store has seen its bytes."""

    user_id: str
    # None for a master-key upload, which lands in the deployment's default workspace.
    workspace_id: uuid.UUID | None
    filename: str | None
    content_type: str | None
    purpose: str
    chunks: AsyncIterator[bytes]


@dataclass(frozen=True)
class NewOutput:
    """A produced file to register after its bytes have been stored."""

    file_id: str
    user_id: str
    workspace_id: uuid.UUID
    filename: str
    mime_type: str
    bytes: int
    purpose: str
    storage_ref: str
    expires_at: datetime | None
    provider: str | None = None
    provider_instance: str | None = None
    provider_container_id: str | None = None


@dataclass(frozen=True)
class FileListing:
    """One page of a caller's files, as the request asked for it."""

    scope: FileScope
    dialect: FileDialect
    # Clamped to ``MAX_LIST_LIMIT``, so a caller that does not bound it cannot ask for the whole table.
    limit: int = DEFAULT_LIST_LIMIT
    ascending: bool = False
    purpose: str | None = None
    # The files a caller named outright, which replaces paging through them.
    file_ids: Sequence[str] | None = None
    cursor: str | None = None


@dataclass(frozen=True)
class FilePage:
    """One page of a caller's files, with what resumes the listing after it."""

    files: list[FileObject]
    # The cursor a following request passes back, spelled for the listing's
    # dialect, or None when this page is the last.
    next_cursor: str | None


@dataclass(frozen=True)
class FileContent:
    """A stored file's bytes, ready to be read out, and what describes them."""

    chunks: AsyncGenerator[bytes, None]
    filename: str
    mime_type: str


@dataclass(frozen=True)
class SweepBatch:
    """One cleanup batch and the key that resumes scanning after it."""

    reclaimed: int
    # Rows inspected, including failed deletions. A short batch ends this scan.
    seen: int
    # Last inspected (created_at, id), or None for an empty batch. Advance past
    # failed deletions so they do not block later candidates in the same tick.
    cursor: tuple[datetime, str] | None


class FileService:
    """Everything the Files API does with a caller's uploads.

    The bytes go to a blob store behind :class:`FileStoragePort` and the
    metadata to a row, and the two are kept in step as far as they can be: an
    upload refused after its bytes are written takes them with it, and a
    discarded file loses its bytes after the row says so. Neither is absolute.
    A cancellation between the write and the commit leaves bytes no row points
    at, because the commit's outcome is unknown there and removing them could
    destroy the bytes of a row that did land.
    """

    def __init__(
        self,
        uow: UnitOfWork,
        repositories: FileRepositories,
        file_store: FileStoragePort,
        config: GatewayConfig,
        default_workspace: DefaultWorkspace,
    ) -> None:
        self._uow = uow
        self._files = repositories.files
        self._file_store = file_store
        self._config = config
        self._default_workspace = default_workspace

    async def store(self, upload: NewFile) -> FileObject:
        """Store an upload's bytes and record the file, and return the row.

        Raises:
            UploadTooLargeError: the upload ran past the deployment's ceiling.
            EmptyUploadError: the upload carried no bytes.
            FileStorageError: the bytes were written but the row would not land.
        """
        file_id = f"file-{uuid.uuid4().hex}"
        max_bytes = self._config.files_max_bytes
        storage_ref, size = await self._file_store.put_stream(file_id, _capped(upload.chunks, max_bytes))
        if size == 0:
            # The size is only known once the stream drains, so a zero-byte blob
            # is already in the store by the time the upload is refused.
            await self._drop_orphan(storage_ref, file_id)
            raise EmptyUploadError

        now = datetime.now(UTC)
        try:
            async with self._uow:
                try:
                    # Resolved here rather than before the upload: it reads the
                    # database, and doing that first would hold the session's
                    # transaction open for as long as the bytes take to store.
                    workspace_id = upload.workspace_id
                    if workspace_id is None:
                        workspace_id = await self._default_workspace()
                    record = FileObject(
                        id=file_id,
                        user_id=upload.user_id,
                        workspace_id=workspace_id,
                        filename=upload.filename or file_id,
                        mime_type=guess_mime_type(upload.filename, upload.content_type),
                        bytes=size,
                        purpose=upload.purpose,
                        storage_ref=storage_ref,
                        created_at=now,
                        expires_at=expiry_for(self._config, now),
                    )
                    await self._files.add(record)
                except BaseException:
                    # Raised inside the block, so the Unit of Work has not
                    # reached its commit and the bytes are certainly
                    # unreferenced. A cancellation is caught here for that
                    # reason: it cannot strand a row that landed, because none
                    # can have landed yet.
                    await self._drop_orphan(storage_ref, file_id)
                    raise
        except DATABASE_ERRORS as exc:
            # Logged before the cleanup, so the failure that ended the upload is
            # on the record whatever the cleanup then does.
            logger.error("Failed to persist file metadata for %s: %s", file_id, exc)
            # The bytes were written before the row was staged; drop them so a
            # failed insert does not leak a blob nothing references.
            await self._drop_orphan(storage_ref, file_id)
            raise FileStorageError(f"Could not record the file {file_id}") from exc
        except Exception:
            # The commit itself failed, so the block rolled back and the row did
            # not land. A cancellation is deliberately not caught out here: it
            # can arrive while the commit is in flight, where the outcome is
            # unknown and dropping the bytes of a row that did land is worse.
            # The inner handler has already dropped the blob for anything that
            # failed before the commit, and dropping twice is a no-op.
            await self._drop_orphan(storage_ref, file_id)
            raise

        logger.info(
            "Stored file %s (%d bytes) for user %s in workspace %s", file_id, size, upload.user_id, workspace_id
        )
        return record

    async def page(self, listing: FileListing) -> FilePage:
        """Return one page of the caller's files, and the cursor that resumes after it.

        Raises:
            UnknownPageCursorError: an Anthropic page token names no position this gateway issued.
            FileNotServedError: an OpenAI cursor names no file the caller owns.
        """
        limit = min(listing.limit, MAX_LIST_LIMIT)
        anthropic = listing.dialect is FileDialect.ANTHROPIC
        cursor_id = listing.cursor
        if anthropic and cursor_id is not None:
            cursor_id = file_id_in(cursor_id)
            if cursor_id is None:
                raise UnknownPageCursorError
        query = FilePageQuery(
            user_id=listing.scope.user_id,
            workspace_id=listing.scope.workspace_id,
            purpose=listing.purpose,
            file_ids=listing.file_ids,
            ascending=listing.ascending,
            # One row past the page says whether another follows, without a count.
            limit=limit + 1,
        )
        async with self._uow:
            if cursor_id is not None:
                query = replace(query, after=await self._position(cursor_id, listing))
            records = await self._files.page(query)

        has_more = len(records) > limit
        files = records[:limit]
        if not has_more:
            return FilePage(files=files, next_cursor=None)
        last = files[-1].id
        return FilePage(files=files, next_cursor=page_token(last) if anthropic else last)

    async def stored_file(self, file_id: str, scope: FileScope) -> FileObject:
        """Return the file the caller is served under ``file_id``.

        Raises:
            FileNotServedError: no such file is served to this caller.
        """
        async with self._uow:
            record = await self._files.live(file_id, scope.user_id, workspace_id=scope.workspace_id)
        if record is None:
            raise FileNotServedError
        return record

    async def content(self, file_id: str, scope: FileScope) -> FileContent:
        """Open the file's bytes for reading, having proved the caller is served it.

        The first chunk is read here so that a blob that is gone or unreadable
        fails now, while the failure can still be reported, rather than after
        the caller has been told the read succeeded.

        Raises:
            FileNotServedError: no such file is served to this caller, or its row holds no bytes.
            FileStorageError: the bytes could not be read.
        """
        record = await self.stored_file(file_id, scope)
        if record.storage_ref is None:
            logger.error("File %s has no stored bytes", file_id)
            raise FileNotServedError
        try:
            chunks = await _primed(self._file_store.get_stream(record.storage_ref))
        except OSError as exc:
            logger.error("Failed to read blob for file %s (ref=%s): %s", file_id, record.storage_ref, exc)
            raise FileStorageError(f"Could not read the bytes of {file_id}") from exc
        return FileContent(chunks=chunks, filename=record.filename, mime_type=record.mime_type)

    async def staged_upload(self, file_id: str, scope: FileScope) -> StagedFile | None:
        """The stored upload under ``file_id``, ready to be read or handed to a sandbox, or None.

        None whenever the file cannot be served to this caller, and for a row
        whose bytes are not in the store, so that a reference to one is dropped
        rather than failing the request that carried it.

        It answers whether or not the deployment serves the Files API, because
        it resolves a reference a request already holds rather than serving that
        API, and its caller has a switch of its own.
        """
        async with self._uow:
            record = await self._files.live(file_id, scope.user_id, workspace_id=scope.workspace_id)
        if record is None or record.storage_ref is None:
            return None
        return StagedFile(record.id, record.filename, record.mime_type, record.storage_ref)

    async def read_bytes(self, staged: StagedFile) -> bytes:
        """The whole of a staged upload's bytes."""
        return await self._file_store.get(staged.storage_ref)

    async def discard(self, file_id: str, scope: FileScope) -> None:
        """Stop serving the file and give its bytes back.

        Raises:
            FileNotServedError: no such file is served to this caller.
            FileStorageError: the file is still served because the row would not change.
        """
        try:
            # One block for the read and the write, so nothing can discard the
            # file between proving the caller is served it and stamping the row.
            async with self._uow:
                record = await self._files.live(file_id, scope.user_id, workspace_id=scope.workspace_id)
                if record is None:
                    raise FileNotServedError
                storage_ref = record.storage_ref
                await self._files.soft_delete(record, datetime.now(UTC))
        except DATABASE_ERRORS as exc:
            logger.error("Failed to delete file %s: %s", file_id, exc)
            raise FileStorageError(f"Could not discard the file {file_id}") from exc

        # The row is already stored, so the file is gone from the caller's view.
        # Removing the blob is best effort: a storage failure must not turn a
        # completed delete into a failure, and only leaves an unreferenced blob
        # for the sweep.
        if storage_ref is not None:
            try:
                await self._file_store.delete(storage_ref)
            except OSError as exc:
                logger.warning("Discarded file %s but failed to remove its blob %s: %s", file_id, storage_ref, exc)

    async def existing_output_ids(self, file_ids: Collection[str]) -> set[str]:
        """Identify already-recorded output IDs without returning their contents or owners."""
        async with self._uow:
            return await self._files.existing_ids(file_ids)

    async def record_output(self, output: NewOutput) -> None:
        """Record stored output, cleaning up on failure but not on an uncertain commit."""
        staged = False
        try:
            row = OutputFileRow(
                file_id=output.file_id,
                user_id=output.user_id,
                workspace_id=output.workspace_id,
                filename=output.filename,
                mime_type=output.mime_type,
                bytes=output.bytes,
                purpose=output.purpose,
                storage_ref=output.storage_ref,
                expires_at=output.expires_at,
                provider=output.provider,
                provider_instance=output.provider_instance,
                provider_container_id=output.provider_container_id,
            )
            async with self._uow:
                await self._files.record_output(row)
                staged = True
        except BaseException:
            # A commit error can follow a successful database commit.
            # Before staging completes, no output can have committed.
            if not staged:
                await discard_output_bytes(self._file_store, output.storage_ref)
            raise

    async def sweep(self, *, batch_size: int, after: tuple[datetime, str] | None = None) -> SweepBatch:
        """Delete expired or revoked bytes between short database transactions."""
        async with self._uow:
            records = await self._files.reclaimable(batch_size=batch_size, after=after)
            candidates = [(record.id, record.storage_ref, record.created_at) for record in records]
        reclaimed: list[str] = []
        for file_id, storage_ref, _ in candidates:
            try:
                if storage_ref is not None:
                    await self._file_store.delete(storage_ref)
            except FileNotFoundError:
                pass
            except OSError as exc:
                logger.warning("File sweep could not remove bytes for %s: %s", file_id, exc)
                continue
            reclaimed.append(file_id)
        if reclaimed:
            async with self._uow:
                await self._files.remove_all(reclaimed)
            logger.info("File sweep reclaimed %d file(s)", len(reclaimed))
        cursor = (candidates[-1][2], candidates[-1][0]) if candidates else None
        return SweepBatch(reclaimed=len(reclaimed), seen=len(candidates), cursor=cursor)

    async def _drop_orphan(self, storage_ref: str, file_id: str) -> None:
        """Remove bytes that no row points at, best effort.

        A store that will not drop them leaves an orphan for an operator to
        reclaim, which is a smaller failure than replacing the refusal that
        caused the cleanup with a storage error.
        """
        try:
            await self._file_store.delete(storage_ref)
        except OSError as exc:
            logger.warning("Could not remove the unreferenced blob %s for %s: %s", storage_ref, file_id, exc)

    async def _position(self, cursor_id: str, listing: FileListing) -> tuple[datetime, str]:
        """The ``(created_at, id)`` key a cursor resumes after.

        Read with the tenant predicates only, so a cursor deleted or expired
        between two pages still says where the next page starts. Another user's
        ID names no position.
        """
        cursor = await self._files.any_owned(cursor_id, listing.scope.user_id, workspace_id=listing.scope.workspace_id)
        if cursor is None:
            if listing.dialect is FileDialect.ANTHROPIC:
                raise UnknownPageCursorError
            raise FileNotServedError
        return cursor.created_at, cursor.id


async def _capped(chunks: AsyncIterator[bytes], max_bytes: int) -> AsyncIterator[bytes]:
    """Yield the upload's bytes, refusing it past ``max_bytes``.

    The cap is applied as the bytes flow into the store rather than to a buffer
    built first, so an oversized upload is never held whole.
    """
    total = 0
    async for chunk in chunks:
        total += len(chunk)
        if total > max_bytes:
            raise UploadTooLargeError(max_bytes)
        yield chunk


async def _primed(chunks: AsyncGenerator[bytes, None]) -> AsyncGenerator[bytes, None]:
    """Read ``chunks``' first item eagerly so a read failure raises here, not later."""
    try:
        first: bytes | None = await chunks.__anext__()
    except StopAsyncIteration:
        first = None

    async def _rest() -> AsyncGenerator[bytes, None]:
        # A reader that stops early closes this outer generator, and that does
        # not reach `chunks` on its own (there is no such thing for a bare
        # `async for`). Without this try/finally the store's file handle stays
        # open until the abandoned generator is collected, which under real
        # traffic (cancelled downloads, closed tabs) means fds pile up.
        try:
            if first is not None:
                yield first
                async for chunk in chunks:
                    yield chunk
        finally:
            await chunks.aclose()

    return _rest()
