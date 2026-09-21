"""Moving files between the ``/v1/files`` store and one code-execution session."""

from __future__ import annotations

import asyncio
import contextlib
import uuid
from collections.abc import AsyncIterator

from gateway.core.config import GatewayConfig
from gateway.core.unit_of_work import UnitOfWork
from gateway.repositories.files import OutputFileRow, record_output_file
from gateway.services.file_service import (
    CODE_EXECUTION_OUTPUT_PURPOSE,
    StagedFile,
    expiry_for,
    guess_mime_type,
)
from gateway.services.file_store import FileStore


class SandboxFileBridge:
    """Moves files between the ``/v1/files`` store and one sandbox session.

    Built per request by the route, once the billed user and workspace are
    known, and handed to the sandbox backend. ``inputs`` are the uploads the
    request referenced for the sandbox; :meth:`store_output` persists a file a
    run produced as a new file row owned by the same user and workspace, so the
    caller can download it through ``GET /v1/files/{id}/content``. ``base_url``
    is where those downloads are served from, for a loop that announces a
    produced file to the caller as a URL.

    Standalone only: it needs the local database that hybrid mode does not have.
    Writes go through the request's Unit of Work, ``uow``: the request session
    is released before the provider is dispatched, so it holds no transaction
    while the tool loop runs, and each stored file is one block of its own.
    """

    def __init__(
        self,
        *,
        file_store: FileStore,
        config: GatewayConfig,
        uow: UnitOfWork,
        user_id: str,
        workspace_id: uuid.UUID,
        inputs: list[StagedFile],
        base_url: str | None = None,
    ) -> None:
        self._file_store = file_store
        self._config = config
        self._uow = uow
        self._user_id = user_id
        self._workspace_id = workspace_id
        self.inputs = inputs
        self.base_url = base_url

    @property
    def max_output_files(self) -> int:
        return self._config.files_output_max_files

    @property
    def max_output_bytes(self) -> int:
        return min(self._config.files_output_max_bytes, self._config.files_max_bytes)

    async def read_input(self, staged: StagedFile) -> bytes:
        return await self._file_store.get(staged.storage_ref)

    async def store_output(self, filename: str, chunks: AsyncIterator[bytes]) -> str | None:
        """Persist ``chunks`` as a new file and return its ``file_id``, or ``None`` when empty.

        Streams into the store, so a produced file is never held whole. Whatever
        stops the row from landing, the blob goes with it, so nothing sits in the
        store that no row and no sweep can reach.
        """
        file_id = f"file-{uuid.uuid4().hex}"
        storage_ref, size = await self._file_store.put_stream(file_id, chunks)
        if size == 0:
            await self._file_store.delete(storage_ref)
            return None
        row = OutputFileRow(
            file_id=file_id,
            user_id=self._user_id,
            workspace_id=self._workspace_id,
            filename=filename,
            mime_type=guess_mime_type(filename),
            bytes=size,
            purpose=CODE_EXECUTION_OUTPUT_PURPOSE,
            storage_ref=storage_ref,
            expires_at=expiry_for(self._config),
        )
        try:
            async with self._uow:
                await record_output_file(self._uow, row)
        except BaseException:
            # Shielded so a cancellation already in flight cannot cut the
            # cleanup short and leave the orphan it exists to prevent.
            with contextlib.suppress(Exception):
                await asyncio.shield(self._file_store.delete(storage_ref))
            raise
        return file_id
