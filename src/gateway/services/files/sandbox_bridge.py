"""Moving files between the ``/v1/files`` store and a code-execution sandbox, Otari's or a provider's."""

from __future__ import annotations

import asyncio
import contextlib
import uuid
from collections.abc import AsyncIterator

from gateway.core.config import GatewayConfig
from gateway.core.database import DATABASE_ERRORS
from gateway.core.unit_of_work import UnitOfWork
from gateway.log_config import logger
from gateway.repositories.files import OutputFileRow, existing_file_ids, record_output_file
from gateway.services.file_service import (
    CODE_EXECUTION_OUTPUT_PURPOSE,
    StagedFile,
    expiry_for,
    guess_mime_type,
)
from gateway.services.file_store import FileStore
from gateway.services.files.provider_files import (
    FileOverBudgetError,
    ProviderFile,
    ProviderFileClient,
    ProviderFileUnavailableError,
    serves_files,
)

# A missing credential or a database failure, which stop a copy before it starts.
_COPY_SETUP_ERRORS: tuple[type[BaseException], ...] = (LookupError, ValueError, *DATABASE_ERRORS)


class SandboxFileBridge:
    """Moves files between the ``/v1/files`` store and a code-execution sandbox.

    Built per request by the route, once the billed user and workspace are
    known, and handed to the sandbox backend. ``inputs`` are the uploads the
    request referenced for the sandbox; :meth:`store_output` persists a file a
    run produced as a new file row owned by the same user and workspace, so the
    caller can download it through ``GET /v1/files/{id}/content``.
    :meth:`copy_provider_files` does the same for what a provider's own sandbox
    produced. ``base_url`` is where those downloads are served from, for a loop
    that announces a produced file to the caller as a URL.

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
        # What the request may still copy from a provider, across every call.
        # An allowance of its own: a request that also runs Otari's sandbox
        # spends that one through ``store_output`` and does not share this.
        self._provider_files_left = self.max_output_files
        self._provider_bytes_left = self.max_output_bytes
        # The files this request will not attempt again, either because they are
        # stored or because trying again cannot change the answer. A provider
        # that merely refused is left out, so a later event naming the same file
        # retries it.
        self._provider_files_settled: set[str] = set()
        self._provider_copy_deadline: float | None = None

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
        await self._record(
            OutputFileRow(
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
        )
        return file_id

    async def copy_provider_files(self, files: list[ProviderFile], *, provider: str, provider_instance: str) -> None:
        """Copy the files a provider's own sandbox produced into the store, each under the provider's ID.

        The provider's ID is kept so a client that sends the turn back still names a file Otari knows.
        A file already recorded, or already settled earlier in the request, is left alone. A file the
        provider merely refused is not settled, so a later event naming it tries again.
        The request copies at most ``max_output_files`` files and ``max_output_bytes`` in total,
        within ``files_provider_copy_max_sec``. Those are the request's own allowance: what Otari's
        sandbox stored through :meth:`store_output` does not come out of it.
        Never raises for a failed copy, because a lost file is a smaller failure than a lost reply.
        """
        new = list({file.file_id: file for file in files if file.file_id not in self._provider_files_settled}.values())
        if not new:
            return
        if not serves_files(provider):
            # Otari cannot read this provider's files back at all, so a later event will fare no better.
            self._provider_files_settled.update(file.file_id for file in new)
            return
        try:
            client = ProviderFileClient.for_run(
                self._config, provider=provider, provider_instance=provider_instance, workspace_id=self._workspace_id
            )
            async with self._uow:
                known = await existing_file_ids(self._uow, [file.file_id for file in new])
        except _COPY_SETUP_ERRORS as exc:
            logger.warning("Not copying %d %s file(s): %s", len(new), provider, exc)
            return
        except Exception:  # noqa: BLE001 - a copy failure must not fail the reply
            logger.exception("Not copying %d %s file(s)", len(new), provider)
            return
        # An id that already has a row is somebody's stored file, and a second copy
        # under it would either duplicate the row or overwrite another owner's.
        self._provider_files_settled.update(known)
        pending = [file for file in new if file.file_id not in known]
        if len(pending) > self._provider_files_left:
            logger.warning(
                "%s named %d files to copy; %d may still be stored for this request",
                provider,
                len(pending),
                self._provider_files_left,
            )
        loop = asyncio.get_running_loop()
        if self._provider_copy_deadline is None:
            self._provider_copy_deadline = loop.time() + self._config.files_provider_copy_max_sec
        for file in pending:
            # Each allowance is reported on its own: "no bytes left" and "no time
            # left" are different operator problems with different knobs.
            if self._provider_files_left <= 0:
                logger.warning("%s file %s skipped: the request's file count is spent", provider, file.file_id)
                self._provider_files_settled.add(file.file_id)
                continue
            if self._provider_bytes_left <= 0:
                logger.warning("%s file %s skipped: the request's byte budget is spent", provider, file.file_id)
                self._provider_files_settled.add(file.file_id)
                continue
            if loop.time() >= self._provider_copy_deadline:
                logger.warning("%s file %s skipped: the request's copy time is spent", provider, file.file_id)
                self._provider_files_settled.add(file.file_id)
                continue
            window = asyncio.timeout_at(self._provider_copy_deadline)
            try:
                async with window:
                    size = await self._copy_provider_file(client, file, self._provider_bytes_left)
            except FileOverBudgetError:
                logger.warning(
                    "%s file %s skipped: over the %d bytes left for this request",
                    provider,
                    file.file_id,
                    self._provider_bytes_left,
                )
                self._provider_files_settled.add(file.file_id)
            except ProviderFileUnavailableError as exc:
                # Left unsettled on purpose: a Responses stream names the same
                # file in up to four events, so the next one retries it free.
                logger.warning("Could not copy %s file %s: %s", provider, file.file_id, exc)
            except Exception:  # noqa: BLE001 - one file that cannot be copied must not stop the rest
                if window.expired():
                    logger.warning("%s file %s skipped: the copy ran past its time limit", provider, file.file_id)
                else:
                    logger.exception("Could not copy %s file %s", provider, file.file_id)
            else:
                # Only a stored file spends a slot: a provider having a bad minute
                # must not cost the files named after it their allowance.
                self._provider_files_left -= 1
                self._provider_bytes_left -= size
                self._provider_files_settled.add(file.file_id)

    async def _copy_provider_file(self, client: ProviderFileClient, file: ProviderFile, budget: int) -> int:
        """Copy one file into the store and record its row, returning its size."""
        # A blob key of Otari's own, so two copies of one provider ID never share a blob.
        blob_key = f"file-{uuid.uuid4().hex}"
        async with contextlib.aclosing(client.read(file, budget_bytes=budget)) as chunks:
            storage_ref, size = await self._file_store.put_stream(blob_key, chunks)
        try:
            filename = file.filename or await client.get_filename(file.file_id) or file.file_id
            row = OutputFileRow(
                file_id=file.file_id,
                user_id=self._user_id,
                workspace_id=self._workspace_id,
                filename=filename,
                mime_type=guess_mime_type(filename),
                bytes=size,
                purpose=CODE_EXECUTION_OUTPUT_PURPOSE,
                storage_ref=storage_ref,
                expires_at=expiry_for(self._config),
                provider=client.provider,
                provider_instance=client.provider_instance,
                provider_container_id=file.container_id,
            )
        except BaseException:
            await self._discard(storage_ref)
            raise
        await self._record(row)
        return size

    async def _record(self, row: OutputFileRow) -> None:
        """Record ``row`` in a block of its own, and remove its blob when the row does not land."""
        try:
            async with self._uow:
                await record_output_file(self._uow, row)
        except BaseException:
            await self._discard(row.storage_ref)
            raise

    async def _discard(self, storage_ref: str) -> None:
        """Remove a blob that no row points at."""
        # Shielded so a cancellation already in flight cannot cut the
        # cleanup short and leave the orphan it exists to prevent.
        with contextlib.suppress(Exception):
            await asyncio.shield(self._file_store.delete(storage_ref))
