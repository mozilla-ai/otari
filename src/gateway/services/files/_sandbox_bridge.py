"""Moving files between the ``/v1/files`` store and a code-execution sandbox, Otari's or a provider's."""

from __future__ import annotations

import asyncio
import contextlib
import uuid
from collections.abc import AsyncIterator

from gateway.core.config import GatewayConfig
from gateway.core.database import DATABASE_ERRORS
from gateway.log_config import logger
from gateway.ports.file_storage_port import FileStoragePort
from gateway.services.files._cleanup import discard_output_bytes
from gateway.services.files._metadata import expiry_for, guess_mime_type
from gateway.services.files._provider_files import (
    FileOverBudgetError,
    ProviderFile,
    ProviderFileClient,
    ProviderFileUnavailableError,
    serves_files,
)
from gateway.services.files._service import FileService, NewOutput
from gateway.services.files._staging import CODE_EXECUTION_OUTPUT_PURPOSE, StagedFile

# A missing credential or a database failure, which stop a copy before it starts.
_COPY_SETUP_ERRORS: tuple[type[BaseException], ...] = (LookupError, ValueError, *DATABASE_ERRORS)
# The time one request may spend copying a provider's files, across every call.
_PROVIDER_COPY_SECONDS = 60.0


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
    Persistence goes through the Files service, which owns the request's
    database transactions. Transfers run outside those transactions.
    """

    def __init__(
        self,
        *,
        file_store: FileStoragePort,
        config: GatewayConfig,
        files: FileService,
        user_id: str,
        workspace_id: uuid.UUID,
        inputs: list[StagedFile],
        base_url: str | None = None,
    ) -> None:
        self._file_store = file_store
        self._config = config
        self._files = files
        self._user_id = user_id
        self._workspace_id = workspace_id
        self.inputs = inputs
        self.base_url = base_url
        # What the request may still copy from a provider, across every call.
        self._provider_files_left = self.max_output_files
        self._provider_bytes_left = self.max_output_bytes
        self._provider_files_handled: set[str] = set()
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
        await self._files.record_output(
            NewOutput(
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
        A file already recorded, or already handled earlier in the request, is left alone.
        The request copies at most ``max_output_files`` files and ``max_output_bytes`` in total, in one time limit.
        Never raises for a failed copy, because a lost file is a smaller failure than a lost reply.
        """
        new = list({file.file_id: file for file in files if file.file_id not in self._provider_files_handled}.values())
        self._provider_files_handled.update(file.file_id for file in new)
        if not new or not serves_files(provider):
            return
        # The client holds a provider connection from the moment it is built, so
        # it is registered for closing before anything that can fail after it.
        async with contextlib.AsyncExitStack() as stack:
            try:
                client = ProviderFileClient.for_run(
                    self._config,
                    provider=provider,
                    provider_instance=provider_instance,
                    workspace_id=self._workspace_id,
                )
                await stack.enter_async_context(contextlib.aclosing(client))
                known = await self._files.existing_output_ids([file.file_id for file in new])
            except _COPY_SETUP_ERRORS as exc:
                logger.warning("Not copying %d %s file(s): %s", len(new), provider, exc)
                return
            except Exception:  # noqa: BLE001 - a copy failure must not fail the reply
                logger.exception("Not copying %d %s file(s)", len(new), provider)
                return
            await self._copy_batch(client, [file for file in new if file.file_id not in known], provider)

    async def _copy_batch(self, client: ProviderFileClient, pending: list[ProviderFile], provider: str) -> None:
        """Copy what the caps and the request's remaining time allow, logging whatever cannot be copied."""
        if len(pending) > self._provider_files_left:
            logger.warning(
                "%s produced %d files; copying the first %d", provider, len(pending), self._provider_files_left
            )
        batch = pending[: self._provider_files_left]
        self._provider_files_left -= len(batch)
        loop = asyncio.get_running_loop()
        if self._provider_copy_deadline is None:
            self._provider_copy_deadline = loop.time() + _PROVIDER_COPY_SECONDS
        for file in batch:
            if self._provider_bytes_left <= 0 or loop.time() >= self._provider_copy_deadline:
                logger.warning("%s file %s skipped: no bytes or time left for this request", provider, file.file_id)
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
            except ProviderFileUnavailableError as exc:
                logger.warning("Could not copy %s file %s: %s", provider, file.file_id, exc)
            except Exception:  # noqa: BLE001 - one file that cannot be copied must not stop the rest
                if window.expired():
                    logger.warning("%s file %s skipped: the copy ran past its time limit", provider, file.file_id)
                else:
                    logger.exception("Could not copy %s file %s", provider, file.file_id)
            else:
                self._provider_bytes_left -= size

    async def _copy_provider_file(self, client: ProviderFileClient, file: ProviderFile, budget: int) -> int:
        """Copy one file into the store and record its row, returning its size."""
        # A blob key of Otari's own, so two copies of one provider ID never share a blob.
        blob_key = f"file-{uuid.uuid4().hex}"
        async with contextlib.aclosing(client.read(file, budget_bytes=budget)) as chunks:
            storage_ref, size = await self._file_store.put_stream(blob_key, chunks)
        try:
            filename = file.filename or await client.get_filename(file.file_id) or file.file_id
            output = NewOutput(
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
            await discard_output_bytes(self._file_store, storage_ref)
            raise
        await self._files.record_output(output)
        return size
