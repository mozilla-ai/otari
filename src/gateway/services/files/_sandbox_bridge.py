"""Moving files between the ``/v1/files`` store and a code-execution sandbox, Otari's or a provider's."""

from __future__ import annotations

import asyncio
import contextlib
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass

from any_llm import LLMProvider

from gateway.core.config import GatewayConfig
from gateway.core.database import DATABASE_ERRORS
from gateway.exceptions.files_exceptions import FileOverBudgetError, ProviderFileUnavailableError
from gateway.log_config import logger
from gateway.models.files import new_file_id
from gateway.ports.provider_file_port import ProviderFileSession
from gateway.services.files._metadata import expiry_for, guess_mime_type
from gateway.services.files._service import FileBackends, FileService, NewOutput
from gateway.services.files._staging import CODE_EXECUTION_OUTPUT_PURPOSE, StagedFile
from gateway.services.provider_kwargs import effective_credential, get_provider_kwargs
from gateway.types.provider_file import ProviderFile

# A missing credential or a database failure, which stop a copy before it starts.
_COPY_SETUP_ERRORS: tuple[type[BaseException], ...] = (LookupError, ValueError, *DATABASE_ERRORS)


@dataclass(frozen=True)
class _ProducedBy:
    """The configured instance whose provider produced a file."""

    provider: str
    instance: str


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
        backends: FileBackends,
        config: GatewayConfig,
        files: FileService,
        user_id: str,
        workspace_id: uuid.UUID,
        inputs: list[StagedFile],
        base_url: str | None = None,
    ) -> None:
        self._file_store = backends.storage
        self._provider_files = backends.provider_files
        self._config = config
        self._files = files
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

        Streams into the store, so a produced file is never held whole.
        """
        file_id = new_file_id()
        size = await self._files.store_produced(
            NewOutput(
                file_id=file_id,
                user_id=self._user_id,
                workspace_id=self._workspace_id,
                filename=filename,
                mime_type=guess_mime_type(filename),
                purpose=CODE_EXECUTION_OUTPUT_PURPOSE,
                expires_at=expiry_for(self._config),
            ),
            chunks,
        )
        return file_id if size else None

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
        # The session holds a provider connection from the moment it is opened,
        # so it is registered for closing before anything that can fail after it.
        async with contextlib.AsyncExitStack() as stack:
            try:
                member = LLMProvider(provider)
                if not self._provider_files.serves(member):
                    # Otari cannot read this provider's files back at all, so a later event will fare no better.
                    self._provider_files_settled.update(file.file_id for file in new)
                    return
                kwargs = get_provider_kwargs(self._config, member, provider_instance, workspace_id=self._workspace_id)
                client = self._provider_files.open_session(
                    provider=member, instance=provider_instance, credential=effective_credential(member, kwargs)
                )
                await stack.enter_async_context(contextlib.aclosing(client))
                known = await self._files.existing_output_ids([file.file_id for file in new])
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
            await self._copy_batch(client, pending, _ProducedBy(provider, provider_instance))

    async def _copy_batch(self, client: ProviderFileSession, pending: list[ProviderFile], source: _ProducedBy) -> None:
        """Copy what the caps and the request's remaining time allow, logging whatever cannot be copied."""
        if len(pending) > self._provider_files_left:
            logger.warning(
                "%s named %d files to copy; %d may still be stored for this request",
                source.provider,
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
                logger.warning("%s file %s skipped: the request's file count is spent", source.provider, file.file_id)
                self._provider_files_settled.add(file.file_id)
                continue
            if self._provider_bytes_left <= 0:
                logger.warning("%s file %s skipped: the request's byte budget is spent", source.provider, file.file_id)
                self._provider_files_settled.add(file.file_id)
                continue
            if loop.time() >= self._provider_copy_deadline:
                logger.warning("%s file %s skipped: the request's copy time is spent", source.provider, file.file_id)
                self._provider_files_settled.add(file.file_id)
                continue
            window = asyncio.timeout_at(self._provider_copy_deadline)
            try:
                async with window:
                    size = await self._copy_provider_file(client, file, self._provider_bytes_left, source)
            except FileOverBudgetError:
                logger.warning(
                    "%s file %s skipped: over the %d bytes left for this request",
                    source.provider,
                    file.file_id,
                    self._provider_bytes_left,
                )
                self._provider_files_settled.add(file.file_id)
            except ProviderFileUnavailableError as exc:
                # Left unsettled on purpose: a Responses stream names the same
                # file in up to four events, so the next one retries it free.
                logger.warning("Could not copy %s file %s: %s", source.provider, file.file_id, exc)
            except Exception:  # noqa: BLE001 - one file that cannot be copied must not stop the rest
                if window.expired():
                    logger.warning(
                        "%s file %s skipped: the copy ran past its time limit", source.provider, file.file_id
                    )
                else:
                    logger.exception("Could not copy %s file %s", source.provider, file.file_id)
            else:
                # Only a stored file spends a slot: a provider having a bad minute
                # must not cost the files named after it their allowance.
                self._provider_files_left -= 1
                self._provider_bytes_left -= size
                self._provider_files_settled.add(file.file_id)

    async def _copy_provider_file(
        self, client: ProviderFileSession, file: ProviderFile, budget: int, source: _ProducedBy
    ) -> int:
        """Copy one file into the store and record its row, returning its size.

        The name is resolved before the read, so the row is complete before a
        byte arrives. A copy that fails or arrives empty is not recorded, so a
        later request can try the same file again.
        """
        filename = file.filename or await client.filename_of(file.file_id) or file.file_id
        output = NewOutput(
            file_id=file.file_id,
            user_id=self._user_id,
            workspace_id=self._workspace_id,
            filename=filename,
            mime_type=guess_mime_type(filename),
            purpose=CODE_EXECUTION_OUTPUT_PURPOSE,
            expires_at=expiry_for(self._config),
            provider=source.provider,
            provider_instance=source.instance,
            provider_container_id=file.container_id,
        )
        async with contextlib.aclosing(client.read(file, budget_bytes=budget)) as chunks:
            return await self._files.store_produced(output, chunks)
