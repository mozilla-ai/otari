"""Short-lived copies of a stored upload at the provider that runs a request's code.

Otari's store stays the source of truth.
A copy carries an expiry the provider enforces, never outlives the file's own,
and is reused while it has enough life left for the request that finds it.
A provider that answers with a longer expiry than it was asked for has the copy
taken back, because the rule is about what exists rather than what was asked.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

from gateway.core.config import GatewayConfig
from gateway.core.database import DATABASE_ERRORS
from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions.files_exceptions import (
    AttachedFileExpiresTooSoonError,
    ProviderUploadDisabledError,
    ProviderUploadFailedError,
)
from gateway.log_config import logger
from gateway.models.files import FileProviderCopy
from gateway.ports.file_storage_port import FileStoragePort
from gateway.repositories.files import FileRepositories
from gateway.services.files._provider_files import ProviderFileClient, minimum_copy_lifetime
from gateway.services.files._staging import StagedFile

# How much of a copy's life must be left for a request to use it. A copy that
# expires while the model's code is still running leaves that code without its
# input, which costs more than the upload it saved.
_REUSE_MARGIN = timedelta(minutes=5)


def _as_utc(value: datetime) -> datetime:
    """``value`` in UTC, reading an offset-less timestamp as UTC, which is what providers report."""
    return value.astimezone(UTC) if value.tzinfo is not None else value.replace(tzinfo=UTC)


class ProviderFileUploader:
    """Gives the provider running a request's code an ID for an upload the request attached.

    Built for one request, so the provider instance and the workspace whose
    credential the copy is made with are fixed for every file it is asked about.
    """

    def __init__(
        self,
        uow: UnitOfWork,
        repositories: FileRepositories,
        file_store: FileStoragePort,
        config: GatewayConfig,
        *,
        provider: str,
        provider_instance: str,
        workspace_id: uuid.UUID | None,
    ) -> None:
        self._uow = uow
        self._copies = repositories.provider_copies
        self._file_store = file_store
        self._config = config
        self._provider = provider
        self._provider_instance = provider_instance
        self._workspace_id = workspace_id

    async def file_id_for(self, staged: StagedFile) -> str:
        """The provider's ID for a copy of ``staged``, uploading one when no usable copy exists.

        Raises:
            ProviderUploadDisabledError: the deployment makes no provider copies.
            AttachedFileExpiresTooSoonError: the copy would outlive the file.
            ProviderUploadFailedError: the copy could not be made or recorded.
        """
        if not self._config.files_provider_upload_enabled:
            raise ProviderUploadDisabledError
        workspace_id = self._workspace_id
        if workspace_id is None:
            # Unreachable from the request path, which always resolves a workspace:
            # a key carries one and a master key falls back to the default. Kept
            # because a copy cannot be keyed without it, so there is nothing to
            # record or reuse, and answering without the file is not an option.
            logger.warning("No workspace resolved to make a copy of file %s", staged.file_id)
            raise ProviderUploadFailedError
        now = datetime.now(UTC)
        try:
            async with self._uow:
                existing = await self._copies.in_account(
                    staged.file_id,
                    provider=self._provider,
                    provider_instance=self._provider_instance,
                    credential_workspace_id=workspace_id,
                )
        except DATABASE_ERRORS as exc:
            logger.warning("Could not read the copies of file %s: %s", staged.file_id, exc)
            raise ProviderUploadFailedError from exc
        if existing is not None and existing.expires_at > now + _REUSE_MARGIN:
            return existing.provider_file_id

        copy = await self._upload(staged, workspace_id)
        try:
            async with self._uow:
                await self._copies.record(copy)
        except DATABASE_ERRORS as exc:
            # The copy is at the provider and nothing names it, so it is
            # unreachable until its own expiry. Refusing stops this request
            # adding a second; a retry while the database is down adds another.
            logger.warning("Could not record the copy of file %s: %s", staged.file_id, exc)
            raise ProviderUploadFailedError from exc
        return copy.provider_file_id

    def _lifetime(self, staged: StagedFile, now: datetime) -> timedelta:
        """How long the copy may live: the configured life, never past the file's own.

        A copy that outlived its file would leave the provider holding something
        Otari no longer serves and can no longer reach.

        Raises:
            AttachedFileExpiresTooSoonError: what is left cannot be asked for,
                because the provider will not store a file for that short a time
                or because it rounds away to nothing.
        """
        ttl = timedelta(hours=self._config.files_provider_upload_ttl_hours)
        if staged.expires_at is not None:
            ttl = min(ttl, _as_utc(staged.expires_at) - now)
        if ttl < minimum_copy_lifetime(self._provider) or ttl <= _REUSE_MARGIN:
            logger.warning(
                "File %s has less life left than provider %s will hold a copy for; "
                "files_retention_hours must exceed that floor for a copy to be possible",
                staged.file_id,
                self._provider,
            )
            raise AttachedFileExpiresTooSoonError
        return ttl

    def _refuse_a_file_expiring_too_soon(self, staged: StagedFile) -> None:
        """Raise where no copy of ``staged`` can be made that expires no later than it does."""
        self._lifetime(staged, datetime.now(UTC))

    async def _discard_an_overlong_copy(
        self, client: ProviderFileClient, staged: StagedFile, provider_file_id: str, expires_at: datetime
    ) -> None:
        """Remove a copy the provider will hold past the file's own expiry, and refuse.

        Otari asks for an expiry and cannot make a provider honor it, so a copy
        that would outlive the file is taken back rather than recorded. Refusing
        either way, because a copy left behind is the thing this promises not to
        do. The comparison is against the file rather than against the life
        asked for, which the provider starts counting a round trip later.

        Raises:
            ProviderUploadFailedError: always.
        """
        removed = await client.discard(provider_file_id)
        logger.warning(
            "Provider %s would hold its copy of file %s until %s, past the file's own expiry; %s",
            self._provider,
            staged.file_id,
            expires_at.isoformat(),
            "removed it" if removed else "it could not be removed",
        )
        raise ProviderUploadFailedError

    async def _upload(self, staged: StagedFile, workspace_id: uuid.UUID) -> FileProviderCopy:
        """Put ``staged``'s bytes at the provider and describe the copy that came back."""
        # Refused before a byte is read, so a file too close to its own expiry
        # costs nothing. The life the provider is told is measured again below.
        self._refuse_a_file_expiring_too_soon(staged)
        try:
            data = await self._file_store.get(staged.storage_ref)
            client = ProviderFileClient.for_run(
                self._config,
                provider=self._provider,
                provider_instance=self._provider_instance,
                workspace_id=workspace_id,
            )
            try:
                # Measured here rather than before the blob read, because the
                # provider starts the copy's life when it accepts the upload.
                now = datetime.now(UTC)
                ttl = self._lifetime(staged, now)
                metadata = await client.upload(
                    data,
                    filename=staged.filename,
                    mime_type=staged.mime_type,
                    expires_in=int(ttl.total_seconds()),
                )
                expires_at = _as_utc(metadata.expires_at) if metadata.expires_at else now + ttl
                if staged.expires_at is not None and expires_at > _as_utc(staged.expires_at):
                    await self._discard_an_overlong_copy(client, staged, metadata.id, expires_at)
            finally:
                await client.aclose()
        except (LookupError, OSError, ValueError) as exc:
            # No credential for the instance, or bytes that could not be read.
            logger.warning("Could not give provider %s a copy of file %s: %s", self._provider, staged.file_id, exc)
            raise ProviderUploadFailedError from exc
        return FileProviderCopy(
            file_id=staged.file_id,
            provider=self._provider,
            provider_instance=self._provider_instance,
            credential_workspace_id=workspace_id,
            provider_file_id=metadata.id,
            expires_at=expires_at,
            created_at=now,
        )
