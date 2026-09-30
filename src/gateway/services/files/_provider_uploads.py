"""Short-lived copies of a stored upload at the provider that runs a request's code.

Otari's store stays the source of truth.
A copy carries an expiry the provider enforces, never outlives the file's own,
and is reused while it has enough life left for the request that finds it.
A provider that answers with a longer expiry than it was asked for has the copy
taken back, because the rule is about what exists rather than what was asked.

Each copy is reserved as a pending row before it is uploaded, confirmed once
the provider holds it, and canceled when it does not.
A copy whose upload or confirmation is cut off can be left unnamed, and the
provider expires it on its own.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

from any_llm import LLMProvider

from gateway.core.config import GatewayConfig
from gateway.core.database import DATABASE_ERRORS
from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions.files_exceptions import (
    AttachedFileExpiresTooSoonError,
    ProviderCopyNotRecordedError,
    ProviderUploadDisabledError,
    ProviderUploadFailedError,
)
from gateway.log_config import logger
from gateway.models.files import FileProviderCopy
from gateway.ports.file_storage_port import FileStoragePort
from gateway.repositories.files import FileProviderCopyRepository
from gateway.services.files._provider_files import ProviderFileClient, minimum_copy_lifetime
from gateway.services.files._staging import StagedFile
from gateway.types.provider_account import ProviderAccount, ResolvedCredential

# How much of a copy's life must be left for a request to use it. A copy that
# expires while the model's code is still running leaves that code without its
# input, which costs more than the upload it saved.
_REUSE_MARGIN = timedelta(minutes=5)

# How long an upload may take to be accepted. The provider counts a copy's life
# from acceptance, so a copy capped by its file's expiry is asked for this much less.
_ACCEPT_SLACK = timedelta(minutes=1)

# The providers whose own code execution reads a file only under an ID from
# their own files API, and which accept an expiry on that file.
_COPY_PROVIDERS = frozenset({LLMProvider.ANTHROPIC})

_RESERVE_ERRORS: tuple[type[BaseException], ...] = (ProviderCopyNotRecordedError, *DATABASE_ERRORS)


def provider_holds_copies(provider: LLMProvider) -> bool:
    """Whether ``provider``'s own code execution can be given an attached file as a copy."""
    return provider in _COPY_PROVIDERS


def _as_utc(value: datetime) -> datetime:
    """``value`` in UTC, reading an offset-less timestamp as UTC, which is what providers report."""
    return value.astimezone(UTC) if value.tzinfo is not None else value.replace(tzinfo=UTC)


class ProviderCopies:
    """Gives a provider account an ID for each upload a request attached."""

    def __init__(
        self, uow: UnitOfWork, copies: FileProviderCopyRepository, file_store: FileStoragePort, config: GatewayConfig
    ) -> None:
        self._uow = uow
        self._copies = copies
        self._file_store = file_store
        self._config = config

    async def file_ids_for(
        self, files: Sequence[StagedFile], account: ProviderAccount, credential: ResolvedCredential
    ) -> dict[str, str]:
        """The provider's ID for a copy of each file in ``account``, uploading one where none is usable.

        Raises:
            ProviderUploadDisabledError: the deployment makes no provider copies.
            AttachedFileExpiresTooSoonError: a copy would outlive its file.
            ProviderUploadFailedError: a copy could not be made or recorded.
        """
        if not self._config.files_provider_upload_enabled:
            raise ProviderUploadDisabledError
        if not files:
            return {}
        # Every file is checked before any is uploaded, so a refusal leaves no copy behind.
        for staged in files:
            self._lifetime(staged, account, datetime.now(UTC))
        ids = {
            file_id: copy.provider_file_id
            for file_id, copy in (await self._usable(files, account)).items()
            if copy.provider_file_id is not None
        }
        missing = [staged for staged in files if staged.file_id not in ids]
        if not missing:
            return ids
        try:
            client = ProviderFileClient(
                provider=account.provider, provider_instance=account.instance, credential=credential
            )
        except LookupError as exc:
            logger.warning("Provider %s cannot hold copies of attached files: %s", account.provider.value, exc)
            raise ProviderUploadFailedError from exc
        try:
            for staged in missing:
                ids[staged.file_id] = await self._copy(client, staged, account)
        finally:
            await client.aclose()
        return ids

    async def _usable(self, files: Sequence[StagedFile], account: ProviderAccount) -> dict[str, FileProviderCopy]:
        """The recorded copies of ``files`` in ``account`` with enough life left, by file ID.

        Raises:
            ProviderUploadFailedError: the copies could not be read.
        """
        try:
            async with self._uow:
                return await self._copies.usable(
                    [staged.file_id for staged in files],
                    account.identity,
                    expiring_after=datetime.now(UTC) + _REUSE_MARGIN,
                )
        except DATABASE_ERRORS as exc:
            logger.warning("Could not read the provider copies of %d file(s): %s", len(files), exc)
            raise ProviderUploadFailedError from exc

    async def _copy(self, client: ProviderFileClient, staged: StagedFile, account: ProviderAccount) -> str:
        """Reserve, upload and confirm one copy of ``staged``, and return the provider's ID for it."""
        copy_id = await self._reserve(staged, account)
        try:
            provider_file_id, expires_at = await self._upload(client, staged, account)
        except BaseException:
            await self._cancel(copy_id, staged)
            raise
        try:
            async with self._uow:
                confirmed = await self._copies.confirm(
                    copy_id, provider_file_id=provider_file_id, expires_at=expires_at
                )
        except DATABASE_ERRORS as exc:
            # The provider holds a copy that only a pending row names. The row
            # stays pending until it is stale, and the copy expires on its own.
            logger.warning("Could not confirm the copy of file %s: %s", staged.file_id, exc)
            raise ProviderUploadFailedError from exc
        if not confirmed:
            removed = await client.discard(provider_file_id)
            logger.warning(
                "The reservation for the copy of file %s was gone before it was confirmed; %s",
                staged.file_id,
                "removed the copy" if removed else "the copy could not be removed",
            )
            raise ProviderUploadFailedError
        return provider_file_id

    async def _reserve(self, staged: StagedFile, account: ProviderAccount) -> uuid.UUID:
        """Record a pending copy of ``staged`` in ``account``, before anything is uploaded.

        Raises:
            ProviderUploadFailedError: the reservation could not be recorded.
        """
        copy = FileProviderCopy(
            id=uuid.uuid4(),
            file_id=staged.file_id,
            account_identity=account.identity,
            provider=account.provider.value,
            provider_instance=account.instance,
            credential_workspace_id=account.workspace_id,
            pending_since=datetime.now(UTC),
        )
        try:
            async with self._uow:
                await self._copies.reserve(copy)
        except _RESERVE_ERRORS as exc:
            logger.warning("Could not reserve a copy of file %s: %s", staged.file_id, exc)
            raise ProviderUploadFailedError from exc
        return copy.id

    async def _cancel(self, copy_id: uuid.UUID, staged: StagedFile) -> None:
        """Remove a reservation whose copy the provider does not hold, best effort.

        A reservation that cannot be removed now stays pending until it is stale.
        """
        try:
            async with self._uow:
                await self._copies.cancel(copy_id)
        except DATABASE_ERRORS as exc:
            logger.warning("Could not cancel the reserved copy of file %s: %s", staged.file_id, exc)

    def _lifetime(self, staged: StagedFile, account: ProviderAccount, now: datetime) -> timedelta:
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
            ttl = min(ttl, _as_utc(staged.expires_at) - now - _ACCEPT_SLACK)
        if ttl < minimum_copy_lifetime(account.provider) or ttl <= _REUSE_MARGIN:
            logger.warning(
                "File %s has less life left than provider %s will hold a copy for; "
                "files_retention_hours must exceed that floor for a copy to be possible",
                staged.file_id,
                account.provider.value,
            )
            raise AttachedFileExpiresTooSoonError
        return ttl

    async def _upload(
        self, client: ProviderFileClient, staged: StagedFile, account: ProviderAccount
    ) -> tuple[str, datetime]:
        """Put ``staged``'s bytes at the provider, and return the copy's ID and expiry.

        Raises:
            ProviderUploadFailedError: the bytes could not be read, the provider
                refused them, or it would hold them past the file's own expiry.
        """
        try:
            data = await self._file_store.get(staged.storage_ref)
        except OSError as exc:
            logger.warning("Could not read file %s to copy it to provider %s: %s", staged.file_id, client.provider, exc)
            raise ProviderUploadFailedError from exc
        # Measured after the blob read, because the provider starts the copy's
        # life when it accepts the upload.
        now = datetime.now(UTC)
        ttl = self._lifetime(staged, account, now)
        metadata = await client.upload(
            data, filename=staged.filename, mime_type=staged.mime_type, expires_in=int(ttl.total_seconds())
        )
        expires_at = _as_utc(metadata.expires_at) if metadata.expires_at else now + ttl
        if staged.expires_at is not None and expires_at > _as_utc(staged.expires_at):
            await self._discard_an_overlong_copy(client, staged, metadata.id, expires_at)
        return metadata.id, expires_at

    async def _discard_an_overlong_copy(
        self, client: ProviderFileClient, staged: StagedFile, provider_file_id: str, expires_at: datetime
    ) -> None:
        """Remove a copy the provider will hold past the file's own expiry, and refuse.

        Otari asks for an expiry and cannot make a provider honor it, so a copy
        that would outlive the file is taken back rather than recorded.

        Raises:
            ProviderUploadFailedError: always.
        """
        removed = await client.discard(provider_file_id)
        logger.warning(
            "Provider %s would hold its copy of file %s until %s, past the file's own expiry; %s",
            client.provider,
            staged.file_id,
            expires_at.isoformat(),
            "removed it" if removed else "it could not be removed",
        )
        raise ProviderUploadFailedError
