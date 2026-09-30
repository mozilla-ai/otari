"""The core ``ProviderFilePort``: a provider's files API reached through any-llm.

OpenAI's container files are the one read any-llm has no call for, so that read
goes over this adapter's own connection.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from functools import cached_property
from urllib.parse import quote

import httpx
from anthropic import AnthropicError
from any_llm import AnyLLM, LLMProvider
from any_llm.exceptions import AnyLLMError
from any_llm.types.files import AsyncFileDownload
from pydantic import ValidationError

from gateway.exceptions.files_exceptions import (
    FileOverBudgetError,
    ProviderFileUnavailableError,
    ProviderUploadFailedError,
)
from gateway.log_config import logger
from gateway.ports.provider_file_port import ProviderFilePort, ProviderFileSession
from gateway.types.provider_account import ResolvedCredential
from gateway.types.provider_file import ProviderCopyReceipt, ProviderFile

__all__ = ["AnyLlmProviderFiles"]

OPENAI_BASE = "https://api.openai.com/v1"

# How long to wait on the provider for a connection or for the next chunk.
_TIMEOUT = httpx.Timeout(30.0)

# The providers whose files this adapter reads and writes.
_FILE_PROVIDERS = frozenset({LLMProvider.ANTHROPIC, LLMProvider.OPENAI})

# How a file call fails. any-llm raises its own error, or re-raises the provider
# SDK's while unified exceptions are off, and the OpenAI container read is httpx.
_FILE_CALL_ERRORS = (AnyLLMError, AnthropicError, httpx.HTTPError, ValidationError)


class AnyLlmProviderFiles(ProviderFilePort):
    """Opens any-llm sessions over the files of Anthropic and OpenAI accounts."""

    def serves(self, provider: LLMProvider) -> bool:
        return provider in _FILE_PROVIDERS

    def open_session(
        self, *, provider: LLMProvider, instance: str, credential: ResolvedCredential
    ) -> ProviderFileSession:
        return _AnyLlmFileSession(provider=provider, provider_instance=instance, credential=credential)


def _declared_size(download: AsyncFileDownload) -> int:
    """The size the provider announced, or ``0`` when it announced none.

    ``AsyncFileDownload`` promises a plain mapping, so the name is matched
    case-insensitively rather than relying on HTTP-aware header lookup.
    """
    for name, value in download.headers.items():
        if name.lower() == "content-length" and value.isdigit():
            return int(value)
    return 0


def _container_file_request(file: ProviderFile, credential: ResolvedCredential) -> tuple[str, dict[str, str]]:
    """The URL and headers that read ``file``'s bytes out of its OpenAI container.

    Raises :class:`ProviderFileUnavailableError` for a file that names no container.
    """
    # OpenAI keys a container file on its container as well as its ID.
    if not file.container_id:
        raise ProviderFileUnavailableError(f"openai file {file.file_id} names no container to read it from")
    base = (credential.api_base or OPENAI_BASE).rstrip("/")
    return (
        f"{base}/containers/{quote(file.container_id, safe='')}/files/{quote(file.file_id, safe='')}/content",
        {"Authorization": f"Bearer {credential.api_key}"},
    )


class _AnyLlmFileSession:
    """Reads and writes one provider account's files through any-llm, with the credential that reaches it.

    Owns the connection its calls run on, so a caller closes it with
    :meth:`aclose` once it has read everything it wants.
    """

    def __init__(self, *, provider: LLMProvider, provider_instance: str, credential: ResolvedCredential) -> None:
        if provider not in _FILE_PROVIDERS:
            raise LookupError(f"otari cannot read files back from provider '{provider.value}'")
        self._provider = provider
        self._provider_instance = provider_instance
        self._credential = credential
        # Handing this client to the provider SDK replaces the one it would have
        # built, whose own default is to follow the redirect a download can answer with.
        self._connection = httpx.AsyncClient(timeout=_TIMEOUT, follow_redirects=True)

    @property
    def provider(self) -> str:
        return self._provider.value

    @property
    def provider_instance(self) -> str:
        return self._provider_instance

    @cached_property
    def _llm(self) -> AnyLLM:
        """The any-llm instance this client's file operations run on."""
        return AnyLLM.create(
            self._provider.value,
            **{
                **self._credential.client_args,
                "api_key": self._credential.api_key,
                "api_base": self._credential.api_base,
                # The connection and the budget a read runs under belong to this
                # client, so an instance's own transport settings do not reach
                # them. A caller shares one deadline across every file it reads,
                # so a retry or a longer wait here spends the time the files
                # after this one need.
                "http_client": self._connection,
                "timeout": _TIMEOUT,
                "max_retries": 0,
            },
        )

    async def aclose(self) -> None:
        """Release the connection this client reads over.

        A release that fails is logged rather than raised, because the caller has
        already read whatever it came for and can do nothing about it.
        """
        try:
            await self._connection.aclose()
        except (httpx.HTTPError, RuntimeError):
            logger.exception("Could not release the %s connection", self.provider)

    async def upload(self, data: bytes, *, filename: str, mime_type: str, expires_in: int) -> ProviderCopyReceipt:
        """Store ``data`` at the provider for ``expires_in`` seconds, and return what it recorded.

        The provider expires the copy itself, which is what keeps Otari's store
        the one place a file is kept indefinitely.

        Raises:
            ProviderUploadFailedError: the provider refused the copy, has no
                files API, or could not be reached.
        """
        try:
            metadata = await self._llm.aupload_file(data, filename=filename, mime_type=mime_type, expires_in=expires_in)
        except (*_FILE_CALL_ERRORS, NotImplementedError) as exc:
            # The class and status only: a provider's error body can echo what it was sent.
            logger.warning(
                "Provider %s refused a copy of %s: %s (status %s)",
                self.provider,
                filename,
                type(exc).__name__,
                getattr(exc, "status_code", None),
            )
            raise ProviderUploadFailedError from exc
        return ProviderCopyReceipt(file_id=metadata.id, expires_at=metadata.expires_at)

    async def discard(self, file_id: str) -> bool:
        """Remove a file this client put at the provider, and say whether it is gone.

        False rather than raising, because a copy that cannot be removed expires
        at the provider on its own.
        """
        try:
            await self._llm.adelete_file(file_id)
        except (*_FILE_CALL_ERRORS, NotImplementedError) as exc:
            logger.warning("Could not remove %s file %s: %s", self.provider, file_id, exc)
            return False
        return True

    async def filename_of(self, file_id: str) -> str | None:
        """The file's name, from Anthropic's file metadata.

        Anthropic's result block leaves the name out.
        ``None`` for any other provider, or when the lookup fails.
        """
        if self._provider is not LLMProvider.ANTHROPIC:
            return None
        try:
            metadata = await self._llm.aretrieve_file(file_id)
        except _FILE_CALL_ERRORS as exc:
            logger.warning("Could not read %s metadata for %s: %s", self.provider, file_id, exc)
            return None
        except Exception:  # noqa: BLE001 - a missing name is not a failure, so no exception escapes
            logger.exception("Unexpected failure reading %s metadata for %s", self.provider, file_id)
            return None
        return metadata.filename or None

    async def read(self, file: ProviderFile, *, budget_bytes: int) -> AsyncGenerator[bytes, None]:
        """Stream one file's bytes, raising :class:`FileOverBudgetError` past ``budget_bytes``.

        A declared size past the budget is refused before a byte is read, when the provider sends one.
        Raises :class:`ProviderFileUnavailableError` when the provider cannot serve the file.
        """
        try:
            async with self._open(file) as download:
                if _declared_size(download) > budget_bytes:
                    raise FileOverBudgetError
                total = 0
                async for chunk in download:
                    total += len(chunk)
                    if total > budget_bytes:
                        raise FileOverBudgetError
                    yield chunk
        except _FILE_CALL_ERRORS as exc:
            raise ProviderFileUnavailableError(f"{self.provider} could not serve file {file.file_id}") from exc

    def _open(self, file: ProviderFile) -> AbstractAsyncContextManager[AsyncFileDownload]:
        if self._provider is LLMProvider.ANTHROPIC:
            return self._llm.adownload_file(file.file_id)
        return self._open_container_file(file)

    @asynccontextmanager
    async def _open_container_file(self, file: ProviderFile) -> AsyncIterator[AsyncFileDownload]:
        """Open an OpenAI container file's download, which any-llm has no call for.

        Stopgap: read it through any-llm once that can reach a container's files
        (mozilla-ai/any-llm#1419, tracked in #1480).
        """
        url, headers = _container_file_request(file, self._credential)
        async with self._connection.stream("GET", url, headers=headers) as response:
            response.raise_for_status()
            yield AsyncFileDownload(
                status_code=response.status_code, headers=response.headers, chunks=response.aiter_bytes()
            )
