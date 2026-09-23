"""Files a provider's own sandbox produced, and the client that reads them back.

A provider-native code execution keeps what it wrote in the provider's container
and answers with the provider's file ID.
The provider does not keep it for long: OpenAI discards a container 20 minutes after its last use.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncGenerator, AsyncIterator, Iterable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any
from urllib.parse import quote

import httpx
from anthropic import AnthropicError
from any_llm import AnyLLM, LLMProvider
from any_llm.exceptions import AnyLLMError
from any_llm.types.files import AsyncFileDownload
from pydantic import ValidationError

from gateway.core.config import GatewayConfig, provider_credential_env_names
from gateway.log_config import logger
from gateway.services.provider_kwargs import get_provider_kwargs

OPENAI_BASE = "https://api.openai.com/v1"

# How long to wait on the provider for a connection or for the next chunk.
_TIMEOUT = httpx.Timeout(30.0)

# The providers a produced file can be read back from.
_FILE_PROVIDERS = frozenset({LLMProvider.ANTHROPIC, LLMProvider.OPENAI})

# How a file call fails. any-llm raises its own error, or re-raises the provider
# SDK's while unified exceptions are off, and the OpenAI container read is httpx.
_FILE_CALL_ERRORS = (AnyLLMError, AnthropicError, httpx.HTTPError, ValidationError)


@dataclass(frozen=True)
class ProviderFile:
    """One file a provider's sandbox produced, as its response announced it."""

    file_id: str
    filename: str | None = None
    container_id: str | None = None


def _anthropic_files_in(blocks: list[Any]) -> list[ProviderFile]:
    files: dict[str, ProviderFile] = {}
    for block in blocks:
        content = getattr(block, "content", None)
        for output in getattr(content, "content", None) or []:
            file_id = getattr(output, "file_id", None)
            if isinstance(file_id, str) and file_id and file_id not in files:
                files[file_id] = ProviderFile(file_id=file_id)
    return list(files.values())


def anthropic_produced_files(result: Any) -> list[ProviderFile]:
    """Provider file ids in an Anthropic Messages response's tool results.

    Both the python and the bash variant of the tool name their outputs in a
    block of their own, and neither gives a filename, so the shape is what this
    looks for rather than a block name.
    """
    return _anthropic_files_in(list(getattr(result, "content", None) or []))


def _field(obj: Any, name: str) -> Any:
    """``obj``'s ``name``, whether a stream event carried it as an object or as a plain dict."""
    return obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)


def _cited_files(annotations: Iterable[Any]) -> list[ProviderFile]:
    """The files the ``container_file_citation`` annotations among ``annotations`` name, each once."""
    files: dict[str, ProviderFile] = {}
    for note in annotations:
        if _field(note, "type") != "container_file_citation":
            continue
        file_id = _field(note, "file_id")
        if not isinstance(file_id, str) or not file_id or file_id in files:
            continue
        filename, container_id = _field(note, "filename"), _field(note, "container_id")
        files[file_id] = ProviderFile(
            file_id=file_id,
            filename=filename if isinstance(filename, str) and filename else None,
            container_id=container_id if isinstance(container_id, str) and container_id else None,
        )
    return list(files.values())


def _annotations_in(parts: Any) -> list[Any]:
    return [note for part in parts or [] for note in _field(part, "annotations") or []]


def responses_produced_files(result: Any) -> list[ProviderFile]:
    """Provider file ids an OpenAI Responses reply cites from its container.

    OpenAI announces a produced file as a ``container_file_citation``
    annotation on the message it wrote, which carries the container and the
    name as well as the id.
    """
    items = _field(result, "output") or []
    return _cited_files(note for item in items for note in _annotations_in(_field(item, "content")))


def produced_files_for(dialect: str, obj: Any) -> list[ProviderFile]:
    """Provider file ids in a completed reply, or in one streamed event, of ``dialect``.

    A Messages stream delivers a server tool result whole in its
    ``content_block_start`` event. A Responses stream names a cited file in the
    annotation, content part and output item events before it repeats the whole
    response on ``response.completed``. Anything else (a delta, a chat
    completion, which has no native code tool) names no file.
    """
    kind = getattr(obj, "type", None)
    if dialect == "messages":
        if kind == "content_block_start":
            return _anthropic_files_in([getattr(obj, "content_block", None)])
        return anthropic_produced_files(obj)
    if dialect == "responses":
        if kind == "response.output_text.annotation.added":
            return _cited_files([_field(obj, "annotation")])
        if kind == "response.content_part.done":
            return _cited_files(_annotations_in([_field(obj, "part")]))
        if kind == "response.output_item.done":
            return _cited_files(_annotations_in(_field(_field(obj, "item"), "content")))
        if kind == "response.completed":
            return responses_produced_files(_field(obj, "response"))
        return responses_produced_files(obj)
    return []


def serves_files(provider: str) -> bool:
    """Whether Otari knows how to fetch a produced file back from ``provider``."""
    try:
        return LLMProvider(provider) in _FILE_PROVIDERS
    except ValueError:
        return False


@dataclass(frozen=True)
class ProviderCredential:
    """What one configured provider instance calls its provider with."""

    api_key: str
    api_base: str | None = None
    client_args: dict[str, Any] = field(default_factory=dict)


def _credentials(
    config: GatewayConfig, provider: LLMProvider, instance: str | None, workspace_id: uuid.UUID | None
) -> ProviderCredential:
    """What ``provider`` is called with to read its files.

    ``instance`` is the configured entry the run dispatched through, so a named
    instance's own settings are the ones used to read back what it produced.
    Falls back to the provider SDK's own environment variable for the key, which
    is how a config with an empty provider stanza is credentialed for dispatch
    too.
    """
    kwargs = get_provider_kwargs(config, provider, instance, workspace_id=workspace_id)
    api_key = kwargs.get("api_key")
    if not api_key:
        for name in provider_credential_env_names(provider.value) or ():
            if value := os.environ.get(name):
                api_key = value
                break
    if not api_key:
        raise LookupError(f"no credential configured for provider '{provider.value}'")
    return ProviderCredential(
        api_key=str(api_key), api_base=kwargs.get("api_base"), client_args=dict(kwargs.get("client_args") or {})
    )


class FileOverBudgetError(Exception):
    """A file ran past the bytes the reply may still store."""


class ProviderFileUnavailableError(Exception):
    """The provider cannot serve a file now: it refused, the connection failed, or the file names no container."""


def _declared_size(download: AsyncFileDownload) -> int:
    """The size the provider announced, or ``0`` when it announced none.

    ``AsyncFileDownload`` promises a plain mapping, so the name is matched
    case-insensitively rather than relying on HTTP-aware header lookup.
    """
    for name, value in download.headers.items():
        if name.lower() == "content-length" and value.isdigit():
            return int(value)
    return 0


def _container_file_request(file: ProviderFile, credential: ProviderCredential) -> tuple[str, dict[str, str]]:
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


class ProviderFileClient:
    """Reads back the files a provider instance's code produced, with that instance's credential.

    Owns the connection its reads run on, so a caller closes it with
    :meth:`aclose` once it has read everything it wants.
    """

    def __init__(self, *, provider: LLMProvider, provider_instance: str, credential: ProviderCredential) -> None:
        if provider not in _FILE_PROVIDERS:
            raise LookupError(f"otari cannot read files back from provider '{provider.value}'")
        self._provider = provider
        self._provider_instance = provider_instance
        self._credential = credential
        # Handing this client to the provider SDK replaces the one it would have
        # built, whose own default is to follow the redirect a download can answer with.
        self._connection = httpx.AsyncClient(timeout=_TIMEOUT, follow_redirects=True)

    @classmethod
    def for_run(
        cls, config: GatewayConfig, *, provider: str, provider_instance: str, workspace_id: uuid.UUID | None
    ) -> ProviderFileClient:
        """The client for the configured instance a run was dispatched through.

        Raises ``LookupError`` for a provider whose files Otari cannot read, and
        when the deployment holds no credential for one it can.
        """
        member = LLMProvider(provider)
        credential = _credentials(config, member, provider_instance, workspace_id)
        return cls(provider=member, provider_instance=provider_instance, credential=credential)

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

    async def get_filename(self, file_id: str) -> str | None:
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
