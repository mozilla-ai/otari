"""Files a provider's own sandbox produced, and the client that reads them back.

A provider-native code execution keeps what it wrote in the provider's
container and answers with the provider's file ID. The provider does not keep
it for long: OpenAI discards a container 20 minutes after its last use.

The HTTP calls below are hand-rolled because any-llm cannot read a container's
files yet; the request for that is
https://github.com/mozilla-ai/any-llm/issues/1419.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncGenerator, Iterable
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import httpx
from any_llm import LLMProvider

from gateway.core.config import GatewayConfig, provider_credential_env_names
from gateway.log_config import logger
from gateway.services.provider_kwargs import get_provider_kwargs

ANTHROPIC_FILES_BASE = "https://api.anthropic.com/v1"
OPENAI_BASE = "https://api.openai.com/v1"
ANTHROPIC_FILES_BETA = "files-api-2025-04-14"
ANTHROPIC_VERSION = "2023-06-01"

# How long to wait on the provider for a connection or for the next chunk.
_TIMEOUT = httpx.Timeout(30.0)


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
    return provider in (LLMProvider.ANTHROPIC.value, LLMProvider.OPENAI.value)


def _credentials(
    config: GatewayConfig, provider: str, instance: str | None, workspace_id: uuid.UUID | None
) -> tuple[str, str | None]:
    """The API key and base URL to read ``provider``'s files with.

    ``instance`` is the configured entry the run dispatched through, so a named
    instance's own key and base URL are the ones used to read back what it
    produced. Falls back to the provider SDK's own environment variable, which
    is how a config with an empty provider stanza is credentialed for dispatch
    too.
    """
    member = LLMProvider(provider)
    kwargs = get_provider_kwargs(config, member, instance, workspace_id=workspace_id)
    api_key = kwargs.get("api_key")
    if not api_key:
        # An empty provider stanza is credentialed by the SDK's own variable,
        # which is how the dispatch that produced the file was credentialed too.
        for name in provider_credential_env_names(provider) or ():
            if value := os.environ.get(name):
                api_key = value
                break
    if not api_key:
        raise LookupError(f"no credential configured for provider '{provider}'")
    return str(api_key), kwargs.get("api_base")


def _anthropic_headers(api_key: str) -> dict[str, str]:
    return {
        "x-api-key": api_key,
        "anthropic-version": ANTHROPIC_VERSION,
        "anthropic-beta": ANTHROPIC_FILES_BETA,
    }


def _request_for(provider: str, file: ProviderFile, api_key: str, api_base: str | None) -> tuple[str, dict[str, str]]:
    """The URL and headers that read ``file``'s bytes from ``provider``."""
    file_id = quote(file.file_id, safe="")
    if provider == LLMProvider.ANTHROPIC.value:
        base = (api_base or ANTHROPIC_FILES_BASE).rstrip("/")
        return f"{base}/files/{file_id}/content", _anthropic_headers(api_key)
    base = (api_base or OPENAI_BASE).rstrip("/")
    # OpenAI keys a container file on its container as well as its ID.
    if not file.container_id:
        raise LookupError(f"{provider} file {file.file_id} names no container to read it from")
    return (
        f"{base}/containers/{quote(file.container_id, safe='')}/files/{file_id}/content",
        {"Authorization": f"Bearer {api_key}"},
    )


class FileOverBudgetError(Exception):
    """A file ran past the bytes the reply may still store."""


class ProviderFileUnavailableError(Exception):
    """The provider cannot serve a file now: it refused, the connection failed, or the file names no container."""


class ProviderFileClient:
    """Reads back the files a provider instance's code produced, with that instance's credential."""

    def __init__(self, *, provider: str, provider_instance: str, api_key: str, api_base: str | None) -> None:
        self._provider = provider
        self._provider_instance = provider_instance
        self._api_key = api_key
        self._api_base = api_base

    @classmethod
    def for_run(
        cls, config: GatewayConfig, *, provider: str, provider_instance: str, workspace_id: uuid.UUID | None
    ) -> ProviderFileClient:
        """The client for the configured instance a run was dispatched through.

        Raises ``LookupError`` when the deployment holds no credential for ``provider``.
        """
        api_key, api_base = _credentials(config, provider, provider_instance, workspace_id)
        return cls(provider=provider, provider_instance=provider_instance, api_key=api_key, api_base=api_base)

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def provider_instance(self) -> str:
        return self._provider_instance

    async def get_filename(self, file_id: str) -> str | None:
        """The file's name, from Anthropic's file metadata.

        Anthropic's result block leaves the name out.
        ``None`` for any other provider, or when the lookup fails.
        """
        if self._provider != LLMProvider.ANTHROPIC.value:
            return None
        base = (self._api_base or ANTHROPIC_FILES_BASE).rstrip("/")
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                response = await client.get(
                    f"{base}/files/{quote(file_id, safe='')}", headers=_anthropic_headers(self._api_key)
                )
                response.raise_for_status()
                metadata = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning("Could not read %s metadata for %s: %s", self._provider, file_id, exc)
            return None
        name = metadata.get("filename") if isinstance(metadata, dict) else None
        return name if isinstance(name, str) and name else None

    async def read(self, file: ProviderFile, *, budget_bytes: int) -> AsyncGenerator[bytes, None]:
        """Stream one file's bytes, raising :class:`FileOverBudgetError` past ``budget_bytes``.

        A declared size past the budget is refused before a byte is read, when the provider sends one.
        Raises :class:`ProviderFileUnavailableError` when the provider cannot serve the file.
        """
        try:
            url, headers = _request_for(self._provider, file, self._api_key, self._api_base)
        except LookupError as exc:
            raise ProviderFileUnavailableError(str(exc)) from exc
        try:
            async with (
                httpx.AsyncClient(timeout=_TIMEOUT) as client,
                client.stream("GET", url, headers=headers) as response,
            ):
                response.raise_for_status()
                declared = response.headers.get("content-length", "")
                if declared.isdigit() and int(declared) > budget_bytes:
                    raise FileOverBudgetError
                total = 0
                async for chunk in response.aiter_bytes():
                    total += len(chunk)
                    if total > budget_bytes:
                        raise FileOverBudgetError
                    yield chunk
        except httpx.HTTPError as exc:
            raise ProviderFileUnavailableError(f"{self._provider} could not serve file {file.file_id}") from exc

