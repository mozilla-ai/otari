"""Files a provider's own sandbox produced, which Otari serves by proxy.

A provider-native code execution keeps what it wrote in the provider's
container, and answers with the provider's file id. Nothing is copied here:
the run is recorded as a ``file_objects`` row with no ``storage_ref``, naming
the provider that holds the bytes, and ``GET /v1/files/{id}/content`` streams
them from that provider on demand. So the same call serves a chart whichever
sandbox drew it, and a caller swapping one model for another changes nothing
but the model.

The row is what makes that safe. A provider authenticates the deployment's own
credential, which is coarser than a workspace-scoped API key, so without a
record of who the run belonged to, any tenant knowing an id could read another
tenant's output. Reads go through :func:`fetch_file`, which applies the same
user and workspace predicate every other file gets.

Ids stay the provider's throughout. Rewriting them into Otari's own would break
a client that echoes the turn back, since the container reference it carries
would name a file the provider never issued.

The HTTP calls below are hand-rolled because any-llm has no files API; the
request for one is https://github.com/mozilla-ai/any-llm/issues/1419, and the
URLs and headers here move there when it lands.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import httpx
from any_llm import LLMProvider

from gateway.core.config import GatewayConfig, provider_credential_env_names
from gateway.core.unit_of_work import UnitOfWork
from gateway.log_config import logger
from gateway.repositories.files import ProviderFileRow, existing_file_ids, record_provider_file_rows
from gateway.services.file_service import CODE_EXECUTION_OUTPUT_PURPOSE, expiry_for, guess_mime_type
from gateway.services.provider_kwargs import get_provider_kwargs

if TYPE_CHECKING:
    from gateway.models.tools import FileObject

ANTHROPIC_FILES_BASE = "https://api.anthropic.com/v1"
OPENAI_BASE = "https://api.openai.com/v1"
ANTHROPIC_FILES_BETA = "files-api-2025-04-14"
ANTHROPIC_VERSION = "2023-06-01"

# How long to wait on the provider for one file's metadata or bytes.
_TIMEOUT = httpx.Timeout(30.0, read=300.0)


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


def responses_produced_files(result: Any) -> list[ProviderFile]:
    """Provider file ids an OpenAI Responses reply cites from its container.

    OpenAI announces a produced file as a ``container_file_citation``
    annotation on the message it wrote, which carries the container and the
    name as well as the id.
    """
    files: dict[str, ProviderFile] = {}
    for item in getattr(result, "output", None) or []:
        for part in getattr(item, "content", None) or []:
            for note in getattr(part, "annotations", None) or []:
                if getattr(note, "type", None) != "container_file_citation":
                    continue
                file_id = getattr(note, "file_id", None)
                if not isinstance(file_id, str) or not file_id or file_id in files:
                    continue
                files[file_id] = ProviderFile(
                    file_id=file_id,
                    filename=getattr(note, "filename", None),
                    container_id=getattr(note, "container_id", None),
                )
    return list(files.values())


def produced_files_for(dialect: str, obj: Any) -> list[ProviderFile]:
    """Provider file ids in a completed reply, or in one streamed event, of ``dialect``.

    A Messages stream delivers a server tool result whole in its
    ``content_block_start`` event; a Responses stream repeats the entire
    response on ``response.completed``. Anything else (a delta, a chat
    completion, which has no native code tool) names no file.
    """
    kind = getattr(obj, "type", None)
    if dialect == "messages":
        if kind == "content_block_start":
            return _anthropic_files_in([getattr(obj, "content_block", None)])
        return anthropic_produced_files(obj)
    if dialect == "responses":
        if kind == "response.completed":
            return responses_produced_files(getattr(obj, "response", None))
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


def _request_for(record: FileObject, api_key: str, api_base: str | None) -> tuple[str, dict[str, str]]:
    """The URL and headers that read ``record``'s bytes from its provider."""
    if record.provider == LLMProvider.ANTHROPIC.value:
        base = (api_base or ANTHROPIC_FILES_BASE).rstrip("/")
        headers = {
            "x-api-key": api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "anthropic-beta": ANTHROPIC_FILES_BETA,
        }
        return f"{base}/files/{record.id}/content", headers
    base = (api_base or OPENAI_BASE).rstrip("/")
    # OpenAI keys a container file on its container as well as its id.
    if not record.provider_container_id:
        raise LookupError(f"{record.provider} file {record.id} names no container to read it from")
    return (
        f"{base}/containers/{record.provider_container_id}/files/{record.id}/content",
        {"Authorization": f"Bearer {api_key}"},
    )


async def _fetch_filename(provider: str, file_id: str, api_key: str, api_base: str | None) -> str | None:
    """Anthropic's metadata call, for the name its result block leaves out.

    One small JSON read per produced file, so a listing and a download both
    name the file the way the run did. A failure is not fatal: the id still
    downloads, it is just announced under its own id.
    """
    if provider != LLMProvider.ANTHROPIC.value:
        return None
    base = (api_base or ANTHROPIC_FILES_BASE).rstrip("/")
    headers = {
        "x-api-key": api_key,
        "anthropic-version": ANTHROPIC_VERSION,
        "anthropic-beta": ANTHROPIC_FILES_BETA,
    }
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            response = await client.get(f"{base}/files/{file_id}", headers=headers)
            response.raise_for_status()
            name = response.json().get("filename")
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning("Could not read %s metadata for %s: %s", provider, file_id, exc)
        return None
    return name if isinstance(name, str) and name else None


async def record_provider_files(
    uow: UnitOfWork,
    files: list[ProviderFile],
    *,
    provider: str,
    user_id: str,
    workspace_id: uuid.UUID,
    config: GatewayConfig,
    provider_instance: str | None = None,
) -> None:
    """Record what a provider's sandbox produced, so its ids serve from Otari's files API.

    Never fatal: the caller has a completed response in hand, and a file it
    cannot be handed later is a smaller failure than losing the answer. A file
    id already recorded is left alone, so a conversation citing the same chart
    twice keeps one row. An OpenAI file cited without its container is skipped,
    since the download is keyed on the container and the row could never serve.
    """
    files = list({file.file_id: file for file in files}.values())
    if provider == LLMProvider.OPENAI.value:
        files = [file for file in files if file.container_id]
    if not files or not serves_files(provider):
        return
    try:
        api_key, api_base = _credentials(config, provider, provider_instance, workspace_id)
    except (LookupError, ValueError) as exc:
        logger.warning("Not recording %d %s file(s): %s", len(files), provider, exc)
        return

    try:
        async with uow:
            known = await existing_file_ids(uow, [file.file_id for file in files])
        expires_at = expiry_for(config)
        rows = []
        for file in files:
            if file.file_id in known:
                continue
            filename = file.filename or await _fetch_filename(provider, file.file_id, api_key, api_base)
            rows.append(
                ProviderFileRow(
                    file_id=file.file_id,
                    user_id=user_id,
                    workspace_id=workspace_id,
                    filename=filename or file.file_id,
                    mime_type=guess_mime_type(filename),
                    purpose=CODE_EXECUTION_OUTPUT_PURPOSE,
                    provider=provider,
                    provider_instance=provider_instance,
                    container_id=file.container_id,
                    expires_at=expires_at,
                )
            )
        if not rows:
            return
        async with uow:
            await record_provider_file_rows(uow, rows)
    except Exception as exc:  # noqa: BLE001 - a recording failure must not fail the response
        logger.warning("Could not record %d %s file(s): %s", len(files), provider, exc)


async def stream_provider_file(record: FileObject, config: GatewayConfig) -> AsyncGenerator[bytes, None]:
    """Stream a provider-held file's bytes, never holding the whole body.

    Raises ``LookupError`` when the provider cannot be credentialed and
    ``httpx.HTTPError`` when it refuses or fails, which the route maps to its
    own status; the provider's own message never reaches the caller.
    """
    api_key, api_base = _credentials(config, str(record.provider), record.provider_instance, record.workspace_id)
    url, headers = _request_for(record, api_key, api_base)
    async with (
        httpx.AsyncClient(timeout=_TIMEOUT) as client,
        client.stream("GET", url, headers=headers) as response,
    ):
        response.raise_for_status()
        async for chunk in response.aiter_bytes():
            yield chunk
