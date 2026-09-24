"""Moving an attachment into a provider's own file storage, for a provider that limits inline data.

Gemini refuses a request whose inline data passes about 20 MB, so a larger
attachment is uploaded through its Files API and referenced by URI. An agent
resends its whole conversation every turn, so an upload is remembered for the
life of the provider's copy rather than repeated per request.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
import uuid
from collections import OrderedDict

from any_llm import AnyLLM, LLMProvider

from gateway.core.config import GatewayConfig
from gateway.services.files.provider_files import _credentials

# The providers whose Files API an attachment can be moved into. Vertex AI
# shares the Gemini request format but has no Files API in any-llm.
_UPLOAD_PROVIDERS = frozenset({LLMProvider.GEMINI})

# Gemini keeps an uploaded file for 48 hours; an hour's margin keeps a
# remembered URI from expiring between the lookup and the provider call.
_REMEMBER_SECONDS = 47 * 3600
_REMEMBERED_MAX = 512

# How long to wait for Gemini to finish processing an upload (video and large
# PDFs are processed before they can be referenced), and how often to ask.
_PROCESSING_SECONDS = 120.0
_POLL_SECONDS = 1.0

_remembered: OrderedDict[tuple[str, str, str], tuple[str, float]] = OrderedDict()


class ProviderUploadError(Exception):
    """The provider did not accept an attachment, or never made it usable."""


def uploads_attachments(provider: LLMProvider | str | None) -> bool:
    """Whether Otari can move an attachment into ``provider``'s own file storage."""
    try:
        return provider is not None and LLMProvider(provider) in _UPLOAD_PROVIDERS
    except ValueError:
        return False


def _recall(key: tuple[str, str, str]) -> str | None:
    entry = _remembered.get(key)
    if entry is None:
        return None
    uri, expires = entry
    if expires <= time.monotonic():
        del _remembered[key]
        return None
    _remembered.move_to_end(key)
    return uri


def _remember(key: tuple[str, str, str], uri: str) -> None:
    _remembered[key] = (uri, time.monotonic() + _REMEMBER_SECONDS)
    _remembered.move_to_end(key)
    while len(_remembered) > _REMEMBERED_MAX:
        _remembered.popitem(last=False)


async def upload_attachment(
    config: GatewayConfig,
    *,
    provider: LLMProvider,
    instance: str | None,
    workspace_id: uuid.UUID | None,
    data: bytes,
    mime: str,
    filename: str,
) -> str:
    """Upload ``data`` with the instance's own credential and return the URI a request references it by.

    Keyed on the instance and the workspace as well as the content, because a
    URI is only readable with the account that uploaded it.

    Raises :class:`ProviderUploadError` when the provider refuses the file or
    leaves it unusable, and ``LookupError`` when no credential is configured.
    """
    key = (instance or provider.value, str(workspace_id or ""), hashlib.sha256(data).hexdigest())
    if (uri := _recall(key)) is not None:
        return uri
    credential = _credentials(config, provider, instance, workspace_id)
    llm = AnyLLM.create(
        provider.value, api_key=credential.api_key, api_base=credential.api_base, **credential.client_args
    )
    metadata = await llm.aupload_file(data, filename=filename or None, mime_type=mime)
    deadline = time.monotonic() + _PROCESSING_SECONDS
    while metadata.status == "PROCESSING":
        if time.monotonic() >= deadline:
            raise ProviderUploadError(f"{provider.value} was still processing {metadata.id} after the wait")
        await asyncio.sleep(_POLL_SECONDS)
        metadata = await llm.aretrieve_file(metadata.id)
    if metadata.status not in (None, "ACTIVE"):
        raise ProviderUploadError(f"{provider.value} left {metadata.id} in state {metadata.status}")
    uri = (metadata.model_extra or {}).get("uri")
    if not isinstance(uri, str) or not uri:
        raise ProviderUploadError(f"{provider.value} returned no URI for {metadata.id}")
    _remember(key, uri)
    return uri
