"""Moving an attachment into a provider's own file storage, for a provider that limits inline data.

Gemini refuses a request past about 20 MB, so a larger attachment is uploaded
through its Files API and referenced by URI. An agent resends its whole
conversation every turn, so an upload is remembered for the life of the
provider's copy rather than repeated per request. Nothing is recorded in the
database: the provider expires the copy on its own, and a forgotten URI only
costs one more upload.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
import uuid
from collections import OrderedDict

from any_llm import LLMProvider

from gateway.exceptions.files_exceptions import ProviderUploadFailedError
from gateway.log_config import logger
from gateway.ports.provider_file_port import ProviderFilePort
from gateway.types.provider_account import ResolvedCredential

# The providers whose file storage an attachment can be moved into. Vertex AI
# shares the Gemini request format but has no Files API in any-llm.
_UPLOAD_PROVIDERS = frozenset({LLMProvider.GEMINI})

# Gemini keeps an uploaded file for 48 hours; an hour's margin keeps a
# remembered URI from expiring between the lookup and the provider call.
_REMEMBER_SECONDS = 47 * 3600
_REMEMBERED_MAX = 512

# The longest one attachment may take to reach the provider and become usable.
# The request waits on it before dispatch, so it is bounded as a whole.
_UPLOAD_SECONDS = 300.0

# Keys the remembered URIs on the account a credential reaches without keeping
# the credential, so the memory holds no key. Per process, as the memory is.
_ACCOUNT_KEY = os.urandom(32)

_remembered: OrderedDict[tuple[str, str, str, str], tuple[str, float]] = OrderedDict()


def uploads_attachments(provider: LLMProvider | str | None) -> bool:
    """Whether Otari can move an attachment into ``provider``'s own file storage."""
    try:
        return provider is not None and LLMProvider(provider) in _UPLOAD_PROVIDERS
    except ValueError:
        return False


def _account(provider: LLMProvider, credential: ResolvedCredential) -> str:
    """The account ``credential`` reaches, the one an uploaded URI is readable by."""
    canonical = json.dumps(
        {"provider": provider.value, "api_base": credential.api_base, "api_key": credential.api_key},
        sort_keys=True,
        separators=(",", ":"),
    )
    # scrypt rather than a fast hash because the input holds an API key. It is only a lookup key, so the
    # cost is kept low; the per-process salt is what keeps it from being matched against known keys.
    return hashlib.scrypt(canonical.encode(), salt=_ACCOUNT_KEY, n=2**12, r=8, p=1, dklen=32).hex()


def _recall(key: tuple[str, str, str, str]) -> str | None:
    entry = _remembered.get(key)
    if entry is None:
        return None
    uri, expires = entry
    if expires <= time.monotonic():
        del _remembered[key]
        return None
    _remembered.move_to_end(key)
    return uri


def _remember(key: tuple[str, str, str, str], uri: str) -> None:
    _remembered[key] = (uri, time.monotonic() + _REMEMBER_SECONDS)
    _remembered.move_to_end(key)
    while len(_remembered) > _REMEMBERED_MAX:
        _remembered.popitem(last=False)


async def upload_attachment(
    provider_files: ProviderFilePort,
    *,
    provider: LLMProvider,
    credential: ResolvedCredential,
    workspace_id: uuid.UUID | None,
    data: bytes,
    mime_type: str,
    filename: str,
) -> str:
    """Upload ``data`` with ``credential`` and return the URI a request references it by.

    Remembered per account and workspace as well as per content and MIME type,
    because a URI is only readable by the account that uploaded it, one workspace
    must not learn what another has uploaded, and the upload carries its type.

    Raises:
        ProviderUploadFailedError: the provider refused the file, left it
            unusable, or took longer than ``_UPLOAD_SECONDS``.
    """
    if provider not in _UPLOAD_PROVIDERS:
        raise ProviderUploadFailedError
    account = await asyncio.to_thread(_account, provider, credential)
    key = (account, str(workspace_id or ""), mime_type, hashlib.sha256(data).hexdigest())
    if (uri := _recall(key)) is not None:
        return uri
    try:
        async with asyncio.timeout(_UPLOAD_SECONDS):
            uri = await provider_files.upload_attachment(
                provider=provider, credential=credential, data=data, filename=filename, mime_type=mime_type
            )
    except TimeoutError as exc:
        logger.warning(
            "Provider %s took over %ss to accept a %d byte attachment", provider.value, _UPLOAD_SECONDS, len(data)
        )
        raise ProviderUploadFailedError from exc
    _remember(key, uri)
    return uri
