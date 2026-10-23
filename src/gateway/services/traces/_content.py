"""What a span was about, sealed at rest with ``OTARI_SECRET_KEY``.

The sealed form names the workspace and span it was written for, so a
ciphertext copied onto another row opens as nothing.
"""

import json
import uuid
from collections.abc import Mapping

from gateway.services.secret_box import (
    SecretBoxUnavailableError,
    SecretDecryptionError,
    decrypt_secret,
    encrypt_secret,
)
from gateway.types.traces import SPAN_CONTENT_KEYS, SPAN_CONTENT_MAX_LENGTH


def _payload(content: Mapping[str, str], workspace_id: uuid.UUID, span_id: str) -> str:
    return json.dumps({"workspace_id": str(workspace_id), "span_id": span_id, "content": dict(content)})


def seal_content(content: Mapping[str, str], *, workspace_id: uuid.UUID, span_id: str) -> str | None:
    """The span's content encrypted for its row, or None when there is none or no key to seal it with."""
    if not content:
        return None
    try:
        return encrypt_secret(_payload(content, workspace_id, span_id))
    except SecretBoxUnavailableError:
        return None


def open_content(sealed: str | None, *, workspace_id: uuid.UUID, span_id: str) -> dict[str, str]:
    """The content sealed for this row; empty when none was, or no configured key opens it."""
    if not sealed:
        return {}
    try:
        payload = json.loads(decrypt_secret(sealed))
    except (SecretBoxUnavailableError, SecretDecryptionError, ValueError):
        return {}
    if not isinstance(payload, dict) or (payload.get("workspace_id"), payload.get("span_id")) != (
        str(workspace_id),
        span_id,
    ):
        return {}
    content = payload.get("content")
    if not isinstance(content, dict):
        return {}
    return {
        key: value
        for key, value in content.items()
        if key in SPAN_CONTENT_KEYS and isinstance(value, str) and len(value) <= SPAN_CONTENT_MAX_LENGTH
    }
