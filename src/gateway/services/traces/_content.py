"""What a span was about, sealed at rest under a data key from :class:`ContentKeyPort`.

The one place that knows how content is sealed. A sealed value reads::

    otc1.<wrapper>.<wrapped data key>.<nonce + AES-256-GCM ciphertext>

each part url-safe base64. The data key is made for the workspace, which the
key source binds (KMS as encryption context), and the ciphertext is
authenticated with the workspace and the span as associated data. So a value
copied onto another span, or into another workspace, opens as nothing.

A value without the ``otc1.`` prefix is a bare ``OTARI_SECRET_KEY`` Fernet token,
the form content was stored in before; it opens as it always did.

Data keys are cached, so a busy workspace costs its key source one call every
few minutes rather than one a span: one write key per workspace, reused for
``_WRITE_KEY_MAX_AGE_S`` or ``_WRITE_KEY_MAX_USES`` seals, and an LRU of opened
keys. After the key source fails for lack of reach or permission, it is not
asked again for ``_PAUSE_AFTER_FAILURE_S``: content is dropped, or reads back
empty, rather than every batch waiting on a source that is down.
"""

import base64
import binascii
import json
import os
import uuid
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from time import monotonic

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from gateway.log_config import logger
from gateway.metrics import REGISTRY, Counter
from gateway.ports.content_key_port import ContentKeyError, ContentKeyPort, DataKey
from gateway.services.secret_box import SecretBoxUnavailableError, SecretDecryptionError, decrypt_secret
from gateway.types.traces import SPAN_CONTENT_KEYS, SPAN_CONTENT_MAX_LENGTH

_FORMAT = "otc1"
_NONCE_BYTES = 12
_PURPOSE = "trace-content"

_WRITE_KEY_MAX_AGE_S = 300.0
_WRITE_KEY_MAX_USES = 10_000
_OPEN_KEY_MAX_AGE_S = 300.0
_OPEN_KEY_CACHE_SIZE = 1_024
_PAUSE_AFTER_FAILURE_S = 30.0

UNSEALED = Counter(
    "gateway_trace_content_unsealed",
    "Spans stored without their content because it could not be sealed, by reason",
    ["reason"],
    registry=REGISTRY,
)
UNOPENED = Counter(
    "gateway_trace_content_unopened",
    "Stored span content that read back empty because it could not be opened, by reason",
    ["reason"],
    registry=REGISTRY,
)


@dataclass
class _WriteKey:
    key: DataKey
    made_at: float
    uses: int = 0


class SpanContent:
    """Seals and opens span content under the data keys ``keys`` makes and opens.

    One per process, so its caches serve every request and the trace writer.
    """

    def __init__(self, keys: ContentKeyPort | None) -> None:
        self._keys = keys
        self._write_keys: dict[str, _WriteKey] = {}
        self._open_keys: OrderedDict[tuple[str, bytes, str], tuple[bytes, float]] = OrderedDict()
        self._paused_until = 0.0

    @classmethod
    def disabled(cls) -> "SpanContent":
        """One that seals nothing and opens only legacy values, for work that never reads content."""
        return cls(None)

    async def seal(self, content: Mapping[str, str], *, workspace_id: uuid.UUID, span_id: str) -> str | None:
        """The span's content sealed for its row; None when there is none, or it could not be sealed."""
        if not content or self._keys is None or self._keys.wrapper is None:
            return None
        if self._paused():
            UNSEALED.labels(reason="paused").inc()
            return None
        try:
            key = await self._write_key(self._keys, workspace_id)
        except ContentKeyError as exc:
            self._failed(exc, "sealed")
            UNSEALED.labels(reason=exc.reason).inc()
            return None
        nonce = os.urandom(_NONCE_BYTES)
        ciphertext = AESGCM(key.plaintext).encrypt(
            nonce, json.dumps(dict(content)).encode(), _associated_data(workspace_id, span_id)
        )
        return ".".join((_FORMAT, key.wrapper, _b64(key.wrapped), _b64(nonce + ciphertext)))

    async def open(self, sealed: str | None, *, workspace_id: uuid.UUID, span_id: str) -> dict[str, str]:
        """The content sealed for this row; empty when none was, or it cannot be opened."""
        if not sealed:
            return {}
        if not sealed.startswith(_FORMAT + "."):
            return _open_legacy(sealed, workspace_id=workspace_id, span_id=span_id)
        parts = sealed.split(".")
        try:
            if len(parts) != 4:
                raise ValueError
            _, wrapper, wrapped, body = parts[0], parts[1], _unb64(parts[2]), _unb64(parts[3])
        except (binascii.Error, ValueError):
            UNOPENED.labels(reason="rejected").inc()
            return {}
        try:
            data_key = await self._open_key(wrapper, wrapped, workspace_id)
        except ContentKeyError as exc:
            self._failed(exc, "opened")
            UNOPENED.labels(reason=exc.reason).inc()
            return {}
        try:
            plaintext = AESGCM(data_key).decrypt(
                body[:_NONCE_BYTES], body[_NONCE_BYTES:], _associated_data(workspace_id, span_id)
            )
            content = json.loads(plaintext)
        except (InvalidTag, ValueError):
            UNOPENED.labels(reason="rejected").inc()
            return {}
        return _allowed(content)

    async def _write_key(self, keys: ContentKeyPort, workspace_id: uuid.UUID) -> DataKey:
        now = monotonic()
        workspace = str(workspace_id)
        held = self._write_keys.get(workspace)
        if held is None or now - held.made_at >= _WRITE_KEY_MAX_AGE_S or held.uses >= _WRITE_KEY_MAX_USES:
            held = _WriteKey(await keys.new_data_key(_context(workspace_id)), made_at=now)
            # A workspace gone quiet keeps no key in memory past its age.
            self._write_keys = {
                other: key for other, key in self._write_keys.items() if now - key.made_at < _WRITE_KEY_MAX_AGE_S
            }
            self._write_keys[workspace] = held
        held.uses += 1
        return held.key

    async def _open_key(self, wrapper: str, wrapped: bytes, workspace_id: uuid.UUID) -> bytes:
        if self._keys is None:
            raise ContentKeyError("unavailable", "no content key is configured")
        cache_key = (wrapper, wrapped, str(workspace_id))
        now = monotonic()
        cached = self._open_keys.get(cache_key)
        if cached is not None and now - cached[1] < _OPEN_KEY_MAX_AGE_S:
            self._open_keys.move_to_end(cache_key)
            return cached[0]
        if self._paused():
            raise ContentKeyError("unavailable", "the content key source is paused after a failure")
        data_key = await self._keys.open_data_key(wrapper, wrapped, _context(workspace_id))
        self._open_keys[cache_key] = (data_key, now)
        self._open_keys.move_to_end(cache_key)
        while len(self._open_keys) > _OPEN_KEY_CACHE_SIZE:
            self._open_keys.popitem(last=False)
        return data_key

    def _paused(self) -> bool:
        return monotonic() < self._paused_until

    def _failed(self, exc: ContentKeyError, action: str) -> None:
        if exc.reason == "rejected":
            return
        # A key source that is down or refusing would fail every span the same way.
        self._paused_until = monotonic() + _PAUSE_AFTER_FAILURE_S
        logger.warning(
            "Trace content could not be %s (%s: %s); the key source is not asked again for %.0fs",
            action,
            exc.reason,
            exc,
            _PAUSE_AFTER_FAILURE_S,
        )


def _context(workspace_id: uuid.UUID) -> dict[str, str]:
    return {"otari:purpose": _PURPOSE, "otari:workspace": str(workspace_id)}


def _associated_data(workspace_id: uuid.UUID, span_id: str) -> bytes:
    return json.dumps(
        {"v": 1, "workspace_id": str(workspace_id), "span_id": span_id}, sort_keys=True, separators=(",", ":")
    ).encode()


def _b64(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode()


def _unb64(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def _allowed(content: object) -> dict[str, str]:
    if not isinstance(content, dict):
        return {}
    return {
        key: value
        for key, value in content.items()
        if key in SPAN_CONTENT_KEYS and isinstance(value, str) and len(value) <= SPAN_CONTENT_MAX_LENGTH
    }


def _open_legacy(sealed: str, *, workspace_id: uuid.UUID, span_id: str) -> dict[str, str]:
    """A bare Fernet token, as content was sealed before data keys, naming its row in the payload."""
    try:
        payload = json.loads(decrypt_secret(sealed))
    except (SecretBoxUnavailableError, SecretDecryptionError, ValueError):
        UNOPENED.labels(reason="rejected").inc()
        return {}
    if not isinstance(payload, dict) or (payload.get("workspace_id"), payload.get("span_id")) != (
        str(workspace_id),
        span_id,
    ):
        UNOPENED.labels(reason="rejected").inc()
        return {}
    return _allowed(payload.get("content"))
