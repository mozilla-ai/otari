"""How memory text and embeddings are sealed at rest. The one place that knows.

A sealed value reads::

    mem1.<nonce + AES-256-GCM ciphertext>

url-safe base64, under a data key of the row's workspace that the row names by
id (``memory_keys``). The ciphertext is authenticated with the workspace, the
scope, the owning member, the row and the field as associated data, so a value
copied onto another row, another field, another member's entry or another
workspace opens as nothing.

An embedding is sealed as its float32 vector, little-endian, so a search opens a
workspace's vectors without ever storing one in the clear.
"""

import base64
import binascii
import json
import os
import sys
import uuid
from array import array
from collections.abc import Sequence
from typing import Literal

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from gateway.models.memory import MemoryScope

FORMAT = "mem1"
PURPOSE = "memory-content"
NONCE_BYTES = 12

SealedField = Literal["content", "embedding"]
SealedRecord = Literal["entry", "proposal"]


class SealedValueError(Exception):
    """A sealed value is malformed, or does not open under this key for this row."""


def key_context(workspace_id: uuid.UUID) -> dict[str, str]:
    """What a workspace data key is bound to: KMS opens it for memory of this workspace and nothing else."""
    return {"otari:purpose": PURPOSE, "otari:workspace": str(workspace_id)}


def associated_data(
    *,
    workspace_id: uuid.UUID,
    scope: MemoryScope,
    owner_user_id: uuid.UUID | None,
    record: SealedRecord,
    record_id: uuid.UUID,
    field: SealedField,
) -> bytes:
    """What a sealed value is bound to, besides its key."""
    return json.dumps(
        {
            "v": 1,
            "workspace_id": str(workspace_id),
            "scope": scope,
            "owner_user_id": str(owner_user_id) if owner_user_id is not None else None,
            "record": record,
            "record_id": str(record_id),
            "field": field,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def seal(key: bytes, plaintext: bytes, aad: bytes) -> str:
    """``plaintext`` sealed under ``key`` for ``aad``."""
    nonce = os.urandom(NONCE_BYTES)
    ciphertext = AESGCM(key).encrypt(nonce, plaintext, aad)
    return f"{FORMAT}.{_b64(nonce + ciphertext)}"


def open_sealed(key: bytes, sealed: str, aad: bytes) -> bytes:
    """The plaintext of a value :func:`seal` made under ``key`` for ``aad``.

    Raises:
        SealedValueError: the value is malformed, or it was sealed under another key or for another row.
    """
    prefix, _, body = sealed.partition(".")
    if prefix != FORMAT or not body:
        raise SealedValueError("not a sealed memory value")
    try:
        raw = _unb64(body)
    except (binascii.Error, ValueError):
        raise SealedValueError("not a sealed memory value") from None
    if len(raw) <= NONCE_BYTES:
        raise SealedValueError("not a sealed memory value")
    try:
        return AESGCM(key).decrypt(raw[:NONCE_BYTES], raw[NONCE_BYTES:], aad)
    except InvalidTag:
        raise SealedValueError("the value does not open under this key for this row") from None


def pack_embedding(vector: Sequence[float]) -> bytes:
    """A vector as float32, little-endian."""
    packed = array("f", vector)
    if sys.byteorder != "little":
        packed.byteswap()
    return packed.tobytes()


def unpack_embedding(data: bytes) -> tuple[float, ...]:
    """The vector :func:`pack_embedding` made.

    Raises:
        SealedValueError: the bytes are not a whole number of float32 values.
    """
    if len(data) % 4:
        raise SealedValueError("not a packed embedding")
    packed = array("f")
    packed.frombytes(data)
    if sys.byteorder != "little":
        packed.byteswap()
    return tuple(packed)


def _b64(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode()


def _unb64(value: str) -> bytes:
    # Validated, because the lenient decoder drops characters it does not know and decodes what is left.
    return base64.b64decode(value + "=" * (-len(value) % 4), altchars=b"-_", validate=True)


def encode_wrapped(wrapped: bytes) -> str:
    """A wrapped data key as it is stored in ``memory_keys``."""
    return _b64(wrapped)


def decode_wrapped(stored: str) -> bytes:
    """The wrapped data key ``memory_keys`` stores.

    Raises:
        SealedValueError: the stored value is not one :func:`encode_wrapped` made.
    """
    try:
        return _unb64(stored)
    except (binascii.Error, ValueError):
        raise SealedValueError("not a stored data key") from None
