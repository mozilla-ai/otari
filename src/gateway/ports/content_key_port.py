"""Where the keys that seal stored content come from.

Content is sealed under a data key: random bytes that exist in plaintext only in
this process's memory, stored beside the ciphertext in wrapped form. This port
makes data keys and unwraps them. It never sees content, and it does not know
how content is sealed with a key; that belongs to the code that seals it.

One adapter wraps data keys with ``OTARI_SECRET_KEY``, another with a key held
in AWS KMS. Every wrapped key names the wrapper that made it, so a deployment
that changes wrapper keeps opening what the old one wrapped for as long as the
old one is configured.

A data key is made for a context, a small mapping of names to values, and opens
only for the same context. An adapter binds it as tightly as its key source
allows: KMS checks it as the encryption context, and a local wrapper seals it
beside the key.

Failures cross as :class:`ContentKeyError`, whose ``reason`` says which kind,
and whose message never carries key material, a wrapped key or a context value.
"""

from dataclasses import dataclass
from typing import Literal, Protocol

ContentKeyFailure = Literal["unavailable", "denied", "rejected"]


class ContentKeyError(Exception):
    """A data key could not be made or opened.

    ``reason`` is ``unavailable`` when the key source could not be reached or
    refused for load, ``denied`` when it refused this process, and ``rejected``
    when the wrapped key is not one it opens for this context.
    """

    def __init__(self, reason: ContentKeyFailure, message: str) -> None:
        super().__init__(message)
        self.reason: ContentKeyFailure = reason


@dataclass(frozen=True, repr=False)
class DataKey:
    """A new data key, in plaintext and wrapped. Its ``repr`` names only the wrapper."""

    wrapper: str
    """Which wrapper made it, stored with the ciphertext so the same one opens it."""
    plaintext: bytes
    """32 bytes for AES-256. Never stored, logged or returned."""
    wrapped: bytes
    """What is stored beside the ciphertext."""

    def __repr__(self) -> str:
        return f"DataKey(wrapper={self.wrapper!r})"


class DataKeySource(Protocol):
    """One place data keys come from: ``OTARI_SECRET_KEY``, or a KMS key.

    A :class:`ContentKeyPort` adapter puts the sources a deployment configured
    on one keyring, so a key opens with whichever source made it.
    """

    wrapper: str
    """The name stored with each key it wraps, so the same source opens it."""

    async def new_data_key(self, context: dict[str, str]) -> DataKey:
        """Make a data key for ``context``. Raises :class:`ContentKeyError`."""
        ...

    async def open_data_key(self, wrapped: bytes, context: dict[str, str]) -> bytes:
        """Return the plaintext of a key this source made for ``context``. Raises :class:`ContentKeyError`."""
        ...

    def close(self) -> None:
        """Release what the source holds open, as the process stops."""
        ...


class ContentKeyPort(Protocol):
    """Makes data keys with one wrapper and opens those of every wrapper configured."""

    @property
    def wrapper(self) -> str | None:
        """The wrapper new data keys are made with, or ``None`` where none is configured."""
        ...

    async def new_data_key(self, context: dict[str, str]) -> DataKey:
        """Make a data key for ``context``.

        Raises:
            ContentKeyError: no wrapper is configured, or it could not make one.
        """
        ...

    async def open_data_key(self, wrapper: str, wrapped: bytes, context: dict[str, str]) -> bytes:
        """Return the plaintext of a key ``wrapper`` made for ``context``.

        Raises:
            ContentKeyError: that wrapper is not configured, could not be reached,
                or does not open this key for this context.
        """
        ...

    def close(self) -> None:
        """Release what every source holds open, as the process stops."""
        ...
