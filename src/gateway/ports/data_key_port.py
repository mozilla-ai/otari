"""Where the keys that encrypt trace content come from.

Content a workspace opts in to capture is encrypted by the application with a
data key (DEK) per session, and stored only with that key wrapped by the
deployment's key-encryption key (KEK). Where the KEK lives is the second
implementation this port exists for (``ARCHITECTURE.md``, rule 7): an
environment variable on a self-hosted deployment, a key management service on a
hosted one. Both adapters ship in the core, because a hybrid data plane runs the
plain build with no overlay to bind one.

Every data key is bound to its context, the workspace and the session it
encrypts. An adapter refuses to unwrap a key under any other context, so a
wrapped key copied onto another tenant's or another session's row yields nothing.

There is no rewrap: a deployment keeps one KEK, and deleting a wrapped key is how
the content it sealed stops being readable through Otari. A database backup keeps
both the wrapped key and the ciphertext until the backup itself expires.

Stability: this interface is not frozen while Otari is pre-1.0.
"""

import uuid
from dataclasses import dataclass, field
from typing import Protocol


class DataKeyUnavailableError(Exception):
    """No key-encryption key is configured, or the one configured cannot be used.

    The message names the setting to fix and never carries key material.
    """


class DataKeyContextMismatchError(Exception):
    """A wrapped key was presented under a context it was not generated for, or does not unwrap at all."""


@dataclass(frozen=True)
class DataKeyContext:
    """What a data key is bound to: one session's content, in one workspace."""

    workspace_id: uuid.UUID
    session: str

    def as_mapping(self) -> dict[str, str]:
        """The context as string pairs, the shape a KMS encryption context takes."""
        return {"workspace_id": str(self.workspace_id), "session": self.session}


@dataclass(frozen=True)
class DataKey:
    """A fresh data key: the plaintext to encrypt with, and the wrapped form to store beside the ciphertext.

    ``key_ref`` names the KEK that wrapped it, so a reader knows which backend to ask.
    The plaintext is held in memory only, never stored or logged.
    """

    plaintext: bytes = field(repr=False)
    wrapped: bytes = field(repr=False)
    key_ref: str


class DataKeyPort(Protocol):
    """Issue data keys bound to a context, and unwrap them under that context only."""

    async def available(self) -> bool:
        """Whether a key can be generated now: the KEK is configured and the backend answers.

        Never raises; an adapter logs why it is not available, without key material.
        """
        ...

    async def generate(self, context: DataKeyContext) -> DataKey:
        """Return a new 256-bit data key, wrapped by the KEK and bound to ``context``.

        Raises:
            DataKeyUnavailableError: no usable KEK is configured.
        """
        ...

    async def unwrap(self, wrapped: bytes, context: DataKeyContext) -> bytes:
        """Return the plaintext of a wrapped key, only under the context it was generated for.

        Raises:
            DataKeyUnavailableError: no usable KEK is configured.
            DataKeyContextMismatchError: the key was bound to another context, or does not unwrap.
        """
        ...
