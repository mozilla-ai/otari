"""How Otari reads and writes the files a provider holds.

The seam between the files domain and a provider's own files API.
A provider-native feature names a file only by an ID that provider issued, so a
file Otari holds is copied there for it, and a file the provider's code produced
is read back from there.
An attachment too large to send inline is put in the provider's storage too, for
a provider that reads such files by URI.

A session is opened for one provider account, with the credential that reaches
it, and owns the connection its calls run on until it is closed.
Upload failures cross as ``ProviderUploadFailedError``, and a file that cannot be
read as ``ProviderFileUnavailableError``, so an adapter translates its own
client's exceptions before they reach a caller.

Stability: this interface is not frozen while Otari is pre-1.0.
Overlay authors should pin a released tag and expect the shape to move.
"""

from collections.abc import AsyncGenerator
from typing import Protocol

from any_llm import LLMProvider

from gateway.types.provider_account import ResolvedCredential
from gateway.types.provider_file import ProviderCopyReceipt, ProviderFile

__all__ = ["ProviderFilePort", "ProviderFileSession"]


class ProviderFileSession(Protocol):
    """The files of one provider account, reached with one credential."""

    async def upload(self, data: bytes, *, filename: str, mime_type: str, expires_in: int) -> ProviderCopyReceipt:
        """Store ``data`` at the provider for ``expires_in`` seconds, and return what it recorded.

        Raises:
            ProviderUploadFailedError: the provider refused the copy, has no
                files API, or could not be reached.
        """
        ...

    async def discard(self, file_id: str) -> bool:
        """Remove a file this session put at the provider, and say whether it is gone."""
        ...

    async def holds(self, file_id: str) -> bool:
        """Whether the provider still holds ``file_id``.

        False only when the provider says it does not, so a lookup that fails
        for any other reason leaves a recorded copy in use.
        """
        ...

    async def filename_of(self, file_id: str) -> str | None:
        """The name the provider holds ``file_id`` under, or None where it does not say."""
        ...

    def read(self, file: ProviderFile, *, budget_bytes: int) -> AsyncGenerator[bytes, None]:
        """Stream one file's bytes.

        Raises:
            FileOverBudgetError: the file runs past ``budget_bytes``.
            ProviderFileUnavailableError: the provider cannot serve the file.
        """
        ...

    async def aclose(self) -> None:
        """Release the connection this session runs on."""
        ...


class ProviderFilePort(Protocol):
    """What a build must answer to reach the files a provider holds."""

    def serves(self, provider: LLMProvider) -> bool:
        """Whether this build can reach ``provider``'s files at all."""
        ...

    def open_session(
        self, *, provider: LLMProvider, instance: str, credential: ResolvedCredential
    ) -> ProviderFileSession:
        """A session over the files of the account ``credential`` reaches.

        Raises:
            LookupError: Otari cannot reach this provider's files.
        """
        ...

    async def upload_attachment(
        self, *, provider: LLMProvider, credential: ResolvedCredential, data: bytes, filename: str, mime_type: str
    ) -> str:
        """Put an attachment where a request to ``provider`` can reference it, and return the URI it is referenced by.

        For a provider that caps the bytes one request carries inline and reads a
        larger file from its own storage (Gemini). Returns once the provider can
        read the file. The provider expires it on its own schedule.

        Raises:
            ProviderUploadFailedError: the provider has no such storage here,
                refused the file, or never made it readable.
        """
        ...
