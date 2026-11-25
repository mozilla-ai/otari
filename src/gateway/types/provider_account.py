"""The provider account a dispatch reaches, and the credential it reaches it with."""

import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from any_llm import LLMProvider

__all__ = ["ProviderAccount", "ResolvedCredential"]


@dataclass(frozen=True, slots=True)
class ResolvedCredential:
    """What a provider is called with, which is what selects the account a call reaches."""

    api_key: str = field(repr=False)
    api_base: str | None = None
    client_args: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def request_headers(self) -> dict[str, str]:
        """The headers every call on this credential carries, empty where it carries none.

        A proxy in front of the address reads them, so they are part of what
        selects the account a call reaches. Derived here so every caller reads
        one answer.
        """
        headers = self.client_args.get("default_headers")
        if not isinstance(headers, Mapping):
            return {}
        return {str(name): str(value) for name, value in headers.items()}


@dataclass(frozen=True, slots=True)
class ProviderAccount:
    """One provider account, named without its credential.

    ``identity`` is opaque: two accounts are the same account exactly when their
    identities are equal, and nothing else may be read out of it.
    ``instance`` and ``workspace_id`` locate the credential again later, which
    the identity cannot, because it is not reversible.
    """

    provider: LLMProvider
    instance: str
    workspace_id: uuid.UUID
    identity: str
