"""A file a provider holds, as Otari names it."""

from dataclasses import dataclass
from datetime import datetime

__all__ = ["ProviderCopyReceipt", "ProviderFile"]


@dataclass(frozen=True)
class ProviderFile:
    """One file a provider's sandbox produced, as its response announced it."""

    file_id: str
    filename: str | None = None
    container_id: str | None = None


@dataclass(frozen=True)
class ProviderCopyReceipt:
    """What a provider answered when it accepted a copy of a file.

    ``expires_at`` is None where the provider did not say when it expires the copy.
    """

    file_id: str
    expires_at: datetime | None = None
