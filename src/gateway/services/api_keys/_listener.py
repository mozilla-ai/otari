"""The listener an API-key deletion calls inside its transaction."""

from typing import Protocol


class ApiKeyDeletionListener(Protocol):
    """React to a key's deletion without committing the caller's transaction."""

    async def key_deleted(self, key_id: str) -> None:
        """Remove state that must not outlive this key."""
