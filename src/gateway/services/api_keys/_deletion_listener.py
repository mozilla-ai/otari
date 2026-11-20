"""The interface a domain implements to react to an API key's deletion.

The api-keys domain deletes the key and calls this; it never imports the domain that implements it.
"""

from typing import Protocol

from gateway.models.api_keys import APIKey


class ApiKeyDeletionListener(Protocol):
    """Reacts to an API key's deletion inside the transaction that makes it.

    NOTE: an implementation must not commit or roll back.
    The caller owns the transaction, so a refused deletion takes the listener's writes back with it.
    """

    async def key_deleted(self, key: APIKey) -> None:
        """A key is about to be deleted."""


class ApiKeyDeletionListeners:
    """Tells each listener in turn, so more than one domain can react to the same deletion."""

    def __init__(self, *listeners: ApiKeyDeletionListener) -> None:
        self._listeners = listeners

    async def key_deleted(self, key: APIKey) -> None:
        for listener in self._listeners:
            await listener.key_deleted(key)


__all__ = ["ApiKeyDeletionListener", "ApiKeyDeletionListeners"]
