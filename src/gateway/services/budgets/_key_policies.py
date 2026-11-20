"""Keeps spend ceilings in step with the API keys they are scoped to.

The api-keys domain defines :class:`ApiKeyDeletionListener` and calls it when a key is deleted;
budgets implements it here, so the key's ``api_token`` ceiling goes with the key rather than being
stranded. It holds repositories and nothing that commits, so a refused deletion takes its writes back.
"""

from gateway.models.api_keys import APIKey
from gateway.repositories.budgets import BudgetRepositories


class BudgetKeyDeletionListener:
    """Deletes a key's ceilings when the key is deleted, inside the caller's Unit of Work block."""

    def __init__(self, repositories: BudgetRepositories) -> None:
        self._repositories = repositories

    async def key_deleted(self, key: APIKey) -> None:
        """Delete the ceilings keyed on this API key."""
        await self._repositories.ceilings.delete_for_api_key(key.id)
