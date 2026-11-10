"""The memory domain's one service."""

import uuid

from gateway.core.unit_of_work import UnitOfWork
from gateway.repositories.memory import MemoryRepositories
from gateway.services.memory._keys import WorkspaceKeys
from gateway.services.memory._store import EntryDraft, EntryStore, OpenedEntry, ProposalDraft


class MemoryService:
    """Store what agents propose and what the consolidator keeps, sealed, and open what a member may recall."""

    def __init__(self, uow: UnitOfWork, repos: MemoryRepositories, keys: WorkspaceKeys) -> None:
        self._uow = uow
        self._repos = repos
        self._store = EntryStore(repos, keys)

    async def propose(self, draft: ProposalDraft) -> uuid.UUID:
        """Store one proposal for the consolidator; its id.

        Raises:
            MemorySealingUnavailableError: no data key could be had, so nothing was stored.
        """
        async with self._uow:
            return await self._store.write_proposal(draft)

    async def keep(self, draft: EntryDraft) -> uuid.UUID:
        """Store one entry version the consolidator decided on; its id.

        Raises:
            MemorySealingUnavailableError: no data key could be had, so nothing was stored.
        """
        async with self._uow:
            return await self._store.write_entry(draft)

    async def recallable(self, workspace_id: uuid.UUID, member_id: uuid.UUID) -> list[OpenedEntry]:
        """The entries ``member_id`` may recall in the workspace, as its switches allow; none while memory is off."""
        async with self._uow:
            settings = await self._repos.settings.of_workspace(workspace_id)
            if settings is None or not settings.enabled:
                return []
            return await self._store.recallable(
                workspace_id,
                member_id=member_id,
                include_workspace=settings.workspace_scope_enabled,
                include_external=not settings.hide_external,
            )
