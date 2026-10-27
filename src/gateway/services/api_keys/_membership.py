"""Keeps members' own API keys in step with their workspace memberships."""

import uuid
from collections.abc import Sequence

from gateway.models.tenancy import WorkspaceMember
from gateway.repositories.api_keys import ApiKeyRepositories


class ApiKeyMembershipListener:
    """Switches off a member's keys in a workspace they are removed from, inside the caller's Unit of Work block.

    Switched off rather than deleted, so an operator can still read the key's usage and turn it back on.
    """

    def __init__(self, repositories: ApiKeyRepositories) -> None:
        self._keys = repositories.keys

    async def member_joined(self, member: WorkspaceMember) -> None:
        """Nothing: a key switched off on removal stays off until an operator turns it back on."""

    async def member_removed(self, member: WorkspaceMember) -> None:
        """Switch off the keys billed to this member in the workspace they leave."""
        await self._keys.deactivate_owned_in_workspace(str(member.user_id), member.workspace_id)

    async def workspace_deleted(self, workspace_id: uuid.UUID, member_ids: Sequence[uuid.UUID]) -> None:
        """Nothing: a workspace that still holds keys cannot be deleted."""


__all__ = ["ApiKeyMembershipListener"]
