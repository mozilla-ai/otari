"""Ends a member's shares of their sessions when they leave a workspace."""

import uuid
from collections.abc import Sequence
from datetime import UTC, datetime

from gateway.models.tenancy import WorkspaceMember
from gateway.repositories.traces import TracesRepositories


class TraceMembershipListener:
    """Ends the shares a member made in a workspace as they leave it, inside the caller's Unit of Work block.

    Nobody reads a departed member's content, and they can no longer manage what
    they shared, so their shares end with their membership. Reading through a share
    checks the owner's membership as well, which covers leaving the organization.
    """

    def __init__(self, repositories: TracesRepositories) -> None:
        self._repositories = repositories

    async def member_joined(self, member: WorkspaceMember) -> None:
        """Nothing: a share that ended stays ended."""

    async def member_removed(self, member: WorkspaceMember) -> None:
        """End every share the member made of a session in the workspace."""
        await self._repositories.shares.revoke_by(member.workspace_id, member.user_id, datetime.now(UTC))

    async def workspace_deleted(self, workspace_id: uuid.UUID, member_ids: Sequence[uuid.UUID]) -> None:
        """Nothing: the workspace's sessions, and their shares, go with it."""

    async def member_left_organization(self, user_id: uuid.UUID, workspace_ids: Sequence[uuid.UUID]) -> None:
        """End every share the person made in any of the organization's workspaces, so a return revives none."""
        now = datetime.now(UTC)
        for workspace_id in workspace_ids:
            await self._repositories.shares.revoke_by(workspace_id, user_id, now)
