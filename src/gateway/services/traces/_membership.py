"""Ends a member's shares of their sessions when they leave a workspace."""

import uuid
from collections.abc import Sequence
from datetime import UTC, datetime

from gateway.models.tenancy import OrganizationMember, WorkspaceMember
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

    async def member_role_changed(self, member: WorkspaceMember) -> None:
        """Nothing: a member's role in the workspace is not who may read their sessions."""

    async def organization_member_changed(
        self, membership: OrganizationMember, workspace_ids: Sequence[uuid.UUID]
    ) -> None:
        """Nothing here: the workspaces a person lost are what ends their shares."""

    async def workspace_deleted(self, workspace_id: uuid.UUID, member_ids: Sequence[uuid.UUID]) -> None:
        """Nothing: the workspace's sessions, and their shares, go with it."""
