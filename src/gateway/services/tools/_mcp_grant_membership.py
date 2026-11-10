"""Ends a member's MCP server grants and connections when they leave a workspace."""

import uuid
from collections.abc import Sequence

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.tenancy import OrganizationMember, WorkspaceMember
from gateway.repositories.tools import (
    McpOAuthFlowRepository,
    McpUserConnectionRepository,
    WorkspaceMcpServerGrantRepository,
    WorkspaceMcpServerRepository,
)


class McpGrantMembershipListener:
    """Removes a member's user grants on a workspace's MCP servers as they leave it, inside the caller's Unit of Work.

    A grant outliving its membership would come back unseen if the person were invited again, so regaining access
    revives none, as with their keys and session shares. Their own connections to those servers, and any
    sign-in they had started, go the same way: the tokens were theirs to use as a member.
    """

    def __init__(self, uow: UnitOfWork) -> None:
        self._grants = WorkspaceMcpServerGrantRepository(uow)
        self._servers = WorkspaceMcpServerRepository(uow)
        self._connections = McpUserConnectionRepository(uow)
        self._flows = McpOAuthFlowRepository(uow)

    async def _remove(self, user_id: uuid.UUID, workspace_ids: Sequence[uuid.UUID]) -> None:
        server_ids = await self._servers.ids_in_workspaces(workspace_ids)
        await self._grants.remove_for_user_on_servers(str(user_id), server_ids)
        await self._connections.remove_for_user_on_servers(str(user_id), server_ids)
        await self._flows.remove_for_user(str(user_id), server_ids)

    async def member_joined(self, member: WorkspaceMember) -> None:
        """Nothing: a grant that ended stays ended."""

    async def member_removed(self, member: WorkspaceMember) -> None:
        """Remove the member's grants on, and connections to, the workspace's servers."""
        await self._remove(member.user_id, [member.workspace_id])

    async def member_role_changed(self, member: WorkspaceMember) -> None:
        """Nothing: a grant does not depend on the member's role."""

    async def organization_member_changed(
        self, membership: OrganizationMember, workspace_ids: Sequence[uuid.UUID]
    ) -> None:
        """Nothing here: the workspaces a person lost are what ends their grants."""

    async def workspace_deleted(self, workspace_id: uuid.UUID, member_ids: Sequence[uuid.UUID]) -> None:
        """Nothing: the workspace's servers, and their grants, go with it."""

    async def member_lost_workspaces(self, user_id: uuid.UUID, workspace_ids: Sequence[uuid.UUID]) -> None:
        """Remove the person's grants on, and connections to, these workspaces' servers."""
        await self._remove(user_id, workspace_ids)


__all__ = ["McpGrantMembershipListener"]
