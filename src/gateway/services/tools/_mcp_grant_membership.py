"""Ends a member's MCP server grants, connections and requests when they leave a workspace."""

import uuid
from collections.abc import Collection, Sequence
from datetime import UTC, datetime
from typing import Protocol

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.tenancy import OrganizationMember, WorkspaceMember
from gateway.repositories.tools import (
    McpOAuthFlowRepository,
    McpServerRequestRepository,
    McpUserConnectionRepository,
    WorkspaceMcpServerGrantRepository,
    WorkspaceMcpServerRepository,
)
from gateway.services.tools._mcp_connections import (
    MAX_BACKGROUND_REVOCATIONS,
    collect_revocations,
    revoke_in_background,
)


class McpOrganizationReach(Protocol):
    """Whether a person still reaches an organization, which decides their sign-ins to its own servers.

    Implemented where workspaces and their members are known, inside the caller's Unit of Work.
    """

    async def organizations_of(self, workspace_ids: Collection[uuid.UUID]) -> set[uuid.UUID]:
        """The organizations these workspaces belong to."""
        ...

    async def people_of(self, member_ids: Collection[uuid.UUID]) -> list[uuid.UUID]:
        """The dashboard accounts these workspace memberships join."""
        ...

    async def still_reaching(
        self, organization_id: uuid.UUID, user_ids: Collection[uuid.UUID], *, losing: Collection[uuid.UUID]
    ) -> set[uuid.UUID]:
        """Which of these dashboard accounts reach the organization once they lose these workspaces."""
        ...


class McpGrantMembershipListener:
    """Removes a member's user grants on a workspace's MCP servers as they leave it, inside the caller's Unit of Work.

    A grant outliving its membership would come back unseen if the person were invited again, so regaining access
    revives none, as with their keys and session shares. Their own connections to those servers, and any
    sign-in they had started, go the same way: the tokens were theirs to use as a member. A server they asked
    for and are still waiting on is withdrawn, so no manager answers someone who has left.

    An organization's server is reached from every workspace of the organization. Leaving one workspace ends
    the grants that applied there and the sign-ins started there; a connection to the server ends only once
    the person reaches no workspace of the organization and is not one of its owners or admins.
    """

    def __init__(self, uow: UnitOfWork, reach: McpOrganizationReach) -> None:
        self._reach = reach
        self._grants = WorkspaceMcpServerGrantRepository(uow)
        self._servers = WorkspaceMcpServerRepository(uow)
        self._connections = McpUserConnectionRepository(uow)
        self._flows = McpOAuthFlowRepository(uow)
        self._requests = McpServerRequestRepository(uow)

    async def _remove(self, user_id: uuid.UUID, workspace_ids: Sequence[uuid.UUID]) -> None:
        server_ids = await self._servers.ids_in_workspaces(workspace_ids)
        server_ids += await self._organization_servers_lost(user_id, workspace_ids)
        await self._revoke_after_commit(server_ids=server_ids, user_ids=[str(user_id)])
        await self._grants.remove_for_user_in_workspaces(str(user_id), workspace_ids)
        await self._connections.remove_for_user_on_servers(str(user_id), server_ids)
        await self._flows.remove_for_user(str(user_id), server_ids)
        await self._flows.remove_for_user_in_workspaces(str(user_id), workspace_ids)
        await self._requests.withdraw_for_requester(user_id, workspace_ids, at=datetime.now(UTC))

    async def _organization_servers_lost(
        self, user_id: uuid.UUID, workspace_ids: Sequence[uuid.UUID]
    ) -> list[uuid.UUID]:
        """The organization servers a person stops reaching as they lose these workspaces."""
        lost: list[uuid.UUID] = []
        for organization_id in await self._reach.organizations_of(workspace_ids):
            shared = await self._servers.ids_of_organization(organization_id)
            if shared and user_id not in await self._reach.still_reaching(
                organization_id, [user_id], losing=workspace_ids
            ):
                lost += shared
        return lost

    async def _revoke_after_commit(
        self,
        *,
        server_ids: Sequence[uuid.UUID],
        user_ids: Collection[str] | None = None,
        shared: bool = False,
        budget: int = MAX_BACKGROUND_REVOCATIONS,
    ) -> int:
        """Ask the authorization servers to end the tokens being deleted, once the caller's change commits.

        The change is the caller's, which may still roll back, so nothing is sent before it commits.
        ``shared`` adds the servers' shared accounts, for a change that deletes the servers. At most
        ``budget`` connections are read, whether or not their tokens decrypt; the rest are counted. Returns
        what is left of the budget.
        """
        if not server_ids:
            return budget
        revocations, unread, read = await collect_revocations(
            self._connections,
            self._servers,
            server_ids=server_ids,
            user_ids=user_ids,
            shared=shared,
            limit=budget,
        )
        if revocations or unread:
            self._connections.after_commit(lambda: revoke_in_background(revocations, unread=unread))
        return max(budget - read, 0)

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
        """The workspace's servers go with it, and every token they hold is revoked once it commits.

        A member it was the last way into the organization for loses their sign-ins to the organization's own
        servers too. Who that is, and their rows, are read and removed as one set, and the tokens revoked
        after the commit share one limit.
        """
        budget = await self._revoke_after_commit(
            server_ids=await self._servers.ids_in_workspaces([workspace_id]), shared=True
        )
        people = await self._reach.people_of(member_ids)
        for organization_id in await self._reach.organizations_of([workspace_id]):
            shared = await self._servers.ids_of_organization(organization_id)
            if not shared or not people:
                continue
            staying = await self._reach.still_reaching(organization_id, people, losing=[workspace_id])
            departing = [str(user_id) for user_id in people if user_id not in staying]
            budget = await self._revoke_after_commit(server_ids=shared, user_ids=departing, budget=budget)
            await self._connections.remove_for_users_on_servers(departing, shared)
            await self._flows.remove_for_users(departing, shared)

    async def member_lost_workspaces(self, user_id: uuid.UUID, workspace_ids: Sequence[uuid.UUID]) -> None:
        """Remove the person's grants on, and connections to, these workspaces' servers."""
        await self._remove(user_id, workspace_ids)


__all__ = ["McpGrantMembershipListener"]
