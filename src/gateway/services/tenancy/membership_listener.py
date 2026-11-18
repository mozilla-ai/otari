"""The interface a domain implements to react to workspace membership changes.

The domain that changes membership calls it and never imports the domain that implements it.
"""

import uuid
from collections.abc import Collection, Sequence
from typing import Protocol

from gateway.models.tenancy import MANAGEMENT_ROLES, OrganizationMember, WorkspaceMember
from gateway.repositories.tenancy import OrganizationMemberRepository, WorkspaceMemberRepository, WorkspaceRepository


class MembershipListener(Protocol):
    """Reacts to a membership change inside the transaction that made it.

    NOTE: an implementation must not commit or roll back.
    The caller owns the transaction, so a refused change takes the listener's writes back with it.
    """

    async def member_joined(self, member: WorkspaceMember) -> None:
        """A member joined a workspace, or an inactive membership became active."""

    async def member_removed(self, member: WorkspaceMember) -> None:
        """A membership is about to be deleted."""

    async def member_role_changed(self, member: WorkspaceMember) -> None:
        """A workspace membership's role changed; ``member`` carries the new one."""

    async def organization_member_changed(
        self, membership: OrganizationMember, workspace_ids: Sequence[uuid.UUID]
    ) -> None:
        """An organization membership's role or status changed; ``workspace_ids`` are the organization's."""

    async def workspace_deleted(self, workspace_id: uuid.UUID, member_ids: Sequence[uuid.UUID]) -> None:
        """A workspace and these memberships are about to be deleted."""

    async def member_lost_workspaces(self, user_id: uuid.UUID, workspace_ids: Sequence[uuid.UUID]) -> None:
        """A person stopped seeing these workspaces: their organization membership ended, or their role changed."""


class MembershipListeners:
    """Tells each listener in turn, so more than one domain can react to the same change."""

    def __init__(self, *listeners: MembershipListener) -> None:
        self._listeners = listeners

    async def member_joined(self, member: WorkspaceMember) -> None:
        for listener in self._listeners:
            await listener.member_joined(member)

    async def member_removed(self, member: WorkspaceMember) -> None:
        for listener in self._listeners:
            await listener.member_removed(member)

    async def member_role_changed(self, member: WorkspaceMember) -> None:
        for listener in self._listeners:
            await listener.member_role_changed(member)

    async def organization_member_changed(
        self, membership: OrganizationMember, workspace_ids: Sequence[uuid.UUID]
    ) -> None:
        for listener in self._listeners:
            await listener.organization_member_changed(membership, workspace_ids)

    async def workspace_deleted(self, workspace_id: uuid.UUID, member_ids: Sequence[uuid.UUID]) -> None:
        for listener in self._listeners:
            await listener.workspace_deleted(workspace_id, member_ids)

    async def member_lost_workspaces(self, user_id: uuid.UUID, workspace_ids: Sequence[uuid.UUID]) -> None:
        for listener in self._listeners:
            await listener.member_lost_workspaces(user_id, workspace_ids)


__all__ = ["MembershipListener", "MembershipListeners"]


class OrganizationReach:
    """Whether a person reaches an organization: through one of its workspaces, or as one of its owners or admins.

    For a listener that has to know, inside the caller's Unit of Work, what a membership change leaves a person.
    """

    def __init__(
        self,
        workspaces: WorkspaceRepository,
        members: WorkspaceMemberRepository,
        organization_members: OrganizationMemberRepository,
    ) -> None:
        self._workspaces = workspaces
        self._members = members
        self._organization_members = organization_members

    async def organization_of(self, workspace_id: uuid.UUID) -> uuid.UUID | None:
        """The organization a workspace belongs to, or ``None`` for one that does not exist."""
        return await self._workspaces.get_organization_id(workspace_id)

    async def still_reaches(
        self, organization_id: uuid.UUID, user_id: uuid.UUID, *, losing: Collection[uuid.UUID]
    ) -> bool:
        """Whether a dashboard account reaches the organization once it loses these workspaces."""
        kept = [
            workspace_id
            for workspace_id in await self._workspaces.get_ids_by_organization(organization_id)
            if workspace_id not in losing
        ]
        if await self._members.get_active_pairs(kept, [user_id]):
            return True
        return user_id in await self._organization_members.active_with_roles(
            organization_id, {user_id}, MANAGEMENT_ROLES
        )
