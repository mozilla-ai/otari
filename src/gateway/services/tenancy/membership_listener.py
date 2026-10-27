"""The interface a domain implements to react to workspace membership changes.

The domain that changes membership calls it and never imports the domain that implements it.
"""

import uuid
from collections.abc import Sequence
from typing import Protocol

from gateway.models.tenancy import OrganizationMember, WorkspaceMember


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


__all__ = ["MembershipListener", "MembershipListeners"]
