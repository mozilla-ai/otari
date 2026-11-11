"""Keeps members' own API keys in step with their workspace and organization memberships."""

import uuid
from collections.abc import Sequence

from gateway.models.api_keys import DisabledReason
from gateway.models.tenancy import OrganizationMember, WorkspaceMember
from gateway.repositories.api_keys import ApiKeyRepositories

_VIEWER = "viewer"


class ApiKeyMembershipListener:
    """Disables a member's API keys when their membership no longer allows them, inside the caller's Unit of Work.

    A removal disables every key billed to them in the workspaces they leave and records the operator reason, so
    they cannot enable it again. A demotion to viewer disables only the keys they created themselves, which they
    can enable again if promoted; a key an operator issued them, or one of unrecorded origin, is left alone.
    Disabled rather than deleted, so an operator can still read the key's usage and enable it again.
    """

    def __init__(self, repositories: ApiKeyRepositories) -> None:
        self._keys = repositories.keys

    async def member_joined(self, member: WorkspaceMember) -> None:
        """Nothing: a key disabled on removal stays disabled until an operator enables it."""

    async def member_removed(self, member: WorkspaceMember) -> None:
        """Disable the keys billed to this member in the workspace they leave."""
        await self._keys.disable_owned_in_workspaces_with_reason(
            str(member.user_id), [member.workspace_id], self_service_only=False, reason=DisabledReason.OPERATOR
        )

    async def member_role_changed(self, member: WorkspaceMember) -> None:
        """Disable the keys a member created here, once they are a viewer of this workspace."""
        if member.role == _VIEWER:
            await self._keys.disable_owned_in_workspaces(
                str(member.user_id), [member.workspace_id], self_service_only=True
            )

    async def organization_member_changed(
        self, membership: OrganizationMember, workspace_ids: Sequence[uuid.UUID]
    ) -> None:
        """Disable a suspended member's keys in the organization, or a viewer's own."""
        if membership.status == "suspended":
            await self._keys.disable_owned_in_workspaces_with_reason(
                str(membership.user_id), workspace_ids, self_service_only=False, reason=DisabledReason.OPERATOR
            )
        elif membership.role == _VIEWER:
            await self._keys.disable_owned_in_workspaces(str(membership.user_id), workspace_ids, self_service_only=True)

    async def workspace_deleted(self, workspace_id: uuid.UUID, member_ids: Sequence[uuid.UUID]) -> None:
        """Nothing: a workspace that still holds keys cannot be deleted."""

    async def member_lost_workspaces(self, user_id: uuid.UUID, workspace_ids: Sequence[uuid.UUID]) -> None:
        """Nothing here: ``organization_member_changed`` already disabled what the change takes away."""


__all__ = ["ApiKeyMembershipListener"]
