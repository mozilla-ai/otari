"""Keeps members' own API keys in step with their workspace and organization memberships."""

import uuid
from collections.abc import Sequence

from gateway.models.tenancy import OrganizationMember, WorkspaceMember
from gateway.repositories.api_keys import ApiKeyRepositories

_VIEWER = "viewer"


class ApiKeyMembershipListener:
    """Switches off a member's keys when their membership no longer allows them, inside the caller's Unit of Work.

    A removal switches off every key billed to them in the workspaces they leave, as an operator would, so they may
    not switch it back on. A demotion to viewer switches off only the keys they minted for themselves, which they may
    switch back on if promoted again; a key an operator issued them, or one of unrecorded origin, is left alone.
    Switched off rather than deleted, so an operator can still read the key's usage and turn it back on.
    """

    def __init__(self, repositories: ApiKeyRepositories) -> None:
        self._keys = repositories.keys

    async def member_joined(self, member: WorkspaceMember) -> None:
        """Nothing: a key switched off on removal stays off until an operator turns it back on."""

    async def member_removed(self, member: WorkspaceMember) -> None:
        """Switch off the keys billed to this member in the workspace they leave."""
        await self._keys.deactivate_owned_in_workspaces(
            str(member.user_id), [member.workspace_id], self_service_only=False, by_operator=True
        )

    async def member_role_changed(self, member: WorkspaceMember) -> None:
        """Switch off the keys a member minted here, once they are a viewer of this workspace."""
        if member.role == _VIEWER:
            await self._keys.deactivate_owned_in_workspaces(
                str(member.user_id), [member.workspace_id], self_service_only=True, by_operator=False
            )

    async def organization_member_changed(
        self, membership: OrganizationMember, workspace_ids: Sequence[uuid.UUID]
    ) -> None:
        """Switch off a suspended member's keys in the organization, or a viewer's self-minted ones."""
        if membership.status == "suspended":
            await self._keys.deactivate_owned_in_workspaces(
                str(membership.user_id), workspace_ids, self_service_only=False, by_operator=True
            )
        elif membership.role == _VIEWER:
            await self._keys.deactivate_owned_in_workspaces(
                str(membership.user_id), workspace_ids, self_service_only=True, by_operator=False
            )

    async def workspace_deleted(self, workspace_id: uuid.UUID, member_ids: Sequence[uuid.UUID]) -> None:
        """Nothing: a workspace that still holds keys cannot be deleted."""


__all__ = ["ApiKeyMembershipListener"]
