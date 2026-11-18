"""What leaving the deployment for good does to the organizations an account belongs to."""

import uuid
from dataclasses import dataclass

from gateway.exceptions.organizations_exceptions import AccountDeletionLastOwnerError
from gateway.repositories.organizations import OrganizationRepositories


@dataclass(frozen=True)
class ReleasedOrganizations:
    """The organizations an account left nobody active in, and their workspaces."""

    organization_ids: list[uuid.UUID]
    workspace_ids: list[uuid.UUID]


class AccountRelease:
    """Release an account's organizations ahead of its deletion. Runs in the caller's Unit of Work block."""

    def __init__(self, repositories: OrganizationRepositories) -> None:
        self._bootstrap_marker = repositories.bootstrap_marker
        self._members = repositories.members
        self._organizations = repositories.organizations
        self._invitations = repositories.invitations
        self._workspaces = repositories.workspaces

    async def is_bootstrap_operator(self, identity_id: uuid.UUID) -> bool:
        """Whether the bootstrap operator marker names this account now, read past the session's cache."""
        return await self._bootstrap_marker.user_id() == identity_id

    async def ensure_releasable(self, identity_id: uuid.UUID) -> None:
        """Refuse while this account is the last active owner of an organization other people are active in.

        Nobody else could then manage or delete that organization.
        """
        await self._abandoned(identity_id, lock=False)

    async def release(self, identity_id: uuid.UUID) -> ReleasedOrganizations:
        """Anonymize every organization only this account is active in, and return them.

        An organization other people are active in keeps its name and loses only this membership,
        which goes with the account. Refused as ``ensure_releasable`` refuses.
        """
        released = await self._abandoned(identity_id, lock=True)
        workspace_ids: list[uuid.UUID] = []
        for organization_id in released:
            await self._organizations.anonymize(organization_id)
            workspace_ids.extend(await self._workspaces.get_ids_by_organization(organization_id))
        await self._invitations.delete_all_in_organizations(released)
        return ReleasedOrganizations(organization_ids=released, workspace_ids=workspace_ids)

    async def withdraw_invitations(self, email: str) -> None:
        """Delete every invitation to this address, in any organization, so none keeps mailing a person who left."""
        await self._invitations.delete_addressed_to(email)

    async def _abandoned(self, identity_id: uuid.UUID, *, lock: bool) -> list[uuid.UUID]:
        """The organizations nobody but this account is active in, raising for one it would leave ownerless."""
        abandoned: list[uuid.UUID] = []
        orphaned: list[str] = []
        memberships = await self._members.get_by_user(identity_id)
        if lock:
            # Locked in one order, so two deletions sharing organizations cannot each hold what the other waits for.
            locked = sorted({row.organization_id for row in memberships}, key=str)
            for organization_id in locked:
                await self._organizations.lock(organization_id)
            # Read again under the locks: a status read before them may have changed since.
            memberships = [
                row
                for row in await self._members.get_by_user(identity_id, refresh=True)
                if row.organization_id in locked
            ]
        for membership in memberships:
            organization_id = membership.organization_id
            active = membership.status == "active"
            others = await self._members.count_active(organization_id) - (1 if active else 0)
            if others == 0:
                abandoned.append(organization_id)
            elif (
                active and membership.role == "owner" and await self._members.count_active_owners(organization_id) <= 1
            ):
                organization = await self._organizations.get(organization_id)
                orphaned.append(organization.name if organization is not None else str(organization_id))
        if orphaned:
            raise AccountDeletionLastOwnerError(orphaned)
        return abandoned
