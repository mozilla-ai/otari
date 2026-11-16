"""What an invitee is told about an invitation to join an organization."""

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from gateway.models.organizations import WorkspaceAssignment
from gateway.models.tenancy import OrganizationMemberRole
from gateway.repositories.tenancy import OrganizationRepository, UserRepository, WorkspaceRepository
from gateway.services.organizations._claimed_domains import ClaimedDomains, proven_domains


@dataclass(frozen=True, kw_only=True)
class InvitationFacts:
    """What an invitee is told about an invitation before deciding to join.

    ``inviter_email`` is None unless the inviter's account is active and the inviter has verified the address.
    ``proven_domains`` holds each email domain that the organization has proven, unless the proof has expired.
    ``workspace_names`` holds the workspaces the invitee joins, A to Z.
    """

    inviter_email: str | None
    inviter_name: str | None
    organization_name: str
    proven_domains: tuple[str, ...]
    role: OrganizationMemberRole
    workspace_names: tuple[str, ...]


class InvitationTerms(Protocol):
    """The values of an invitation that decide what its invitee is told."""

    @property
    def invited_by_user_id(self) -> uuid.UUID | None: ...

    @property
    def organization_id(self) -> uuid.UUID: ...

    @property
    def role(self) -> OrganizationMemberRole: ...

    @property
    def workspace_assignments(self) -> Sequence[WorkspaceAssignment]: ...


def _workspace_ids(assignments: Sequence[WorkspaceAssignment]) -> list[uuid.UUID]:
    ids = []
    for assignment in assignments:
        try:
            ids.append(uuid.UUID(str(assignment.get("workspace_id"))))
        except ValueError:
            continue
    return ids


class InvitationFactsReader:
    """Reads what an invitee is told about an invitation."""

    def __init__(
        self,
        *,
        domains: ClaimedDomains,
        organizations: OrganizationRepository,
        users: UserRepository,
        workspaces: WorkspaceRepository,
    ) -> None:
        self._domains = domains
        self._organizations = organizations
        self._users = users
        self._workspaces = workspaces

    async def read(self, terms: InvitationTerms, *, now: datetime) -> InvitationFacts | None:
        """Return the facts of an invitation as they are at ``now``, or None when the organization no longer exists.

        A workspace that is no longer in the organization is left out.
        A workspace whose stored ID is not a UUID is left out too.

        NOTE: Callers must call this inside an open Unit of Work block.
        """
        organization_id = terms.organization_id
        organization = await self._organizations.get(organization_id)
        if organization is None:
            return None
        inviter = None if terms.invited_by_user_id is None else await self._users.get(terms.invited_by_user_id)
        inviter_email = None
        inviter_name = None
        if inviter is not None:
            inviter_name = inviter.full_name or None
            # An address that an admin vouched for has not been verified by its owner.
            if inviter.is_active and inviter.email_verified_at is not None and inviter.email_vouched_at is None:
                inviter_email = inviter.email
        claims = await self._domains.for_organization(organization_id)
        workspaces = await self._workspaces.get_by_ids(_workspace_ids(terms.workspace_assignments))
        names = [workspace.name for workspace in workspaces if workspace.organization_id == organization_id]
        return InvitationFacts(
            inviter_email=inviter_email,
            inviter_name=inviter_name,
            organization_name=organization.name,
            proven_domains=proven_domains(claims, now=now),
            role=terms.role,
            workspace_names=tuple(sorted(names, key=lambda name: (name.casefold(), name))),
        )
