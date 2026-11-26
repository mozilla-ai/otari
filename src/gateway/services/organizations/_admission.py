"""Admitting an invited person to an organization and the workspaces the invitation offered."""

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from gateway.models.tenancy import OrganizationMemberRole, WorkspaceAssignmentRequest


@dataclass(frozen=True, kw_only=True)
class AdmittedMember:
    """The membership an admission settled on, and whether it had to create one."""

    created: bool
    member_id: uuid.UUID


class MembershipAdmission(Protocol):
    """Makes a person an active member of an organization, with the workspaces an invitation offered them."""

    async def admit(
        self,
        *,
        organization_id: uuid.UUID,
        role: OrganizationMemberRole,
        user_id: uuid.UUID,
        workspaces: Sequence[WorkspaceAssignmentRequest],
    ) -> AdmittedMember:
        """Make this person an active member with this role, granting each workspace that still exists.

        The organization becomes the person's active one, and the member can own API keys from then on.
        A workspace that has been deleted is dropped rather than refused.

        Raises:
            MembershipUpdateError: the role offered would leave the organization with no active owner.
            OrganizationMemberAddressUnprovenError: the person holds no membership here and only another
                organization recorded their address.
            OrganizationMemberDeactivatedError: the account is deactivated, so a membership grants nothing.
            OrganizationNotFoundError: the organization no longer exists.

        """
        ...
