from dataclasses import dataclass
from typing import Self

from gateway.core.unit_of_work import UnitOfWork
from gateway.repositories.organizations.membership_origin_repository import MembershipOriginRepository
from gateway.repositories.organizations.organization_invitation_repository import OrganizationInvitationRepository
from gateway.repositories.tenancy import (
    OrganizationMemberRepository,
    OrganizationRepository,
    WorkspaceRepository,
)
from gateway.repositories.users_repository import UserAccessRepository


@dataclass(frozen=True)
class OrganizationRepositories:
    """The organizations domain's repositories, all on one Unit of Work."""

    invitations: OrganizationInvitationRepository
    members: OrganizationMemberRepository
    membership_origins: MembershipOriginRepository
    organizations: OrganizationRepository
    users: UserAccessRepository
    workspaces: WorkspaceRepository

    @classmethod
    def on(cls, uow: UnitOfWork) -> Self:
        """Build every repository on this Unit of Work."""
        return cls(
            invitations=OrganizationInvitationRepository(uow),
            members=OrganizationMemberRepository(uow),
            membership_origins=MembershipOriginRepository(uow),
            organizations=OrganizationRepository(uow),
            users=UserAccessRepository(uow),
            workspaces=WorkspaceRepository(uow),
        )
