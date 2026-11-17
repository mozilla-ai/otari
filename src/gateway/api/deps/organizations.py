"""How a request reaches the organizations domain: organizations, workspaces, members and invitations."""

from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps._base import (
    ContainerDep,
    UnitOfWorkDep,
    get_config,
)
from gateway.api.deps.ports import ApiKeyFormatPortDep
from gateway.core.config import (
    GatewayConfig,
)
from gateway.core.database import get_db
from gateway.core.unit_of_work import UnitOfWork
from gateway.ports.rate_limit_store_port import RateLimitStorePort
from gateway.repositories.api_keys import ApiKeyRepositories
from gateway.repositories.budgets import BudgetRepositories
from gateway.repositories.identity import BlockedEmailRepository
from gateway.repositories.organizations import OrganizationRepositories
from gateway.repositories.tenancy import (
    OrganizationDomainRepository,
    OrganizationRepository,
    UserRepository,
    WorkspaceRepository,
)
from gateway.repositories.traces import TracesRepositories
from gateway.services.api_keys import ApiKeyMembershipListener
from gateway.services.budgets import BudgetMembershipListener
from gateway.services.identity import EmailBlocklist
from gateway.services.mail import Mailer
from gateway.services.oauth_service import provider_label
from gateway.services.organizations import (
    DeliveryRule,
    ExpiredInvitationCleanup,
    InvitationDelivery,
    InvitationFactsReader,
    InvitationMailLimit,
    OrganizationClaimedDomains,
    UserAccess,
)
from gateway.services.tenancy import (
    InvitationClaimSettings,
    MembershipListener,
    MembershipListeners,
    OrganizationDomainService,
    OrganizationService,
)
from gateway.services.tenancy.authorization import WorkspaceAccess
from gateway.services.tenancy.workspace_activation_service import WorkspaceActivationService
from gateway.services.tenancy.workspace_service import WorkspaceService
from gateway.services.tools import McpGrantMembershipListener
from gateway.services.traces import TraceMembershipListener


def get_user_access(uow: UnitOfWorkDep) -> UserAccess:
    """Build the organizations domain's read of users' access lists on the request's Unit of Work."""
    return UserAccess(OrganizationRepositories.on(uow))


def get_membership_listener(uow: UnitOfWorkDep) -> MembershipListener:
    """Return the listener that keeps members' budget ceilings, own keys, session shares and MCP access in step.

    It writes through the request's Unit of Work, so a service that changes membership must be built on the same one.
    """
    return MembershipListeners(
        BudgetMembershipListener(BudgetRepositories.on(uow)),
        ApiKeyMembershipListener(ApiKeyRepositories.on(uow)),
        TraceMembershipListener(TracesRepositories.on(uow)),
        McpGrantMembershipListener(uow),
    )


MembershipListenerDep = Annotated[MembershipListener, Depends(get_membership_listener)]


def get_organization_service(db: Annotated[AsyncSession, Depends(get_db)]) -> OrganizationService:
    """Build the request's organization service.

    It reads only. A membership write needs the listener this pairing leaves unset.
    """
    return OrganizationService(db, membership_listener=None)


OrganizationServiceDep = Annotated[OrganizationService, Depends(get_organization_service)]


def get_membership_organization_service(
    db: Annotated[AsyncSession, Depends(get_db)],
    uow: UnitOfWorkDep,
    membership_listener: MembershipListenerDep,
) -> OrganizationService:
    """Build the request's organization service for writing memberships and creating workspaces."""
    return OrganizationService(db, membership_listener=membership_listener, uow=uow)


MembershipOrganizationServiceDep = Annotated[OrganizationService, Depends(get_membership_organization_service)]


def get_organization_domain_service(db: Annotated[AsyncSession, Depends(get_db)]) -> OrganizationDomainService:
    """Build the email-domain service on the request's session."""
    return OrganizationDomainService(db)


OrganizationDomainServiceDep = Annotated[OrganizationDomainService, Depends(get_organization_domain_service)]


def get_invitation_claim_settings(config: Annotated[GatewayConfig, Depends(get_config)]) -> InvitationClaimSettings:
    """Return what this deployment asks of, and offers, an invitee who chooses a first password from an invitation link.

    The terms it publishes, and the ways in a refusal can offer instead: its sign-in providers by display name,
    and a reset link where mail can send one.
    """
    return InvitationClaimSettings(
        terms_url=config.terms_url,
        sign_in_providers=tuple(provider_label(name) for name in config.oauth_providers),
        can_send_links=Mailer(config).can_send_links,
    )


InvitationClaimSettingsDep = Annotated[InvitationClaimSettings, Depends(get_invitation_claim_settings)]


def get_invitation_mail_limit(
    config: Annotated[GatewayConfig, Depends(get_config)], container: ContainerDep
) -> InvitationMailLimit:
    """Return the invitation mail limits, counted in the rate-limit store this build bound."""
    return InvitationMailLimit(
        container.resolve(RateLimitStorePort, None),
        per_organization=config.invitation_mail_limit_per_organization,
        per_address=config.invitation_mail_limit_per_address,
    )


InvitationMailLimitDep = Annotated[InvitationMailLimit, Depends(get_invitation_mail_limit)]


def build_invitation_delivery(uow: UnitOfWork, config: GatewayConfig) -> InvitationDelivery:
    """Build the sender of due invitation emails on a worker job's Unit of Work."""
    domains = OrganizationClaimedDomains(OrganizationDomainRepository(uow))
    return InvitationDelivery(
        uow,
        OrganizationRepositories.on(uow),
        rule=DeliveryRule(domains, EmailBlocklist(uow, BlockedEmailRepository(uow))),
        facts=InvitationFactsReader(
            domains=domains,
            organizations=OrganizationRepository(uow),
            users=UserRepository(uow),
            workspaces=WorkspaceRepository(uow),
        ),
        mailer=Mailer(config),
    )


def build_invitation_cleanup(uow: UnitOfWork) -> ExpiredInvitationCleanup:
    """Build the remover of expired invitations on a worker job's Unit of Work."""
    return ExpiredInvitationCleanup(uow, OrganizationRepositories.on(uow))


def get_workspace_service(
    db: Annotated[AsyncSession, Depends(get_db)],
    uow: UnitOfWorkDep,
    membership_listener: MembershipListenerDep,
) -> WorkspaceService:
    return WorkspaceService(db, uow=uow, membership_listener=membership_listener)


WorkspaceServiceDep = Annotated[WorkspaceService, Depends(get_workspace_service)]


def get_workspace_activation_service(
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    key_format: ApiKeyFormatPortDep,
) -> WorkspaceActivationService:
    return WorkspaceActivationService(db, config, key_format)


WorkspaceActivationServiceDep = Annotated[WorkspaceActivationService, Depends(get_workspace_activation_service)]


def get_workspace_access(
    db: Annotated[AsyncSession, Depends(get_db)],
    organizations: OrganizationServiceDep,
) -> WorkspaceAccess:
    """Build the workspace visibility and management checks on the request's session."""
    return WorkspaceAccess(db, organizations)


WorkspaceAccessDep = Annotated[WorkspaceAccess, Depends(get_workspace_access)]
