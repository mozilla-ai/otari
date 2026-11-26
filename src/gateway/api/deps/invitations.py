"""How a request reaches the invitation flow.

The flow spans three domains, so it belongs to none of their modules:
organizations owns the invitation, identity answers whether a connection
refuses a way in, and tenancy admits the member. The three builders share most
of their collaborators and are kept together for that reason.
"""

from typing import Annotated

from fastapi import Depends

from gateway.api.deps._base import ContainerDep, UnitOfWorkDep, get_config
from gateway.api.deps.identity import OIDCConnectionServiceDep
from gateway.api.deps.organizations import MembershipOrganizationServiceDep, invitation_policy_of
from gateway.core.config import GatewayConfig
from gateway.ports.rate_limit_store_port import RateLimitStorePort
from gateway.repositories.identity import BlockedEmailRepository
from gateway.repositories.organizations import OrganizationRepositories
from gateway.repositories.tenancy import (
    OrganizationDomainRepository,
    OrganizationRepository,
    UserRepository,
    WorkspaceRepository,
)
from gateway.services.identity import EmailBlocklist
from gateway.services.mail import Mailer
from gateway.services.organizations import (
    DeliveryRule,
    InvitationAcceptance,
    InvitationFactsReader,
    InvitationLink,
    Invitations,
    OrganizationClaimedDomains,
)


def get_invitation_link(uow: UnitOfWorkDep, config: Annotated[GatewayConfig, Depends(get_config)]) -> InvitationLink:
    """Build the reader of what one invitation token still opens, on the request's Unit of Work."""
    blocked = EmailBlocklist(uow, BlockedEmailRepository(uow))
    return InvitationLink(
        uow,
        OrganizationRepositories.on(uow),
        blocked=blocked,
        delivery=DeliveryRule(OrganizationClaimedDomains(OrganizationDomainRepository(uow)), blocked),
        mailer=Mailer(config),
    )


InvitationLinkDep = Annotated[InvitationLink, Depends(get_invitation_link)]


def get_invitations(
    uow: UnitOfWorkDep, config: Annotated[GatewayConfig, Depends(get_config)], container: ContainerDep
) -> Invitations:
    """Build the invitation service on the request's Unit of Work, held to this deployment's limits."""
    return Invitations(
        uow,
        OrganizationRepositories.on(uow),
        domains=OrganizationClaimedDomains(OrganizationDomainRepository(uow)),
        mailer=Mailer(config),
        policy=invitation_policy_of(config),
        store=container.resolve(RateLimitStorePort, None),
    )


InvitationsDep = Annotated[Invitations, Depends(get_invitations)]


def get_invitation_acceptance(
    uow: UnitOfWorkDep,
    link: InvitationLinkDep,
    organizations: MembershipOrganizationServiceDep,
    connections: OIDCConnectionServiceDep,
) -> InvitationAcceptance:
    """Build the acceptor of one invitation on the request's Unit of Work.

    ``organizations`` admits the member and ``connections`` answers whether an
    organization refuses a way in outside its own connection. Each is named by
    the protocol the service declares, so the service names neither.

    NOTE: Admitting writes a membership, so the organization service must be the
    pairing carrying a Unit of Work and the membership listener. The read-only
    pairing satisfies the protocol and refuses at the first write.
    """
    return InvitationAcceptance(
        uow,
        OrganizationRepositories.on(uow),
        admission=organizations,
        facts=InvitationFactsReader(
            domains=OrganizationClaimedDomains(OrganizationDomainRepository(uow)),
            organizations=OrganizationRepository(uow),
            users=UserRepository(uow),
            workspaces=WorkspaceRepository(uow),
        ),
        link=link,
        sso=connections,
    )


InvitationAcceptanceDep = Annotated[InvitationAcceptance, Depends(get_invitation_acceptance)]
