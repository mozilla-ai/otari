"""The rule admitting an SSO identity to the one organization its domain proves."""

import uuid
from datetime import datetime
from typing import Protocol

from gateway.core.email_domains import email_domain
from gateway.models.identity import OIDC_PROVIDER_PREFIX, OIDCConnection, ProviderIdentity
from gateway.models.tenancy import OrganizationDomain


class IssuerConnections(Protocol):
    """Lists the enabled connections an OIDC issuer resolves to."""

    async def list_enabled_by_resolved_issuer(self, issuer: str) -> list[OIDCConnection]: ...


class OrganizationConnections(Protocol):
    """Looks up one organization's own OIDC connection."""

    async def get_for_organization(self, organization_id: uuid.UUID) -> OIDCConnection | None: ...


class VerifiedDomainClaims(Protocol):
    """Looks up the one proven claim on an email domain."""

    async def get_verified_by_domain(self, domain: str) -> OrganizationDomain | None: ...


async def organization_for_sso_identity(
    external: ProviderIdentity,
    *,
    connections: IssuerConnections,
    domains: VerifiedDomainClaims,
    now: datetime,
) -> OIDCConnection | None:
    """Return the connection that admits ``external``, or ``None`` where none does.

    One organization must both hold a current proof of the identity's email domain and have a
    connection to the issuer that authenticated it. Either half alone admits nobody, because
    nothing proves an organization controls the issuer it names, so a connection on its own would
    let any organization collect whoever authenticates at a stranger's issuer.
    """
    if external.email is None:
        return None
    domain = email_domain(external.email)
    if domain is None:
        return None
    claim = await domains.get_verified_by_domain(domain)
    if claim is None or not claim.enabled or claim.proof_expired(now=now):
        return None

    issuer = external.provider.removeprefix(OIDC_PROVIDER_PREFIX)
    for connection in await connections.list_enabled_by_resolved_issuer(issuer):
        if connection.organization_id == claim.organization_id:
            return connection
    return None


async def organization_for_email_domain(
    email: str, *, connections: OrganizationConnections, domains: VerifiedDomainClaims, now: datetime
) -> uuid.UUID | None:
    """Return the id of the one organization whose enabled connection admits a sign-in at ``email``'s domain.

    None where the domain carries no current proof, and where the organization that proves it
    has no enabled connection, so the two answer alike.
    """
    domain = email_domain(email)
    if domain is None:
        return None
    claim = await domains.get_verified_by_domain(domain)
    if claim is None or not claim.enabled or claim.proof_expired(now=now):
        return None
    connection = await connections.get_for_organization(claim.organization_id)
    if connection is None or not connection.enabled:
        return None
    return claim.organization_id


__all__ = [
    "IssuerConnections",
    "OrganizationConnections",
    "VerifiedDomainClaims",
    "organization_for_email_domain",
    "organization_for_sso_identity",
]
