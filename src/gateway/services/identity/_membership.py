"""The rule admitting an SSO identity to the one organization its domain proves."""

from datetime import datetime
from typing import Protocol

from gateway.core.email_domains import email_domain
from gateway.models.identity import OIDC_PROVIDER_PREFIX, OIDCConnection, ProviderIdentity
from gateway.models.tenancy import OrganizationDomain


class IssuerConnections(Protocol):
    """Lists the enabled connections an OIDC issuer resolves to."""

    async def list_enabled_by_resolved_issuer(self, issuer: str) -> list[OIDCConnection]: ...


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


__all__ = ["IssuerConnections", "VerifiedDomainClaims", "organization_for_sso_identity"]
