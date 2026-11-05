"""Whether an account already holds the link that an organization's own SSO enforcement admits."""

import uuid

from gateway.models.identity import oidc_identity_provider
from gateway.models.tenancy import Organization
from gateway.repositories.identity.oauth_identity_link_repository import OAuthIdentityLinkRepository
from gateway.repositories.identity.oidc_connection_repository import OIDCConnectionRepository


async def linked_providers_for_organization(
    organization: Organization,
    user_id: uuid.UUID,
    *,
    connections: OIDCConnectionRepository,
    links: OAuthIdentityLinkRepository,
) -> frozenset[str]:
    """Return the provider naming a sign-in link ``user_id`` already holds to ``organization``'s own connection.

    Empty where the organization has no connection, or the account holds no link to it.
    """
    connection = await connections.get_for_organization(organization.id)
    if connection is None:
        return frozenset()
    provider = oidc_identity_provider(connection.resolved_issuer)
    if await links.get_for_user(provider=provider, user_id=user_id) is None:
        return frozenset()
    return frozenset({provider})


__all__ = ["linked_providers_for_organization"]
