"""Whether an account already holds the link that an organization's own SSO enforcement admits."""

import uuid

from gateway.models.identity import oidc_identity_provider
from gateway.models.tenancy import Organization, User
from gateway.repositories.identity.oauth_identity_link_repository import OAuthIdentityLinkRepository
from gateway.repositories.identity.oidc_connection_repository import OIDCConnectionRepository
from gateway.services.tenancy import OrganizationService, sso_blocks_membership


async def has_link_to_organization(
    organization: Organization,
    user_id: uuid.UUID,
    *,
    connections: OIDCConnectionRepository,
    links: OAuthIdentityLinkRepository,
) -> bool:
    """Whether ``user_id`` already holds a sign-in link to ``organization``'s own connection.

    False where the organization has no connection, as well as where the account holds no link.
    """
    connection = await connections.get_for_organization(organization.id)
    if connection is None:
        return False
    provider = oidc_identity_provider(connection.resolved_issuer)
    return await links.get_for_user(provider=provider, user_id=user_id) is not None


async def sso_enforcement_blocks_any_membership(
    user: User,
    *,
    organizations: OrganizationService,
    connections: OIDCConnectionRepository,
    links: OAuthIdentityLinkRepository,
) -> bool:
    """Whether some organization ``user`` actively belongs to requires sign-in through its own connection."""
    for membership in await organizations.members.get_by_user(user.id, active_only=True):
        organization = await organizations.organizations.get(membership.organization_id)
        if organization is None or not organization.sso_required:
            continue
        linked = await has_link_to_organization(organization, user.id, connections=connections, links=links)
        if sso_blocks_membership(organization=organization, user=user, is_linked=linked):
            return True
    return False


__all__ = ["has_link_to_organization", "sso_enforcement_blocks_any_membership"]
