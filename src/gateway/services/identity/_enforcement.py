"""Whether an account already holds the link that an organization's own SSO enforcement admits."""

import uuid

from gateway.models.identity import OIDC_PROVIDER_PREFIX, oidc_identity_provider
from gateway.models.tenancy import Organization, User
from gateway.repositories.identity.oauth_identity_link_repository import OAuthIdentityLinkRepository
from gateway.repositories.identity.oidc_connection_repository import OIDCConnectionRepository
from gateway.repositories.identity.webauthn_credential_repository import WebAuthnCredentialRepository
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


async def has_route_without(
    user: User,
    *,
    provider: str,
    passkey_rp_id: str | None,
    offered_providers: frozenset[str],
    links: OAuthIdentityLinkRepository,
    passkeys: WebAuthnCredentialRepository,
) -> bool:
    """Whether ``user`` can still sign in with ``provider`` unavailable.

    A password and a passkey need no provider at all. A link needs the provider it names to still
    be on offer, which is why both of the deployment's own answers are asked for rather than assumed.

    NOTE: ``passkey_rp_id`` is the relying party this deployment serves, and ``None`` where it
    serves none, because a passkey registered under another is one no authenticator will offer.

    NOTE: ``offered_providers`` is the built-in providers an operator has configured. A link at an
    organization's own connection is a way in whatever this holds, because the organization
    administers that provider rather than the deployment.
    """
    if user.hashed_password is not None:
        return True
    if passkey_rp_id is not None and await passkeys.exists_for_user(user.id, rp_id=passkey_rp_id):
        return True
    others = await links.get_other_providers_for_user(provider=provider, user_id=user.id)
    return any(other.startswith(OIDC_PROVIDER_PREFIX) or other in offered_providers for other in others)


async def sso_enforcement_refuses_every_membership(
    user: User,
    *,
    organizations: OrganizationService,
    connections: OIDCConnectionRepository,
    links: OAuthIdentityLinkRepository,
) -> bool:
    """Whether every organization ``user`` actively belongs to refuses them a way in outside its own connection.

    False where the account belongs to no organization, so an account outside every organization is
    refused nothing.
    """
    memberships = await organizations.members.get_by_user(user.id, active_only=True)
    if not memberships:
        return False
    # Resolved once: whether the deployment admits this caller does not vary by organization.
    is_operator = await organizations.is_deployment_operator(user)
    for membership in memberships:
        organization = await organizations.organizations.get(membership.organization_id)
        if organization is None or not organization.sso_required:
            return False
        linked = await has_link_to_organization(organization, user.id, connections=connections, links=links)
        if not sso_blocks_membership(
            organization=organization,
            membership=membership,
            is_linked=linked,
            is_deployment_operator=is_operator,
        ):
            return False
    return True


__all__ = ["has_link_to_organization", "has_route_without", "sso_enforcement_refuses_every_membership"]
