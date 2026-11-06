"""Manage an organization's own OpenID Connect sign-in connection."""

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from apron_auth import BrowserBoundAuthorization, StoredStateClient

from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions.identity_exceptions import (
    OIDCConnectionConflictError,
    OIDCConnectionEnforcedError,
    OIDCConnectionNotFoundError,
    OIDCConnectionSecretRequiredError,
    OIDCConnectionUnprovenError,
    OIDCDiscoveryError,
    SsoBreakGlassRequiredError,
)
from gateway.exceptions.shared_exceptions import SecretBoxUnavailableTenancyError
from gateway.log_config import logger
from gateway.models.identity import MAX_OIDC_ISSUER_LENGTH, OIDC_PROVIDER_PREFIX, OIDCConnection
from gateway.models.tenancy import DOMAIN_PROOF_TTL, Organization, OrganizationDomain, User
from gateway.repositories.identity import IdentityRepositories
from gateway.repositories.tenancy import OrganizationDomainRepository
from gateway.schemas.identity import (
    OIDCConnectionDomain,
    OIDCConnectionPublic,
    OIDCConnectionRequest,
    OrganizationSsoEnforcementPublic,
    OrganizationSsoPreflightMemberPublic,
    OrganizationSsoPreflightPublic,
)
from gateway.services.identity._connection_clients import client_for, pending_flow_provider
from gateway.services.identity._discovery import DISCOVERY_TTL, DiscoveredProvider, discover_connection_metadata
from gateway.services.identity._enforcement import (
    linked_providers_for_organization,
    sso_enforcement_blocks_any_membership,
)
from gateway.services.identity._membership import organization_for_email_domain
from gateway.services.secret_box import encrypt_secret, secret_box_configured
from gateway.services.tenancy import OrganizationService, sso_blocks_membership

# The preflight count is exact; the member list is a sample, so a large organization's
# response stays small enough for the dashboard to render in one page.
_PREFLIGHT_SAMPLE_LIMIT = 50


@dataclass(frozen=True)
class _StoredSecret:
    """An encrypted client secret and the last four characters served in its place."""

    encrypted: str
    last4: str


def _require_a_secret_to_keep(request: OIDCConnectionRequest, stored: OIDCConnection | None) -> None:
    """Refuse a submission that leaves the client secret out with no stored one that may stand in.

    The stored secret proves this deployment is one client at one issuer, so it may only be kept
    where the submission still names both. Carrying it to an issuer or a client ID the caller just
    changed would hand it to whoever controls the new one.

    Raises:
        OIDCConnectionSecretRequiredError: There is no stored secret that may stand in.

    """
    if request.client_secret is not None:
        return
    if stored is None or stored.issuer != request.issuer or stored.client_id != request.client_id:
        raise OIDCConnectionSecretRequiredError


def _secret_to_store(request: OIDCConnectionRequest, stored: OIDCConnection | None) -> _StoredSecret:
    """The client secret to store: the submitted one, encrypted, or the stored one kept as it is.

    Raises:
        OIDCConnectionSecretRequiredError: Nothing was submitted and a concurrent request removed
            the connection whose secret would have been kept.

    """
    if request.client_secret is not None:
        return _StoredSecret(encrypted=encrypt_secret(request.client_secret), last4=request.client_secret[-4:])
    if stored is None:
        raise OIDCConnectionSecretRequiredError
    return _StoredSecret(encrypted=stored.encrypted_client_secret, last4=stored.client_secret_last4)


class OIDCConnectionService:
    """Save, test, read, delete, and serve the metadata of an organization's own OIDC connection.

    Every ``*_for_user`` method is gated on
    ``OrganizationService.require_active_organization_management_access``, checked before any
    outbound discovery request, so a caller who may not manage the organization cannot make this
    deployment dial a host of their choosing. A method that writes runs in a Unit of Work block, so
    what it and its collaborators stage commits together.
    """

    def __init__(
        self,
        uow: UnitOfWork,
        repositories: IdentityRepositories,
        *,
        organizations: OrganizationService,
        domains: OrganizationDomainRepository,
        allow_private_hosts: bool,
    ) -> None:
        self._uow = uow
        self._connections = repositories.connections
        self._links = repositories.links
        self._sessions = repositories.sessions
        self._organizations = organizations
        self._domains = domains
        self._allow_private_hosts = allow_private_hosts

    async def _managed_organization(self, user: User) -> Organization:
        """The caller's active organization, with their authority to manage it proven.

        The only way this unit proves authority over an organization, so a management method added
        here cannot obtain one without the check.
        """
        organization = await self._organizations.get_active_organization_for_user(user)
        await self._organizations.require_active_organization_management_access(user=user, organization=organization)
        return organization

    async def get_for_user(self, *, user: User) -> OIDCConnectionPublic | None:
        """Return the caller's organization's connection, or None. Owners and admins only."""
        organization = await self._managed_organization(user)

        async with self._uow:
            connection = await self._connections.get_for_organization(organization.id)
            if connection is None:
                return None
            domains = await self._domains.list_by_organization(organization.id)
            return self._to_public(connection, domains)

    async def save_for_user(self, *, user: User, request: OIDCConnectionRequest) -> OIDCConnectionPublic:
        """Discover the submitted issuer and store the caller's organization's connection. Owners and admins only.

        Discovery runs before any write, and only a successful discovery opens the transaction that
        stores it, so a refused issuer leaves an existing connection, its document, and its secret
        exactly as they were.

        Raises:
            OIDCConnectionConflictError: Another request changed this organization's connection while
                this one was storing it.
            OIDCConnectionSecretRequiredError: The submission left the client secret out where the
                stored one cannot stand in.
            OIDCDiscoveryError: The issuer could not be read into a usable configuration.
            SecretBoxUnavailableTenancyError: This deployment has no secret key to encrypt the client
                secret with.
        """
        organization = await self._managed_organization(user)
        if not secret_box_configured():
            raise SecretBoxUnavailableTenancyError(stored="an organization's own sign-in connection")

        async with self._uow:
            stored = await self._connections.get_for_organization(organization.id)
        _require_a_secret_to_keep(request, stored)

        await self._uow.release()
        discovered = await discover_connection_metadata(request.issuer, allow_private_hosts=self._allow_private_hosts)
        if len(discovered.resolved_issuer) > MAX_OIDC_ISSUER_LENGTH:
            # Every sign-in names a provider derived from the stored issuer, so one too long to name
            # would refuse the whole organization rather than this request.
            raise OIDCDiscoveryError

        async with self._uow:
            connection = await self._connections.get_for_organization(organization.id)
            secret = _secret_to_store(request, connection)
            if connection is None:
                connection = await self._store_first(
                    organization_id=organization.id,
                    request=request,
                    discovered=discovered,
                    secret=secret,
                )
            else:
                self._overwrite(connection, request=request, discovered=discovered, secret=secret)
            domains = await self._domains.list_by_organization(organization.id)
            return self._to_public(connection, domains)

    async def test_for_user(self, *, user: User) -> OIDCConnectionPublic:
        """Re-discover the caller's organization's own issuer and store the result. Owners and admins only.

        Raises:
            OIDCConnectionNotFoundError: The organization has not configured a connection yet.
            OIDCDiscoveryError: The issuer could not be read into a usable configuration. Nothing stored
                is changed.
        """
        organization = await self._managed_organization(user)

        async with self._uow:
            connection = await self._connections.get_for_organization(organization.id)
        if connection is None:
            raise OIDCConnectionNotFoundError
        issuer, connection_id = connection.issuer, connection.id

        await self._uow.release()
        discovered = await discover_connection_metadata(issuer, allow_private_hosts=self._allow_private_hosts)

        async with self._uow:
            # NOTE: Re-read rather than reuse the row fetched above, which may have been edited
            # or deleted while discovery was out.
            current = await self._connections.get_by_id(connection_id)
            if current is None:
                raise OIDCConnectionNotFoundError
            current.resolved_issuer = discovered.resolved_issuer
            current.discovery_document = discovered.document
            current.discovered_at = discovered.discovered_at
            self._connections.stage(current)
            domains = await self._domains.list_by_organization(organization.id)
            return self._to_public(current, domains)

    async def delete_for_user(self, *, user: User) -> None:
        """Remove the caller's organization's connection. Owners and admins only.

        Raises:
            OIDCConnectionNotFoundError: The organization has not configured a connection yet.
            OIDCConnectionEnforcedError: The organization still requires sign-in through this
                connection.
        """
        organization = await self._managed_organization(user)

        async with self._uow:
            connection = await self._connections.get_for_organization(organization.id)
            if connection is None:
                raise OIDCConnectionNotFoundError
            if organization.sso_required:
                raise OIDCConnectionEnforcedError
            await self._connections.delete(connection)

    async def metadata_for_sign_in(self, connection: OIDCConnection) -> DiscoveredProvider:
        """Return this connection's provider metadata for the sign-in path.

        The connection is one the caller resolved already, and no authority check runs,
        because a person signing in has no authority over the organization they sign in to.
        A refresh that fails returns the stored document rather than raising, so a provider's
        outage does not lock an organization's own people out of their own deployment.

        NOTE: Metadata past ``DISCOVERY_TTL`` is re-discovered, and only that path commits what
        the caller staged. The caller must hold no block open, and must commit anything of its
        own it needs to be durable before calling this.
        """
        if datetime.now(UTC) - connection.discovered_at <= DISCOVERY_TTL:
            return _stored_metadata(connection)

        issuer, connection_id, organization_id = connection.issuer, connection.id, connection.organization_id

        # Past the TTL every sign-in discovers at once, so a held connection costs most here. This
        # method stages nothing of its own, so the answer is not checked.
        await self._uow.release()
        try:
            discovered = await discover_connection_metadata(issuer, allow_private_hosts=self._allow_private_hosts)
        except OIDCDiscoveryError:
            logger.warning(
                "OIDC metadata refresh failed for organization %s; serving the stored document",
                organization_id,
            )
            return _stored_metadata(connection)

        async with self._uow:
            # NOTE: Re-read rather than reuse the row above, which may have been edited or deleted
            # while discovery was out.
            current = await self._connections.get_by_id(connection_id)
            if current is not None:
                current.resolved_issuer = discovered.resolved_issuer
                current.discovery_document = discovered.document
                current.discovered_at = discovered.discovered_at
                self._connections.stage(current)
        return discovered

    async def connection_for_sign_in(self, *, slug: str) -> OIDCConnection | None:
        """Return the organization's enabled connection at ``slug``, for the sign-in path.

        No authority check runs: a person signing in has no authority over the organization they
        sign in to. None for a slug naming no organization and for one naming an organization with
        no enabled connection alike, so the answer does not reveal which of the two it is.
        """
        async with self._uow:
            organization = await self._organizations.organizations.get_by_slug(slug)
            if organization is None:
                return None
            connection = await self._connections.get_for_organization(organization.id)
        return connection if connection is not None and connection.enabled else None

    async def organization_for_sign_in(self, *, email: str) -> Organization | None:
        """Return the one organization whose enabled connection admits a sign-in at ``email``'s domain.

        None where the domain carries no current proof, and where the organization that proves it
        has no enabled connection, so the two answer alike.
        """
        async with self._uow:
            organization_id = await organization_for_email_domain(
                email, connections=self._connections, domains=self._domains, now=datetime.now(UTC)
            )
            if organization_id is None:
                return None
            return await self._organizations.organizations.get(organization_id)

    async def connection_for_pending_state(self, state: str) -> OIDCConnection | None:
        """Return the connection one pending SSO flow's own ``state`` names, or None.

        None where ``state`` names no pending sign-in through an organization's own connection,
        including one minted for a built-in provider, or one whose connection no longer exists or
        is disabled.
        """
        async with self._uow:
            provider = await pending_flow_provider(self._connections.db, state)
            if provider is None or not provider.startswith(OIDC_PROVIDER_PREFIX):
                return None
            try:
                connection_id = uuid.UUID(provider.removeprefix(OIDC_PROVIDER_PREFIX))
            except ValueError:
                return None
            connection = await self._connections.get_by_id(connection_id)
        return connection if connection is not None and connection.enabled else None

    async def start_sign_in(
        self, connection: OIDCConnection, metadata: DiscoveredProvider, *, redirect_uri: str, allow_private_hosts: bool
    ) -> BrowserBoundAuthorization:
        """Mint an authorization URL for ``connection``'s own provider, and stage the pending state."""
        async with self._uow:
            client = client_for(
                connection,
                metadata,
                db=self._connections.db,
                redirect_uri=redirect_uri,
                allow_private_hosts=allow_private_hosts,
            )
            return await client.get_authorization_url(redirect_uri=redirect_uri)

    async def client_for_sign_in(
        self, connection: OIDCConnection, metadata: DiscoveredProvider, *, redirect_uri: str, allow_private_hosts: bool
    ) -> StoredStateClient:
        """Build the apron-auth client ``connection``'s own provider signs in through."""
        async with self._uow:
            return client_for(
                connection,
                metadata,
                db=self._connections.db,
                redirect_uri=redirect_uri,
                allow_private_hosts=allow_private_hosts,
            )

    async def record_first_sign_in(self, connection: OIDCConnection) -> None:
        """Record that a sign-in through ``connection`` succeeded.

        No authority check runs, because a person signing in has no authority over the
        organization they sign in to. A connection that already carries a record keeps its
        first one.
        """
        async with self._uow:
            current = await self._connections.get_by_id(connection.id)
            if current is None or current.first_sign_in_at is not None:
                return
            current.first_sign_in_at = datetime.now(UTC)
            self._connections.stage(current)

    async def get_enforcement_for_user(self, *, user: User) -> OrganizationSsoEnforcementPublic:
        """Return the caller's organization's current sign-in enforcement. Owners and admins only."""
        organization = await self._managed_organization(user)
        return OrganizationSsoEnforcementPublic(
            required=organization.sso_required,
            break_glass_user_id=organization.sso_break_glass_user_id,
            break_glass_waived=organization.sso_break_glass_waived,
        )

    async def preflight_sso_enforcement_for_user(self, *, user: User) -> OrganizationSsoPreflightPublic:
        """Report who the caller's organization's sign-in enforcement would affect. Owners and admins only.

        An active member is affected when they hold no sign-in link to the connection: they would
        get a new account the first time they use it, and any API key on their current account
        would stay there rather than moving.

        Raises:
            OIDCConnectionNotFoundError: The organization has not configured a connection yet.
        """
        organization = await self._managed_organization(user)

        async with self._uow:
            connection = await self._connections.get_for_organization(organization.id)
            if connection is None:
                raise OIDCConnectionNotFoundError
            provider = oidc_identity_provider(connection.resolved_issuer)
            total, affected_count = await self._organizations.members.count_active_members_by_link(
                organization.id, provider=provider
            )
            sample = await self._organizations.members.list_active_members_without_link(
                organization.id, provider=provider, limit=_PREFLIGHT_SAMPLE_LIMIT
            )

        return OrganizationSsoPreflightPublic(
            total_members=total,
            affected_count=affected_count,
            affected_members=[
                OrganizationSsoPreflightMemberPublic(
                    user_id=member.user_id, email=member_user.email, full_name=member_user.full_name
                )
                for member, member_user in sample
            ],
            affected_members_truncated=affected_count > len(sample),
        )

    async def set_sso_enforcement_for_user(
        self,
        *,
        user: User,
        required: bool,
        break_glass_user_id: uuid.UUID | None,
        break_glass_waived: bool,
    ) -> None:
        """Set whether the caller's organization requires sign-in through its own connection. Owners and admins only.

        Setting it to ``True`` additionally revokes the live session of every active member it
        currently refuses. Turning it off has no precondition, so an organization can always
        undo it, even with no connection configured.

        Raises:
            OIDCConnectionNotFoundError: ``required`` is set, and the organization has not
                configured a connection yet.
            OIDCConnectionUnprovenError: ``required`` is set before anybody has signed in through
                the connection.
            SsoBreakGlassRequiredError: ``required`` is set without naming one of the
                organization's own owners to keep a non-SSO route, or waiving that exception.
        """
        organization = await self._managed_organization(user)

        async with self._uow:
            connection: OIDCConnection | None = None
            if required:
                connection = await self._require_connection_ready_for_enforcement(
                    organization, break_glass_user_id=break_glass_user_id, break_glass_waived=break_glass_waived
                )
            await self._organizations.organizations.update_organization(
                organization,
                {
                    "sso_required": required,
                    "sso_break_glass_user_id": break_glass_user_id,
                    "sso_break_glass_waived": break_glass_waived,
                },
            )
            if connection is not None:
                await self._revoke_blocked_sessions(organization, connection)

    async def _require_connection_ready_for_enforcement(
        self, organization: Organization, *, break_glass_user_id: uuid.UUID | None, break_glass_waived: bool
    ) -> OIDCConnection:
        """Return this organization's connection, proven fit to require sign-in through.

        Raises:
            OIDCConnectionNotFoundError: The organization has not configured a connection yet.
            OIDCConnectionUnprovenError: Nobody has signed in through the connection yet.
            SsoBreakGlassRequiredError: No owner is named to keep a non-SSO route and none is
                waived, or the named account is not one of the organization's own owners.
        """
        connection = await self._connections.get_for_organization(organization.id)
        if connection is None:
            raise OIDCConnectionNotFoundError
        if connection.first_sign_in_at is None:
            raise OIDCConnectionUnprovenError
        if break_glass_user_id is None and not break_glass_waived:
            raise SsoBreakGlassRequiredError
        if break_glass_user_id is not None:
            owner = await self._organizations.members.get_active_by_organization_and_user(
                organization.id, break_glass_user_id
            )
            if owner is None or owner.role != "owner":
                raise SsoBreakGlassRequiredError
        return connection

    async def _revoke_blocked_sessions(self, organization: Organization, connection: OIDCConnection) -> None:
        """End the live session of every active member this organization's enforcement now refuses."""
        provider = oidc_identity_provider(connection.resolved_issuer)
        roster = await self._organizations.members.get_active_by_organization_with_users(organization.id)
        linked_ids = await self._links.linked_user_ids(
            provider=provider, user_ids=[identity.id for _, identity in roster]
        )
        blocked_ids = {
            identity.id
            for _, identity in roster
            if sso_blocks_membership(
                organization=organization,
                user=identity,
                linked_providers=frozenset({provider}) if identity.id in linked_ids else frozenset(),
            )
        }
        if blocked_ids:
            await self._sessions.delete_for_users(blocked_ids)

    async def blocks_membership(self, *, user: User, organization: Organization) -> bool:
        """Whether ``organization``'s own SSO enforcement refuses ``user`` from acting in it."""
        async with self._uow:
            linked = await linked_providers_for_organization(
                organization, user.id, connections=self._connections, links=self._links
            )
        return sso_blocks_membership(organization=organization, user=user, linked_providers=linked)

    async def blocks_any_membership(self, *, user: User) -> bool:
        """Whether some organization ``user`` actively belongs to requires sign-in through its own connection."""
        async with self._uow:
            return await sso_enforcement_blocks_any_membership(
                user, organizations=self._organizations, connections=self._connections, links=self._links
            )

    async def _store_first(
        self,
        *,
        organization_id: uuid.UUID,
        request: OIDCConnectionRequest,
        discovered: DiscoveredProvider,
        secret: _StoredSecret,
    ) -> OIDCConnection:
        """Store an organization's first connection, taking over the row a concurrent save inserted.

        NOTE: Two first saves race here, because each reads no connection before spending seconds in
        discovery. The loser takes the winner's row over rather than refusing a valid submission.
        """
        try:
            return await self._connections.add(
                organization_id=organization_id,
                issuer=request.issuer,
                resolved_issuer=discovered.resolved_issuer,
                client_id=request.client_id,
                encrypted_client_secret=secret.encrypted,
                client_secret_last4=secret.last4,
                display_name=request.display_name,
                discovery_document=discovered.document,
                discovered_at=discovered.discovered_at,
                default_role=request.default_role,
                enabled=request.enabled,
            )
        except OIDCConnectionConflictError:
            winner = await self._connections.get_for_organization(organization_id)
            if winner is None:
                raise
            self._overwrite(winner, request=request, discovered=discovered, secret=secret)
            return winner

    def _overwrite(
        self,
        connection: OIDCConnection,
        *,
        request: OIDCConnectionRequest,
        discovered: DiscoveredProvider,
        secret: _StoredSecret,
    ) -> None:
        """Move every submitted field onto an existing connection and stage it."""
        connection.issuer = request.issuer
        connection.resolved_issuer = discovered.resolved_issuer
        connection.client_id = request.client_id
        connection.encrypted_client_secret = secret.encrypted
        connection.client_secret_last4 = secret.last4
        connection.display_name = request.display_name
        connection.discovery_document = discovered.document
        connection.discovered_at = discovered.discovered_at
        connection.default_role = request.default_role
        connection.enabled = request.enabled
        self._connections.stage(connection)

    @staticmethod
    def _to_public(connection: OIDCConnection, domains: list[OrganizationDomain]) -> OIDCConnectionPublic:
        return OIDCConnectionPublic(
            id=connection.id,
            issuer=connection.issuer,
            resolved_issuer=connection.resolved_issuer,
            client_id=connection.client_id,
            client_secret_last4=connection.client_secret_last4,
            display_name=connection.display_name,
            default_role=connection.default_role,
            enabled=connection.enabled,
            discovered_at=connection.discovered_at,
            domains=[
                OIDCConnectionDomain(
                    domain=domain.domain,
                    proof_expires_at=None if domain.verified_at is None else domain.verified_at + DOMAIN_PROOF_TTL,
                    enabled=domain.enabled,
                )
                for domain in domains
            ],
        )


def _stored_metadata(connection: OIDCConnection) -> DiscoveredProvider:
    return DiscoveredProvider(
        resolved_issuer=connection.resolved_issuer,
        document=connection.discovery_document,
        discovered_at=connection.discovered_at,
    )


__all__ = ["OIDCConnectionService"]
