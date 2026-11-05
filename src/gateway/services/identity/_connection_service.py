"""Manage an organization's own OpenID Connect sign-in connection."""

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from apron_auth import BrowserBoundAuthorization, StoredStateClient

from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions.identity_exceptions import (
    OIDCConnectionConflictError,
    OIDCConnectionNotFoundError,
    OIDCConnectionSecretRequiredError,
    OIDCDiscoveryError,
)
from gateway.exceptions.shared_exceptions import SecretBoxUnavailableTenancyError
from gateway.log_config import logger
from gateway.models.identity import OIDC_PROVIDER_PREFIX, OIDCConnection
from gateway.models.tenancy import DOMAIN_PROOF_TTL, Organization, OrganizationDomain, User
from gateway.repositories.identity import IdentityRepositories
from gateway.repositories.tenancy import OrganizationDomainRepository
from gateway.schemas.identity import OIDCConnectionDomain, OIDCConnectionPublic, OIDCConnectionRequest
from gateway.services.identity._connection_clients import client_for, pending_flow_provider
from gateway.services.identity._discovery import DISCOVERY_TTL, DiscoveredProvider, discover_connection_metadata
from gateway.services.identity._membership import organization_for_email_domain
from gateway.services.secret_box import encrypt_secret, secret_box_configured
from gateway.services.tenancy import OrganizationService


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
        """
        organization = await self._managed_organization(user)

        async with self._uow:
            connection = await self._connections.get_for_organization(organization.id)
            if connection is None:
                raise OIDCConnectionNotFoundError
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
