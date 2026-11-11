"""Manage an organization's own OpenID Connect sign-in connection."""

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from apron_auth import BrowserBoundAuthorization, StoredStateClient

from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions.identity_exceptions import (
    OIDCConnectionConflictError,
    OIDCConnectionDisabledError,
    OIDCConnectionEnforcedError,
    OIDCConnectionNotFoundError,
    OIDCConnectionSecretRequiredError,
    OIDCConnectionUnprovenError,
    OIDCDiscoveryError,
    SsoBreakGlassRequiredError,
    SsoBreakGlassRouteRequiredError,
    SsoNotAddressableError,
)
from gateway.exceptions.shared_exceptions import SecretBoxUnavailableTenancyError
from gateway.log_config import logger
from gateway.models.identity import (
    MAX_OIDC_ISSUER_LENGTH,
    OIDC_PROVIDER_PREFIX,
    OIDCConnection,
    oidc_identity_provider,
)
from gateway.models.tenancy import DOMAIN_PROOF_TTL, OWNER_ROLE, Organization, OrganizationMember, User
from gateway.repositories.identity import IdentityRepositories, OAuthPendingStateRepository
from gateway.schemas.identity import (
    OIDCConnectionDomain,
    OIDCConnectionPublic,
    OIDCConnectionRequest,
    OrganizationSsoEnforcementPublic,
    OrganizationSsoPreflightMemberPublic,
    OrganizationSsoPreflightPublic,
    OrganizationSsoPublic,
)
from gateway.services.identity._connection_clients import client_for, pending_flow_provider
from gateway.services.identity._discovery import DISCOVERY_TTL, DiscoveredProvider, discover_connection_metadata
from gateway.services.identity._enforcement import (
    has_link_to_organization,
    has_route_without,
    sso_enforcement_refuses_every_membership,
)
from gateway.services.identity._membership import organization_for_email_domain
from gateway.services.organizations import ClaimedDomain, ClaimedDomains
from gateway.services.secret_box import encrypt_secret, secret_box_configured
from gateway.services.tenancy import OrganizationService, sso_blocks_membership, sso_would_block_membership

# The preflight count is exact; the member list is a sample, so a large organization's
# response stays bounded.
_PREFLIGHT_SAMPLE_LIMIT = 50


@dataclass(frozen=True)
class _StoredSecret:
    """An encrypted client secret and the last four characters served in its place."""

    encrypted: str
    last4: str


def _secret_to_store(request: OIDCConnectionRequest, stored: OIDCConnection | None) -> _StoredSecret:
    """The client secret to store: the submitted one, encrypted, or the stored one kept as it is.

    The stored secret proves this deployment is one client at one issuer, so it is kept only where
    the submission still names both. Carrying it to an issuer or a client ID the caller just changed
    would hand it to whoever controls the new one.

    NOTE: Decided here, against the row being written over, rather than against an earlier read of
    it. A submission that leaves the secret out therefore costs one discovery before its refusal.

    Raises:
        OIDCConnectionSecretRequiredError: Nothing was submitted and no stored secret may stand in.

    """
    if request.client_secret is not None:
        return _StoredSecret(encrypted=encrypt_secret(request.client_secret), last4=request.client_secret[-4:])
    if stored is None or stored.issuer != request.issuer or stored.client_id != request.client_id:
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
        domains: ClaimedDomains,
        sign_in_states: OAuthPendingStateRepository,
        allow_private_hosts: bool,
        passkey_rp_id: str | None,
        offered_providers: frozenset[str],
        sso_redirect_uri: str | None,
    ) -> None:
        self._uow = uow
        self._repositories = repositories
        self._connections = repositories.connections
        self._links = repositories.links
        self._passkeys = repositories.passkeys
        self._pending_states = repositories.pending_states
        # NOTE: On the request's session rather than this service's Unit of Work. The library claims
        # the pending state during an exchange this service does not wrap, by which time a block
        # opened here has closed, and a repository on a closed block refuses the read.
        self._sign_in_states = sign_in_states
        self._sessions = repositories.sessions
        self._users = repositories.users
        self._organizations = organizations
        self._domains = domains
        self._allow_private_hosts = allow_private_hosts
        self._passkey_rp_id = passkey_rp_id
        self._offered_providers = offered_providers
        # This is None where the deployment has no public address of its own.
        self._sso_redirect_uri = sso_redirect_uri

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
        return await self._connection_for(await self._managed_organization(user))

    async def settings_for_user(self, *, user: User) -> OrganizationSsoPublic:
        """Return the caller's organization's sign-in connection settings. Owners and admins only."""
        organization = await self._managed_organization(user)
        return self._settings(organization, await self._connection_for(organization))

    def _settings(self, organization: Organization, connection: OIDCConnectionPublic | None) -> OrganizationSsoPublic:
        """Assemble the settings answer.

        NOTE: Enforcement is reported as of the organization the request resolved. Only
        ``set_sso_enforcement_for_user`` writes it, and every decision taken on it is re-read
        under ``lock_settings``.
        """
        return OrganizationSsoPublic(
            redirect_uri=self._sso_redirect_uri,
            connection=connection,
            enforcement=self._enforcement_of(organization),
        )

    async def _connection_for(self, organization: Organization) -> OIDCConnectionPublic | None:
        async with self._uow:
            connection = await self._connections.get_for_organization(organization.id)
            if connection is None:
                return None
            domains = await self._domains.for_organization(organization.id)
            return self._to_public(connection, domains)

    async def save_for_user(self, *, user: User, request: OIDCConnectionRequest) -> OrganizationSsoPublic:
        """Discover the submitted issuer and store the caller's organization's connection. Owners and admins only.

        Answers with the organization's whole settings, as every method that changes the connection
        does, so a caller needs no second read to show the result.

        Discovery runs before any write, and only a successful discovery opens the transaction that
        stores it, so a refused issuer leaves an existing connection, its document, and its secret
        exactly as they were. Whether the connection admits sign-in is not submitted here, so this
        can neither pause nor resume one. A new connection is stored admitting sign-in.

        NOTE: Where the organization requires sign-in through this connection, storing it also
        revokes the live session of every active member the stored issuer leaves with no link.

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

        await self._uow.release()
        discovered = await discover_connection_metadata(request.issuer, allow_private_hosts=self._allow_private_hosts)
        if len(discovered.resolved_issuer) > MAX_OIDC_ISSUER_LENGTH:
            # Every sign-in names a provider derived from the stored issuer, so one too long to name
            # would refuse the whole organization rather than this request.
            raise OIDCDiscoveryError

        async with self._uow:
            # Serialized against the enforcement switch on the organization row. This write reads
            # that switch and the issuer it is storing together, to sign out whoever enforcement
            # now refuses. An enforcement change interleaved with it would answer that from a
            # different pair.
            await self._organizations.organizations.lock_settings(organization.id)
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
            domains = await self._domains.for_organization(organization.id)
            await self._revoke_blocked_sessions(organization, connection)
            stored = self._to_public(connection, domains)

        return self._settings(organization, stored)

    async def set_enabled_for_user(self, *, user: User, enabled: bool) -> OrganizationSsoPublic:
        """Pause or resume sign-in through the caller's organization's connection. Owners and admins only.

        Needs no credential and reaches no provider, so a connection can be paused while its
        provider is unreachable, which is when an administrator is most likely to want to. The
        only way to change the setting, which is what keeps storing a connection from moving it.

        Raises:
            OIDCConnectionNotFoundError: The organization has not configured a connection yet.
            OIDCConnectionDisabledError: The organization requires sign-in through this connection,
                which pausing it would leave admitting nobody.
        """
        organization = await self._managed_organization(user)

        async with self._uow:
            # Serialized with the enforcement switch, which must not be turned on while this is
            # turning sign-in off.
            await self._organizations.organizations.lock_settings(organization.id)
            connection = await self._connections.get_for_organization(organization.id)
            if connection is None:
                raise OIDCConnectionNotFoundError
            if not enabled and await self._organizations.organizations.is_sso_enforced(organization.id):
                raise OIDCConnectionDisabledError
            connection.enabled = enabled
            self._connections.stage(connection)
            domains = await self._domains.for_organization(organization.id)
            stored = self._to_public(connection, domains)

        return self._settings(organization, stored)

    async def test_for_user(self, *, user: User) -> OrganizationSsoPublic:
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
            if current.issuer != issuer:
                # The row now names an issuer this document did not come from, and sign-in finds a
                # connection by its resolved issuer and dials the endpoints in its document.
                raise OIDCConnectionConflictError
            current.resolved_issuer = discovered.resolved_issuer
            current.discovery_document = discovered.document
            current.discovered_at = discovered.discovered_at
            self._connections.stage(current)
            domains = await self._domains.for_organization(organization.id)
            stored = self._to_public(current, domains)

        return self._settings(organization, stored)

    async def delete_for_user(self, *, user: User) -> None:
        """Remove the caller's organization's connection. Owners and admins only.

        Raises:
            OIDCConnectionNotFoundError: The organization has not configured a connection yet.
            OIDCConnectionEnforcedError: The organization still requires sign-in through this
                connection.
        """
        organization = await self._managed_organization(user)

        async with self._uow:
            # Every path deciding on the enforcement switch and the connection together serializes
            # on this row. Without it, a delete that read the switch off and an enable that still
            # saw the connection both commit, and the organization then requires a connection that
            # does not exist.
            await self._organizations.organizations.lock_settings(organization.id)
            connection = await self._connections.get_for_organization(organization.id)
            if connection is None:
                raise OIDCConnectionNotFoundError
            if await self._organizations.organizations.is_sso_enforced(organization.id):
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
            # while discovery was out. A row now naming another issuer keeps what it has: this
            # document did not come from it, and a sign-in is in flight on the metadata returned
            # below rather than on what is stored.
            current = await self._connections.get_by_id(connection_id)
            if current is not None and current.issuer == issuer:
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
            provider = await pending_flow_provider(self._pending_states, state)
            if provider is None or not provider.startswith(OIDC_PROVIDER_PREFIX):
                return None
            try:
                connection_id = uuid.UUID(provider.removeprefix(OIDC_PROVIDER_PREFIX))
            except ValueError:
                return None
            connection = await self._connections.get_by_id(connection_id)
        return connection if connection is not None and connection.enabled else None

    async def start_sign_in(
        self, connection: OIDCConnection, metadata: DiscoveredProvider
    ) -> BrowserBoundAuthorization:
        """Mint an authorization URL for ``connection``'s own provider, and stage the pending state.

        Raises:
            SsoNotAddressableError: This deployment has no address a provider may send a browser
                back to.
        """
        redirect_uri = self._registered_redirect_uri()
        async with self._uow:
            client = client_for(
                connection,
                metadata,
                pending=self._sign_in_states,
                redirect_uri=redirect_uri,
                allow_private_hosts=self._allow_private_hosts,
            )
            return await client.get_authorization_url(redirect_uri=redirect_uri)

    async def client_for_sign_in(self, connection: OIDCConnection, metadata: DiscoveredProvider) -> StoredStateClient:
        """Build the client ``connection``'s own provider signs in through.

        Raises:
            SsoNotAddressableError: This deployment has no address a provider may send a browser
                back to.
        """
        async with self._uow:
            return client_for(
                connection,
                metadata,
                pending=self._sign_in_states,
                redirect_uri=self._registered_redirect_uri(),
                allow_private_hosts=self._allow_private_hosts,
            )

    def _registered_redirect_uri(self) -> str:
        if self._sso_redirect_uri is None:
            raise SsoNotAddressableError
        return self._sso_redirect_uri

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

    @staticmethod
    def _enforcement_of(organization: Organization) -> OrganizationSsoEnforcementPublic:
        return OrganizationSsoEnforcementPublic(
            required=organization.sso_required,
            break_glass_user_id=organization.sso_break_glass_user_id,
            break_glass_waived=organization.sso_break_glass_waived,
        )

    async def preflight_sso_enforcement_for_user(self, *, user: User) -> OrganizationSsoPreflightPublic:
        """Report who the caller's organization's sign-in enforcement would affect. Owners and admins only.

        An active member is affected when enforcement would refuse them: they would get a new
        account the first time they use the connection, and any API key on their current account
        would stay there rather than moving.

        NOTE: Counted by the same rule that revokes sessions and admits a break-glass owner, so what
        this reports and what a save then does cannot describe different people. Holding no link is
        not that rule: it also catches the deployment's operator and the named exception, neither of
        whom loses anything.

        Raises:
            OIDCConnectionNotFoundError: The organization has not configured a connection yet.
        """
        organization = await self._managed_organization(user)

        async with self._uow:
            connection = await self._connections.get_for_organization(organization.id)
            if connection is None:
                raise OIDCConnectionNotFoundError
            provider = oidc_identity_provider(connection.resolved_issuer)
            total = await self._organizations.members.count_active(organization.id)
            refused = await self.members_enforcement_would_refuse(organization, connection)
            sample = refused[:_PREFLIGHT_SAMPLE_LIMIT]
            candidates = await self._break_glass_candidates(organization, provider)

        affected_count = len(refused)
        return OrganizationSsoPreflightPublic(
            total_members=total,
            break_glass_candidates=candidates,
            affected_count=affected_count,
            affected_members=[
                OrganizationSsoPreflightMemberPublic(
                    user_id=member.user_id, email=member_user.email, full_name=member_user.full_name
                )
                for member, member_user in sample
            ],
            affected_members_truncated=affected_count > len(sample),
        )

    async def _break_glass_candidates(
        self, organization: Organization, provider: str
    ) -> list[OrganizationSsoPreflightMemberPublic]:
        """The owners who may be named to keep a route in that this connection does not provide.

        Leaves out the deployment's own operator, who enforcement never refuses and so needs no
        exception, and any owner whose only way in is the connection itself.

        NOTE: The first exclusion is scoped to a deployment operator, not to whoever owns the
        organization, so an organization on a deployment somebody else operates keeps its own
        owners as candidates.
        """
        memberships = await self._organizations.members.get_active_by_organization_with_users(organization.id)
        operator_ids = await self._organizations.deployment_operator_ids([account for _, account in memberships])
        candidates = []
        for member, account in memberships:
            if member.role != OWNER_ROLE or account.id in operator_ids:
                continue
            if not await has_route_without(
                account,
                provider=provider,
                passkey_rp_id=self._passkey_rp_id,
                offered_providers=self._offered_providers,
                repositories=self._repositories,
            ):
                continue
            candidates.append(
                OrganizationSsoPreflightMemberPublic(
                    user_id=account.id, email=account.email, full_name=account.full_name
                )
            )
        return candidates

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
        currently refuses. Turning it off asks only that a named account exists, so an organization
        can always undo it, even with no connection configured and even once the named owner has
        been removed.

        Raises:
            OIDCConnectionNotFoundError: ``required`` is set, and the organization has not
                configured a connection yet.
            OIDCConnectionUnprovenError: ``required`` is set before anybody has signed in through
                the connection.
            SsoBreakGlassRequiredError: An account is named that this deployment does not hold,
                ``required`` is set naming one that is not one of the organization's own active
                owners, or ``required`` is set naming none and waiving none.
            SsoBreakGlassRouteRequiredError: ``required`` is set naming an account that could not
                sign in while the connection is unavailable.
        """
        organization = await self._managed_organization(user)

        async with self._uow:
            await self._organizations.organizations.lock_settings(organization.id)
            connection: OIDCConnection | None = None
            if break_glass_user_id is not None:
                await self._require_break_glass_account(break_glass_user_id)
                if required:
                    await self._require_break_glass_owner(organization, break_glass_user_id)
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
        if not connection.enabled:
            raise OIDCConnectionDisabledError
        if break_glass_user_id is None and not break_glass_waived:
            raise SsoBreakGlassRequiredError
        if break_glass_user_id is not None:
            await self._require_a_route_without_the_connection(break_glass_user_id, connection)
        return connection

    async def _require_a_route_without_the_connection(self, user_id: uuid.UUID, connection: OIDCConnection) -> None:
        """Refuse an exempted owner who could not sign in while this connection is unavailable.

        Raises:
            SsoBreakGlassRouteRequiredError: The account holds no password, no passkey of this
                deployment's and no link at another provider.
        """
        account = await self._users.get(user_id)
        if account is None or not await has_route_without(
            account,
            provider=oidc_identity_provider(connection.resolved_issuer),
            passkey_rp_id=self._passkey_rp_id,
            offered_providers=self._offered_providers,
            repositories=self._repositories,
        ):
            raise SsoBreakGlassRouteRequiredError

    async def _require_break_glass_account(self, user_id: uuid.UUID) -> None:
        """Refuse a break-glass account this deployment does not hold.

        Held whether or not enforcement is on, because the column is a foreign key and an id naming
        nobody would otherwise reach the database as an integrity error instead of a refusal. What
        the named account must be to act on the exception is the owner rule, which is held only
        while enforcement is on.

        Raises:
            SsoBreakGlassRequiredError: This deployment holds no such account.
        """
        if await self._users.get(user_id) is None:
            raise SsoBreakGlassRequiredError

    async def _require_break_glass_owner(self, organization: Organization, user_id: uuid.UUID) -> None:
        """Refuse a break-glass account that is not one of this organization's own active owners.

        Held while enforcement is on, which is the only time the exception grants anything. An owner
        demoted since they were named must not stand in the way of turning enforcement off.

        Raises:
            SsoBreakGlassRequiredError: The account is not an active owner here.
        """
        owner = await self._organizations.members.get_active_by_organization_and_user(organization.id, user_id)
        if owner is None or owner.role != OWNER_ROLE:
            raise SsoBreakGlassRequiredError

    async def members_enforcement_would_refuse(
        self, organization: Organization, connection: OIDCConnection
    ) -> list[tuple[OrganizationMember, User]]:
        """Every active membership this organization's enforcement would refuse, with its account.

        The one place the rule is applied to a whole organization, so what a preflight reports and
        what a save then revokes cannot describe different people. Answers the rule without the
        switch, because a preflight has to report who would lose access before it is on; a caller
        acting on the answer applies the switch itself.

        Reads in bulk: one query for the links and one for the deployment's own authority, rather
        than either per member.
        """
        memberships = await self._organizations.members.get_active_by_organization_with_users(organization.id)
        if not memberships:
            return []
        provider = oidc_identity_provider(connection.resolved_issuer)
        accounts = [account for _, account in memberships]
        linked_ids = await self._links.linked_user_ids(provider=provider, user_ids=[a.id for a in accounts])
        operator_ids = await self._organizations.deployment_operator_ids(accounts)
        return [
            (membership, account)
            for membership, account in memberships
            if sso_would_block_membership(
                organization=organization,
                membership=membership,
                is_linked=account.id in linked_ids,
                is_deployment_operator=account.id in operator_ids,
            )
        ]

    async def revoke_refused_sessions(self, *, organization: Organization) -> None:
        """End the live session of every active member this organization's enforcement now refuses.

        For a caller that has just changed who the enforcement refuses without touching the
        connection or the switch, such as a membership whose role no longer earns the break-glass
        exception. No authority check runs: the caller proved its own, and this grants nothing.
        """
        async with self._uow:
            connection = await self._connections.get_for_organization(organization.id)
            if connection is None:
                return
            await self._revoke_blocked_sessions(organization, connection)

    async def _revoke_blocked_sessions(self, organization: Organization, connection: OIDCConnection) -> None:
        """End the live session of every active member this organization's enforcement now refuses.

        The switch is read from the row rather than taken off ``organization``, whose attribute
        holds whatever was true when the caller resolved it.
        """
        if not await self._organizations.organizations.is_sso_enforced(organization.id):
            return
        refused = await self.members_enforcement_would_refuse(organization, connection)
        if refused:
            await self._sessions.delete_for_users({account.id for _, account in refused})

    async def blocks_membership(self, *, user: User, organization: Organization) -> bool:
        """Whether ``organization``'s own SSO enforcement refuses ``user`` from acting in it.

        An organization requiring nothing refuses nobody, which is answered off the row already in
        hand rather than by reading its connection and the account's links.
        """
        if not organization.sso_required:
            return False
        async with self._uow:
            membership = await self._organizations.members.get_active_by_organization_and_user(organization.id, user.id)
            if membership is None:
                return False
            linked = await has_link_to_organization(
                organization, user.id, connections=self._connections, links=self._links
            )
            is_operator = await self._organizations.is_deployment_operator(user)
        return sso_blocks_membership(
            organization=organization,
            membership=membership,
            is_linked=linked,
            is_deployment_operator=is_operator,
        )

    async def refuses_every_membership(self, *, user: User) -> bool:
        """Whether some organization ``user`` actively belongs to requires sign-in through its own connection."""
        async with self._uow:
            return await sso_enforcement_refuses_every_membership(
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
        self._connections.stage(connection)

    @staticmethod
    def _to_public(connection: OIDCConnection, domains: list[ClaimedDomain]) -> OIDCConnectionPublic:
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
