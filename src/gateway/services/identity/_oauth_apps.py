"""User-approved app credentials, delivered with S256 PKCE."""

import hashlib
import secrets
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from urllib.parse import urlencode

from apron_auth.pkce import generate_code_challenge

from gateway.auth.models import MCP_ACCESS_TOKEN_PREFIX, MCP_REFRESH_TOKEN_PREFIX, PERSONAL_ACCESS_TOKEN_PREFIX
from gateway.core.oauth_apps import APP_KEY_LIFETIMES, open_app_site, redirect_matches
from gateway.core.settings.identity import OAuthAppConfig
from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions.api_keys_exceptions import AppKeyOwnerInactiveError
from gateway.exceptions.identity_exceptions import OAuthAppGrantError, PersonalAccessTokenLimitReachedError
from gateway.exceptions.organizations_exceptions import NotAuthorizedError, WorkspaceNotFoundError
from gateway.log_config import logger
from gateway.models.identity import APP_CREDENTIAL_ACCESS_TOKEN, APP_CREDENTIAL_API_KEY, AppCredential
from gateway.models.tenancy import User
from gateway.ports.api_key_format_port import ApiKeyFormatPort
from gateway.repositories.identity.oauth_app_grant_repository import OAuthAppGrantRepository
from gateway.repositories.identity.oauth_dynamic_client_repository import OAuthDynamicClientRepository
from gateway.schemas.identity import AppAuthorizationRequest
from gateway.schemas.identity.oauth_apps import (
    AppAuthorizationParameters,
    AppDetails,
    AppTokenRequest,
    ClientRegistrationRequest,
    ClientRegistrationResponse,
)
from gateway.services.api_keys import ApiKeyService
from gateway.services.budgets import BudgetService
from gateway.services.identity._personal_access_tokens import PersonalAccessTokenService
from gateway.services.tenancy import OrganizationService, WorkspaceService
from gateway.types.key_scope import KeyScopeSet

# A self-registered client's ID, so one is told apart from a registered app's or an open app's origin.
_DYNAMIC_CLIENT_PREFIX = "mcp-"
# How long a registration nobody approves is kept.
_UNAUTHORIZED_CLIENT_LIFETIME = timedelta(days=1)
# How long an MCP client's sign-in may be approved for, renewals included. A self-registered client
# is unverified, so a link that asks for longer, or for no end, is not what the person gets.
_MCP_SIGN_IN_LIFETIMES = frozenset({"1-day", "7-days", "30-days", "90-days"})


@dataclass(frozen=True)
class _App:
    """An app an authorization code may be delivered to, and what its approval redeems to."""

    name: str
    scopes: KeyScopeSet
    verified: bool
    credential: AppCredential


@dataclass(frozen=True)
class IssuedCredential:
    """What a redeemed approval hands the app."""

    token: str
    scopes: KeyScopeSet
    expires_in: int | None
    manage_path: str
    refresh_token: str | None = None


def _code_hash(code: str) -> str:
    """How an authorization code is stored and looked up, so the code itself is never kept."""
    return hashlib.sha256(code.encode()).hexdigest()


class OAuthAppService:
    def __init__(
        self,
        uow: UnitOfWork,
        grants: OAuthAppGrantRepository,
        keys: ApiKeyService,
        key_format: ApiKeyFormatPort,
        organizations: OrganizationService,
        workspaces: WorkspaceService,
        budgets: BudgetService,
        access_tokens: PersonalAccessTokenService,
        clients: OAuthDynamicClientRepository,
        *,
        apps: Mapping[str, OAuthAppConfig],
        open_apps: bool = False,
        dynamic_clients: bool = False,
        spend_limits: bool = True,
        ui_link: Callable[[str], str],
        issuer: str,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self.uow, self.grants, self.keys = uow, grants, keys
        self.key_format = key_format
        self.organizations, self.workspaces, self.budgets = organizations, workspaces, budgets
        self.apps, self.open_apps, self.spend_limits = apps, open_apps, spend_limits
        self.access_tokens, self.clients, self.dynamic_clients = access_tokens, clients, dynamic_clients
        self.ui_link, self.issuer, self.clock = ui_link, issuer, clock

    async def resolve_app(self, client_id: str, redirect_uri: str) -> _App:
        """The app an authorization code may be delivered to at this callback.

        A registered client ID admits only its registered callbacks. With open apps on, any
        other client ID is admitted when it is its callback's origin, and the app is that site.
        A self-registered client admits the callbacks it registered, and redeems to an access token.
        """
        app = self.apps.get(client_id)
        if app is not None:
            if any(redirect_matches(uri, redirect_uri) for uri in app.redirect_uris):
                return _App(
                    app.name, KeyScopeSet.checked(app.scopes, is_service_key=False), True, APP_CREDENTIAL_API_KEY
                )
        elif self.dynamic_clients and client_id.startswith(_DYNAMIC_CLIENT_PREFIX):
            async with self.uow:
                client = await self.clients.get(client_id)
            if client is not None and any(redirect_matches(uri, redirect_uri) for uri in client.redirect_uris):
                return _App(client.client_name, KeyScopeSet(frozenset()), False, APP_CREDENTIAL_ACCESS_TOKEN)
        elif self.open_apps and (site := open_app_site(client_id, redirect_uri)) is not None:
            # Not validated: an admitted origin's host can be longer than a registered name may be.
            app = OAuthAppConfig.model_construct(name=site, redirect_uris=[redirect_uri])
            return _App(site, KeyScopeSet.checked(app.scopes, is_service_key=False), False, APP_CREDENTIAL_API_KEY)
        raise OAuthAppGrantError("invalid_request", "Unknown app or redirect URI")

    async def details(self, client_id: str, redirect_uri: str) -> AppDetails:
        """What the consent screen shows: the app, where it returns, and what it will be given."""
        app = await self.resolve_app(client_id, redirect_uri)
        return AppDetails(
            name=app.name,
            redirect_uri=redirect_uri,
            issuer=self.issuer,
            scopes=list(app.scopes),
            verified=app.verified,
            spend_limits=self.spend_limits and app.credential == APP_CREDENTIAL_API_KEY,
            credential=app.credential,
        )

    async def authorization_url(self, request: AppAuthorizationParameters) -> str:
        """Continue through the dashboard's existing sign-in and consent gate."""
        await self.resolve_app(request.client_id, request.redirect_uri)
        return self.ui_link("/oauth/authorize?" + urlencode(request.model_dump(exclude_none=True)))

    async def register_client(self, request: ClientRegistrationRequest) -> ClientRegistrationResponse:
        """Register a client that names itself, as RFC 7591 lets an MCP client do on first connect.

        Raises:
            OAuthAppGrantError: registration is off here.
        """
        if not self.dynamic_clients:
            raise OAuthAppGrantError("invalid_request", "Client registration is not enabled")
        now = self.clock()
        client_id = _DYNAMIC_CLIENT_PREFIX + secrets.token_urlsafe(24)
        name = request.client_name or "MCP client"
        async with self.uow:
            await self.clients.prune_unauthorized(now - _UNAUTHORIZED_CLIENT_LIFETIME)
            await self.clients.add(
                client_id=client_id, client_name=name, redirect_uris=request.redirect_uris, created_at=now
            )
        return ClientRegistrationResponse(
            client_id=client_id,
            client_id_issued_at=int(now.timestamp()),
            client_name=name,
            redirect_uris=request.redirect_uris,
        )

    def manage_url(self, path: str = "/keys") -> str:
        """The dashboard page where a person sees and revokes what apps were issued.

        Absolute, since the app links to it from its own origin; with no interface address
        configured, the gateway serves the dashboard itself.
        """
        link = self.ui_link(path)
        return link if "://" in link else self.issuer + link

    async def authorize(self, user: User, request: AppAuthorizationRequest) -> str:
        """Record explicit consent, for the scopes the screen showed that the registration still names.

        An app that redeems to an access token is approved for the person themselves, in the
        organization the screen showed, so no workspace is named and no scope narrows it.

        Raises:
            OAuthAppGrantError: the registration and the screen share no scope, a key's approval
                names no workspace, or an access token's names another organization than the
                person's or a lifetime an MCP sign-in may not have.
        """
        app = await self.resolve_app(request.client_id, request.redirect_uri)
        now = self.clock()
        lifetime = APP_KEY_LIFETIMES[request.expiration]
        code = secrets.token_urlsafe(32)
        if app.credential == APP_CREDENTIAL_ACCESS_TOKEN:
            if request.expiration not in _MCP_SIGN_IN_LIFETIMES:
                raise OAuthAppGrantError("invalid_request", "An MCP client is approved for at most 90 days")
            if request.organization_id is None or request.organization_id != user.active_organization_id:
                raise OAuthAppGrantError(
                    "invalid_request", "Your organization changed while you were approving the client"
                )
            async with self.uow:
                await self.organizations.lock_identity(user.id)
                await self.clients.mark_authorized(request.client_id, now)
                await self.grants.issue(
                    code_hash=_code_hash(code),
                    client_id=request.client_id,
                    redirect_uri=request.redirect_uri,
                    challenge=request.code_challenge,
                    user_id=user.id,
                    workspace_id=None,
                    scopes=KeyScopeSet(frozenset()),
                    expires_at=now + timedelta(minutes=2),
                    key_lifetime_sec=None if lifetime is None else int(lifetime.total_seconds()),
                    credential=APP_CREDENTIAL_ACCESS_TOKEN,
                    organization_id=request.organization_id,
                )
                await self.grants.prune_expired(now)
            return self._redirect(request, code)
        if request.workspace_id is None:
            raise OAuthAppGrantError("invalid_request", "Choose a workspace for the app's key")
        approved = app.scopes & request.scopes
        if not approved:
            raise OAuthAppGrantError("invalid_scope", "The app's access changed while you were approving it")
        if approved != app.scopes:
            logger.info(
                "OAuth app approval narrowed: client_id=%s dropped=%s",
                request.client_id,
                ",".join(scope.value for scope in app.scopes if scope not in approved),
            )
        async with self.uow:
            # Identity before workspace, matching redemption and organization deletion.
            await self.organizations.lock_identity(user.id)
            workspace = await self.workspaces.workspace_in_active_organization(
                user=user, workspace_id=request.workspace_id
            )
            await self.organizations.hold_workspace_write_access(
                user=user, organization_id=workspace.organization_id, workspace_id=workspace.id, revoking=False
            )
            await self.grants.issue(
                code_hash=_code_hash(code),
                client_id=request.client_id,
                redirect_uri=request.redirect_uri,
                challenge=request.code_challenge,
                user_id=user.id,
                workspace_id=workspace.id,
                scopes=approved,
                expires_at=now + timedelta(minutes=2),
                spend_limit=request.limit if self.spend_limits else None,
                key_lifetime_sec=None if lifetime is None else int(lifetime.total_seconds()),
            )
            await self.grants.prune_expired(now)
        return self._redirect(request, code)

    def _redirect(self, request: AppAuthorizationRequest, code: str) -> str:
        return request.redirect_uri + "?" + urlencode({"code": code, "state": request.state, "iss": self.issuer})

    async def exchange(self, request: AppTokenRequest) -> IssuedCredential:
        """Consume the approval and mint its credential atomically, under current membership and restrictions.

        A refresh token instead renews an MCP client's sign-in.

        Returns:
            The credential, with the scopes a key holds, which the token response names because they
            can be fewer than the app's registration.
        """
        if request.refresh_token is not None:
            return await self._renew(request.client_id, request.refresh_token)
        if request.redirect_uri is None or request.code is None or request.code_verifier is None:
            raise OAuthAppGrantError("invalid_request", "Missing authorization code parameters")
        try:
            app = await self.resolve_app(request.client_id, request.redirect_uri)
        except OAuthAppGrantError:
            if request.client_id in self.apps:
                logger.warning("OAuth app exchange refused: client_id=%s reason=redirect", request.client_id)
                raise OAuthAppGrantError("invalid_grant", "Invalid or expired authorization grant") from None
            # Any other client_id is whatever an unauthenticated caller sent, line breaks and Rich
            # markup included, so it never reaches the log.
            logger.warning("OAuth app exchange refused: reason=unknown_client")
            raise OAuthAppGrantError("invalid_client", "Unknown app") from None
        try:
            return await self._exchange(
                request, app, code=request.code, redirect_uri=request.redirect_uri, code_verifier=request.code_verifier
            )
        except (
            OAuthAppGrantError,
            NotAuthorizedError,
            WorkspaceNotFoundError,
            AppKeyOwnerInactiveError,
            PersonalAccessTokenLimitReachedError,
        ) as exc:
            # Refusals contain no request credentials. Unexpected failures must remain server errors.
            logger.warning("OAuth app exchange refused: client_id=%s reason=%s", request.client_id, str(exc))
            raise OAuthAppGrantError("invalid_grant", "Invalid or expired authorization grant") from None

    async def _renew(self, client_id: str, refresh_token: str) -> IssuedCredential:
        signed_in = None
        if self.dynamic_clients and client_id.startswith(_DYNAMIC_CLIENT_PREFIX):
            signed_in = await self.access_tokens.renew(client_id, refresh_token)
        if signed_in is None:
            # The client_id is whatever an unauthenticated caller sent, so it never reaches the log.
            logger.warning("OAuth app refresh refused")
            raise OAuthAppGrantError("invalid_grant", "Invalid or expired refresh token")
        return IssuedCredential(
            token=signed_in.access_token,
            scopes=KeyScopeSet(frozenset()),
            expires_in=signed_in.expires_in,
            manage_path="/account",
            refresh_token=signed_in.refresh_token,
        )

    async def _exchange(
        self, request: AppTokenRequest, app: _App, *, code: str, redirect_uri: str, code_verifier: str
    ) -> IssuedCredential:
        now = self.clock()
        async with self.uow:
            grant = await self.grants.find_bound(
                code_hash=_code_hash(code),
                client_id=request.client_id,
                redirect_uri=redirect_uri,
                challenge=generate_code_challenge(code_verifier),
                now=now,
            )
            # Null on a row built in memory, which an API key grant always was before access tokens.
            credential = grant.credential if grant is not None and grant.credential else APP_CREDENTIAL_API_KEY
            if grant is None or credential != app.credential:
                raise OAuthAppGrantError("invalid_grant", "Invalid or expired authorization grant")
            lifetime = None if grant.key_lifetime_sec is None else timedelta(seconds=grant.key_lifetime_sec)
            if credential == APP_CREDENTIAL_ACCESS_TOKEN:
                if lifetime is None or grant.organization_id is None:
                    raise OAuthAppGrantError("invalid_grant", "Invalid or expired authorization grant")
                await self.organizations.lock_identity(grant.user_id)
                if not await self.grants.consume(grant.code_hash, now):
                    raise OAuthAppGrantError("invalid_grant", "Authorization grant was already consumed")
                signed_in = await self.access_tokens.sign_in_client(
                    grant.user_id,
                    organization_id=grant.organization_id,
                    client_id=request.client_id,
                    name=f"{app.name} (MCP)",
                    lifetime=lifetime,
                )
                if signed_in is None:
                    raise AppKeyOwnerInactiveError
                return IssuedCredential(
                    token=signed_in.access_token,
                    scopes=KeyScopeSet(frozenset()),
                    expires_in=signed_in.expires_in,
                    manage_path="/account",
                    refresh_token=signed_in.refresh_token,
                )
            held = grant.scopes
            if not held or grant.workspace_id is None:
                # A grant written by a newer worker naming only scopes this one
                # does not know. A partial overlap narrows to the known ones.
                raise OAuthAppGrantError("invalid_grant", "Invalid or expired authorization grant")
            user = await self.organizations.hold_delegated_workspace_access(
                user_id=grant.user_id, workspace_id=grant.workspace_id
            )
            if not await self.grants.consume(grant.code_hash, now):
                raise OAuthAppGrantError("invalid_grant", "Authorization grant was already consumed")
            key, key_id = await self.keys.issue_app_key(
                user=user,
                workspace_id=grant.workspace_id,
                client_id=request.client_id,
                app_name=app.name,
                scopes=held,
                key_format=self.key_format,
                expires_at=None if lifetime is None else now + lifetime,
            )
            if grant.spend_limit is not None:
                await self.budgets.cap_app_key(
                    key_id=key_id,
                    workspace_id=grant.workspace_id,
                    name=f"{app.name} (OAuth)",
                    max_budget=grant.spend_limit,
                )
            return IssuedCredential(token=key, scopes=held, expires_in=None, manage_path="/keys")

    async def revoke(self, token: str) -> None:
        """Revoke only the presented API key, access token or sign-in; an unknown token is a successful no-op."""
        if token.startswith((PERSONAL_ACCESS_TOKEN_PREFIX, MCP_ACCESS_TOKEN_PREFIX, MCP_REFRESH_TOKEN_PREFIX)):
            await self.access_tokens.revoke_presented(token)
        else:
            await self.keys.revoke_key_by_token(token)
