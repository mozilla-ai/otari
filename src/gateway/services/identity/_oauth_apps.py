"""User-approved app credentials, delivered with S256 PKCE."""

import hashlib
import secrets
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from urllib.parse import urlencode

from apron_auth.pkce import generate_code_challenge

from gateway.core.oauth_apps import APP_KEY_LIFETIMES, open_app_site, redirect_matches
from gateway.core.settings.identity import OAuthAppConfig
from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions.api_keys_exceptions import AppKeyOwnerInactiveError
from gateway.exceptions.identity_exceptions import OAuthAppGrantError
from gateway.exceptions.organizations_exceptions import NotAuthorizedError, WorkspaceNotFoundError
from gateway.log_config import logger
from gateway.models.tenancy import User
from gateway.ports.api_key_format_port import ApiKeyFormatPort
from gateway.repositories.identity.oauth_app_grant_repository import OAuthAppGrantRepository
from gateway.schemas.identity import AppAuthorizationRequest
from gateway.schemas.identity.oauth_apps import AppAuthorizationParameters, AppDetails, AppTokenRequest
from gateway.services.api_keys import ApiKeyService
from gateway.services.budgets import BudgetService
from gateway.services.tenancy import OrganizationService, WorkspaceService
from gateway.types.key_scope import KeyScopeSet


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
        *,
        apps: Mapping[str, OAuthAppConfig],
        open_apps: bool = False,
        spend_limits: bool = True,
        ui_link: Callable[[str], str],
        issuer: str,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self.uow, self.grants, self.keys = uow, grants, keys
        self.key_format = key_format
        self.organizations, self.workspaces, self.budgets = organizations, workspaces, budgets
        self.apps, self.open_apps, self.spend_limits = apps, open_apps, spend_limits
        self.ui_link, self.issuer, self.clock = ui_link, issuer, clock

    def require_app(self, client_id: str, redirect_uri: str) -> OAuthAppConfig:
        """The app an authorization code may be delivered to at this callback.

        A registered client ID admits only its registered callbacks. With open apps on, any
        other client ID is admitted when it is its callback's origin, and the app is that site.
        """
        app = self.apps.get(client_id)
        if app is not None:
            if any(redirect_matches(uri, redirect_uri) for uri in app.redirect_uris):
                return app
        elif self.open_apps and (site := open_app_site(client_id, redirect_uri)) is not None:
            # Not validated: an admitted origin's host can be longer than a registered name may be.
            return OAuthAppConfig.model_construct(name=site, redirect_uris=[redirect_uri])
        raise OAuthAppGrantError("invalid_request", "Unknown app or redirect URI")

    def details(self, client_id: str, redirect_uri: str) -> AppDetails:
        """What the consent screen shows: the app, where it returns, and what its keys will hold."""
        app = self.require_app(client_id, redirect_uri)
        return AppDetails(
            name=app.name,
            redirect_uri=redirect_uri,
            issuer=self.issuer,
            scopes=app.scopes,
            verified=client_id in self.apps,
            spend_limits=self.spend_limits,
        )

    def authorization_url(self, request: AppAuthorizationParameters) -> str:
        """Continue through the dashboard's existing sign-in and consent gate."""
        self.require_app(request.client_id, request.redirect_uri)
        return self.ui_link("/oauth/authorize?" + urlencode(request.model_dump(exclude_none=True)))

    def manage_url(self) -> str:
        """The dashboard page where a person sees and revokes the keys apps were issued.

        Absolute, since the app links to it from its own origin; with no interface address
        configured, the gateway serves the dashboard itself.
        """
        link = self.ui_link("/keys")
        return link if "://" in link else self.issuer + link

    async def authorize(self, user: User, request: AppAuthorizationRequest) -> str:
        """Record explicit consent, for the scopes the screen showed that the registration still names.

        Raises:
            OAuthAppGrantError: the registration and the screen share no scope.
        """
        app = self.require_app(request.client_id, request.redirect_uri)
        registered = KeyScopeSet.checked(app.scopes, is_service_key=False)
        approved = registered & request.scopes
        if not approved:
            raise OAuthAppGrantError("invalid_scope", "The app's access changed while you were approving it")
        if approved != registered:
            logger.info(
                "OAuth app approval narrowed: client_id=%s dropped=%s",
                request.client_id,
                ",".join(scope.value for scope in registered if scope not in approved),
            )
        now = self.clock()
        lifetime = APP_KEY_LIFETIMES[request.expiration]
        code = secrets.token_urlsafe(32)
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
        return request.redirect_uri + "?" + urlencode({"code": code, "state": request.state, "iss": self.issuer})

    async def exchange(self, request: AppTokenRequest) -> tuple[str, KeyScopeSet]:
        """Consume the approval and mint its key atomically, under current membership and restrictions.

        Returns:
            The key and the scopes it holds, which the token response names because they can be
            fewer than the app's registration.
        """
        if request.client_id not in self.apps and not (
            self.open_apps and open_app_site(request.client_id, request.redirect_uri)
        ):
            # Any other client_id is whatever an unauthenticated caller sent, line breaks and Rich
            # markup included, so it never reaches the log. An open app's is a validated origin.
            logger.warning("OAuth app exchange refused: reason=unknown_client")
            raise OAuthAppGrantError("invalid_client", "Unknown app")
        try:
            app = self.require_app(request.client_id, request.redirect_uri)
            return await self._exchange(request, app.name)
        except (OAuthAppGrantError, NotAuthorizedError, WorkspaceNotFoundError, AppKeyOwnerInactiveError) as exc:
            # Refusals contain no request credentials. Unexpected failures must remain server errors.
            logger.warning("OAuth app exchange refused: client_id=%s reason=%s", request.client_id, str(exc))
            raise OAuthAppGrantError("invalid_grant", "Invalid or expired authorization grant") from None

    async def _exchange(self, request: AppTokenRequest, app_name: str) -> tuple[str, KeyScopeSet]:
        now = self.clock()
        async with self.uow:
            grant = await self.grants.find_bound(
                code_hash=_code_hash(request.code),
                client_id=request.client_id,
                redirect_uri=request.redirect_uri,
                challenge=generate_code_challenge(request.code_verifier),
                now=now,
            )
            if grant is None:
                raise OAuthAppGrantError("invalid_grant", "Invalid or expired authorization grant")
            held = grant.scopes
            if not held:
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
                app_name=app_name,
                scopes=held,
                key_format=self.key_format,
                expires_at=None if grant.key_lifetime_sec is None else now + timedelta(seconds=grant.key_lifetime_sec),
            )
            if grant.spend_limit is not None:
                await self.budgets.cap_app_key(
                    key_id=key_id,
                    workspace_id=grant.workspace_id,
                    name=f"{app_name} (OAuth)",
                    max_budget=grant.spend_limit,
                )
            return key, held

    async def revoke(self, token: str) -> None:
        """Revoke only the presented API key; an unknown token is a successful no-op."""
        await self.keys.revoke_key_by_token(token)
