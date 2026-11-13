"""User-approved app credentials, delivered with S256 PKCE."""

import hashlib
import secrets
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from urllib.parse import urlencode

from apron_auth.pkce import generate_code_challenge

from gateway.core.oauth_apps import redirect_matches
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
from gateway.services.tenancy import OrganizationService, WorkspaceService


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
        *,
        apps: Mapping[str, OAuthAppConfig],
        ui_link: Callable[[str], str],
        issuer: str,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self.uow, self.grants, self.keys = uow, grants, keys
        self.key_format = key_format
        self.organizations, self.workspaces = organizations, workspaces
        self.apps, self.ui_link, self.issuer, self.clock = apps, ui_link, issuer, clock

    def require_registered_redirect(self, client_id: str, redirect_uri: str) -> None:
        """Only an explicitly registered callback may receive an authorization code."""
        app = self.apps.get(client_id)
        if app is None or not any(redirect_matches(uri, redirect_uri) for uri in app.redirect_uris):
            raise OAuthAppGrantError("invalid_request", "Unknown app or redirect URI")

    def _app_name(self, client_id: str) -> str:
        """The operator's display name for a client ``require_registered_redirect`` has admitted."""
        return self.apps[client_id].name

    def details(self, client_id: str, redirect_uri: str) -> AppDetails:
        self.require_registered_redirect(client_id, redirect_uri)
        return AppDetails(name=self._app_name(client_id), redirect_uri=redirect_uri, issuer=self.issuer)

    def authorization_url(self, request: AppAuthorizationParameters) -> str:
        """Continue through the dashboard's existing sign-in and consent gate."""
        self.require_registered_redirect(request.client_id, request.redirect_uri)
        return self.ui_link("/#/oauth/authorize?" + urlencode(request.model_dump()))

    async def authorize(self, user: User, request: AppAuthorizationRequest) -> str:
        """Record explicit consent without leaving an active key for an abandoned approval."""
        self.require_registered_redirect(request.client_id, request.redirect_uri)
        now = self.clock()
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
                expires_at=now + timedelta(minutes=2),
            )
            await self.grants.prune_expired(now)
        return request.redirect_uri + "?" + urlencode({"code": code, "state": request.state, "iss": self.issuer})

    async def exchange(self, request: AppTokenRequest) -> str:
        """Consume the approval and mint its key atomically, under current membership and restrictions."""
        if request.client_id not in self.apps:
            # An unregistered client_id is whatever an unauthenticated caller sent, line breaks and
            # Rich markup included, so it never reaches the log.
            logger.warning("OAuth app exchange refused: reason=unknown_client")
            raise OAuthAppGrantError("invalid_client", "Unknown app")
        try:
            self.require_registered_redirect(request.client_id, request.redirect_uri)
            return await self._exchange(request)
        except (OAuthAppGrantError, NotAuthorizedError, WorkspaceNotFoundError, AppKeyOwnerInactiveError) as exc:
            # Refusals contain no request credentials. Unexpected failures must remain server errors.
            logger.warning("OAuth app exchange refused: client_id=%s reason=%s", request.client_id, str(exc))
            raise OAuthAppGrantError("invalid_grant", "Invalid or expired authorization grant") from None

    async def _exchange(self, request: AppTokenRequest) -> str:
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
            user = await self.organizations.hold_delegated_workspace_access(
                user_id=grant.user_id, workspace_id=grant.workspace_id
            )
            if not await self.grants.consume(grant.code_hash, now):
                raise OAuthAppGrantError("invalid_grant", "Authorization grant was already consumed")
            return await self.keys.issue_app_key(
                user=user,
                workspace_id=grant.workspace_id,
                client_id=request.client_id,
                app_name=self._app_name(request.client_id),
                key_format=self.key_format,
            )

    async def revoke(self, token: str) -> None:
        """Revoke only the presented API key; an unknown token is a successful no-op."""
        await self.keys.revoke_key_by_token(token)
