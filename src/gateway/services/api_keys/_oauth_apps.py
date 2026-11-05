"""User-approved app credentials, delivered with S256 PKCE."""

import base64
import hashlib
import secrets
import uuid
from datetime import UTC, datetime, timedelta
from urllib.parse import urlencode

from gateway.auth.models import hash_key, key_suffix
from gateway.core.config import GatewayConfig
from gateway.core.settings.api_keys import parse_oauth_redirect
from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions import TenancyError, TenancyForbiddenError, TenancyValidationError
from gateway.models.tenancy import User
from gateway.ports.api_key_format_port import ApiKeyFormatPort
from gateway.repositories.api_keys.api_key_repository import ApiKeyRepository
from gateway.repositories.api_keys.oauth_app_grant_repository import OAuthAppGrantRepository
from gateway.schemas.api_keys import AppAuthorizationParameters, AppAuthorizationRequest, AppDetails, AppTokenRequest
from gateway.services.organizations import UserAccess
from gateway.services.tenancy import OrganizationService, WorkspaceService


def pkce_challenge(verifier: str) -> str:
    """RFC 7636 S256, without base64 padding."""
    return base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("ascii")).digest()).rstrip(b"=").decode()


def redirect_matches(registered: str, requested: str) -> bool:
    """RFC 8252 loopback callbacks may choose a free port; every other component is exact."""
    expected, actual = parse_oauth_redirect(registered), parse_oauth_redirect(requested)
    if expected is None or actual is None:
        return False
    if registered == requested:
        return True
    if expected.scheme != "http" or expected.hostname not in ("localhost", "127.0.0.1", "::1"):
        return False
    return actual.scheme == expected.scheme and actual.hostname == expected.hostname and actual.path == expected.path


class OAuthAppGrantError(Exception):
    """A refused app-token request, rendered using OAuth's error response contract."""

    def __init__(self, error: str, description: str) -> None:
        super().__init__(description)
        self.error, self.description = error, description


class OAuthAppService:
    def __init__(
        self,
        uow: UnitOfWork,
        config: GatewayConfig,
        keys: ApiKeyRepository,
        grants: OAuthAppGrantRepository,
        users: UserAccess,
        key_format: ApiKeyFormatPort,
        organizations: OrganizationService,
        workspaces: WorkspaceService,
    ) -> None:
        self.uow, self.config, self.keys, self.grants = uow, config, keys, grants
        self.users, self.key_format = users, key_format
        self.organizations, self.workspaces = organizations, workspaces

    def details(self, client_id: str, redirect_uri: str) -> AppDetails:
        """Only explicitly registered redirect URIs can receive a code."""
        app = self.config.oauth_apps.get(client_id)
        if app is None or not any(redirect_matches(uri, redirect_uri) for uri in app.redirect_uris):
            raise TenancyValidationError("Unknown app or redirect URI")
        return AppDetails(name=app.name, redirect_uri=redirect_uri)

    def authorization_url(self, request: AppAuthorizationParameters) -> str:
        """Send a valid authorization request into the dashboard's existing sign-in and consent gate."""
        try:
            self.details(request.client_id, request.redirect_uri)
        except TenancyValidationError:
            raise OAuthAppGrantError("invalid_request", "Unknown app or redirect URI") from None
        return self.config.ui_link("/#/oauth/authorize?" + urlencode(request.model_dump()))

    async def authorize(self, user: User, request: AppAuthorizationRequest) -> str:
        """Record explicit consent. An abandoned approval never leaves an active API key."""
        self.details(request.client_id, request.redirect_uri)
        now = datetime.now(UTC)
        code = secrets.token_urlsafe(32)
        async with self.uow:
            await self.grants.prune_expired(now)
        async with self.uow:
            # The grant's identity FK also locks this row. Take it before the
            # workspace, matching redemption and organization deletion.
            await self.organizations.lock_identity(user.id)
            workspace = await self.workspaces.workspace_in_active_organization(
                user=user, workspace_id=request.workspace_id
            )
            await self.organizations.hold_workspace_write_access(
                user=user, organization_id=workspace.organization_id, workspace_id=workspace.id, revoking=False
            )
            await self.grants.issue(
                code_hash=hashlib.sha256(code.encode()).hexdigest(),
                client_id=request.client_id,
                redirect_uri=request.redirect_uri,
                challenge=request.code_challenge,
                user_id=user.id,
                workspace_id=workspace.id,
                expires_at=now + timedelta(minutes=2),
            )
        return request.redirect_uri + "?" + urlencode({"code": code, "state": request.state})

    async def exchange(self, request: AppTokenRequest) -> str:
        """Consume the approval and mint its key atomically, with current membership and restrictions."""
        if request.client_id not in self.config.oauth_apps:
            raise OAuthAppGrantError("invalid_client", "Unknown app")
        try:
            return await self._exchange(request)
        except TenancyError:
            raise OAuthAppGrantError("invalid_grant", "Invalid or expired authorization grant") from None

    async def _exchange(self, request: AppTokenRequest) -> str:
        app = self.details(request.client_id, request.redirect_uri)
        async with self.uow:
            grant = await self.grants.find_bound(
                code_hash=hashlib.sha256(request.code.encode()).hexdigest(),
                client_id=request.client_id,
                redirect_uri=request.redirect_uri,
                challenge=pkce_challenge(request.code_verifier),
                now=datetime.now(UTC),
            )
            if grant is None:
                raise TenancyValidationError("Invalid or expired authorization grant")
            user = await self.organizations.hold_delegated_workspace_access(
                user_id=grant.user_id, workspace_id=grant.workspace_id
            )
            if not await self.users.ensure_app_key_owner(
                str(user.id), user.full_name or user.email or "OAuth app user"
            ):
                raise TenancyForbiddenError("Your spend identity has been deactivated")
            if not await self.grants.consume(grant.code_hash, datetime.now(UTC)):
                raise TenancyValidationError("Invalid or expired authorization grant")
            key = self.key_format.mint()
            await self.keys.issue_app_key(
                key_id=str(uuid.uuid4()),
                key_hash=hash_key(key),
                prefix=self.key_format.fingerprint(key),
                suffix=key_suffix(key),
                name=f"{app.name} (OAuth)",
                user_id=str(user.id),
                workspace_id=grant.workspace_id,
                client_id=request.client_id,
            )
        return key
