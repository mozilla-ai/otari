"""Connections: OAuth grants an application's users give this deployment for third-party apps.

The app-level half of what Octonous's ``OAuthService`` does, brought to otari
so an application built on the gateway can let *its* users connect Slack,
GitHub or Google, and so a gateway-run tool, an MCP server or an overlay can
then act on those accounts with a credential the gateway holds. The user is
whoever the application says it is: the ``user`` string on its requests,
scoped to the workspace its API key belongs to (:class:`EndUser`). The protocol
mechanics (authorization URL, PKCE, the code exchange, refresh, revocation,
identity fetch) come from apron-auth and its per-provider presets; what lives
here is what a deployment owns: which apps are configured, where the browser
is sent back to, the pending state between the two halves of a flow, and the
encrypted rows the tokens end up in.

Security posture, carried over from Octonous's ``OAUTH_FLOW.md``:

- PKCE on by default (the presets decide per provider), with the verifier
  stored encrypted alongside the state and never sent to the browser.
- The callback is hosted here and trusts only the ``state``: it names the end
  user, the provider and where to send the browser afterwards, all recorded
  when the application (not the browser) started the flow. It is consumed
  atomically so a code can be exchanged once.
- Tokens are Fernet-encrypted with ``OTARI_SECRET_KEY`` at rest and never
  serialized; :meth:`ConnectedAccountService.access_token` is the one way out
  and is for in-process callers.
- The redirect URI is derived from ``public_base_url`` and never taken from a
  request, so a browser cannot choose where a provider sends a code.
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any
from urllib.parse import urlencode, urlsplit

from apron_auth import OAuthClient
from apron_auth.errors import OAuthError
from apron_auth.models import OAuthPendingState, ProviderConfig, TokenSet
from apron_auth.protocols import RevocationHandler
from apron_auth.providers import atlassian as apron_atlassian
from apron_auth.providers import github as apron_github
from apron_auth.providers import google as apron_google
from apron_auth.providers import hubspot as apron_hubspot
from apron_auth.providers import linear as apron_linear
from apron_auth.providers import microsoft as apron_microsoft
from apron_auth.providers import notion as apron_notion
from apron_auth.providers import salesforce as apron_salesforce
from apron_auth.providers import slack as apron_slack
from apron_auth.providers import typeform as apron_typeform
from pydantic import BaseModel, Field
from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.config import CONNECTED_APP_PROVIDERS, GatewayConfig
from gateway.log_config import logger
from gateway.models.entities import ConnectedAccount, ConnectedAccountOAuthState, EndUser
from gateway.services.secret_box import (
    SecretBoxUnavailableError,
    SecretDecryptionError,
    decrypt_secret,
    encrypt_secret,
)
from gateway.services.tenancy.errors import (
    ConnectedAccountExchangeError,
    ConnectedAccountLimitReachedError,
    ConnectedAccountNotFoundError,
    ConnectedAccountReturnUrlError,
    ConnectedAccountStateInvalidError,
    ConnectedAppNotConfiguredError,
    SecretBoxUnavailableTenancyError,
)

#: A pending flow is good for this long. OAuth guidance says under ten minutes;
#: Octonous used exactly ten.
OAUTH_STATE_TTL = timedelta(minutes=10)
#: Refresh a token this close to expiry rather than hand out one about to die.
REFRESH_LEEWAY = timedelta(seconds=60)
#: Ceiling on accounts per end user, a sanity bound rather than a product limit.
MAX_CONNECTED_ACCOUNTS_PER_USER = 50
#: The ``user`` field on a request, and the external id here, share this bound.
MAX_EXTERNAL_USER_ID = 255

_PRESETS: dict[str, Any] = {
    "atlassian": apron_atlassian.preset,
    "github": apron_github.preset,
    "google": apron_google.preset,
    "hubspot": apron_hubspot.preset,
    "linear": apron_linear.preset,
    "microsoft": apron_microsoft.preset,
    "notion": apron_notion.preset,
    "salesforce": apron_salesforce.preset,
    "slack": apron_slack.preset,
    "typeform": apron_typeform.preset,
}
assert set(_PRESETS) == set(CONNECTED_APP_PROVIDERS), "CONNECTED_APP_PROVIDERS and _PRESETS must name the same apps"

_PROVIDER_LABELS = {
    "atlassian": "Atlassian",
    "github": "GitHub",
    "google": "Google",
    "hubspot": "HubSpot",
    "linear": "Linear",
    "microsoft": "Microsoft",
    "notion": "Notion",
    "salesforce": "Salesforce",
    "slack": "Slack",
    "typeform": "Typeform",
}


# ---------------------------------------------------------------------------
# Public shapes
# ---------------------------------------------------------------------------


class ScopePublic(BaseModel):
    scope: str
    label: str
    description: str
    access_type: str
    required: bool


class ConnectedAppPublic(BaseModel):
    """One app an end user may connect."""

    provider: str
    label: str
    scopes: list[str] = Field(description="The scopes this deployment asks for by default.")
    scope_details: list[ScopePublic] = Field(description="Labels and descriptions for the scopes the preset documents.")
    connected_accounts: int = Field(description="How many accounts the named user has connected for this app.")


class ConnectedAccountPublic(BaseModel):
    """A connected account without its tokens."""

    id: uuid.UUID
    user: str = Field(description="The application's id for the user who connected it.")
    provider: str
    account_identifier: str | None
    account_label: str | None = Field(description="What the provider calls the account: a name, an email, a workspace.")
    label: str | None = Field(description="The user's own label, when set.")
    scopes: list[str]
    expires_at: datetime | None
    has_refresh_token: bool
    created_at: datetime
    updated_at: datetime


class ConnectedAccountsPublic(BaseModel):
    data: list[ConnectedAccountPublic]
    count: int


class AuthorizeRequest(BaseModel):
    user: str = Field(min_length=1, max_length=MAX_EXTERNAL_USER_ID, description="Your application's id for the user.")
    scopes: list[str] | None = Field(
        default=None,
        max_length=64,
        description="Scopes to ask for instead of the deployment's defaults for this app.",
    )
    return_url: str | None = Field(
        default=None,
        max_length=2048,
        description=(
            "Where to send the browser once the account is connected (or the attempt failed). "
            "https, or http on localhost. Otari appends connection=ok&provider=...&connection_id=..., "
            "or connection=error&provider=...&reason=.... Defaults to a page on this deployment."
        ),
    )


class AuthorizePublic(BaseModel):
    authorization_url: str = Field(description="Send the user's browser here; Otari handles the rest of the flow.")
    expires_at: datetime = Field(description="When the link stops working; start again after that.")


class ConnectedAccountUpdate(BaseModel):
    label: Annotated[str | None, Field(max_length=64)] = None


class AccessToken(BaseModel):
    """A live credential for in-process consumers; never leaves the gateway."""

    provider: str
    token: str
    token_type: str
    expires_at: datetime | None
    extra: dict[str, str] = Field(
        default_factory=dict, description="Secondary tokens, e.g. Slack's user token under 'user'."
    )


# ---------------------------------------------------------------------------
# Pieces
# ---------------------------------------------------------------------------


def redirect_uri(config: GatewayConfig, provider: str) -> str:
    """Where the provider sends the browser back to; derived, never supplied.

    A plain path with no fragment, which ``gateway.main`` turns into the
    dashboard hash route that finishes the flow, for the same RFC 6749 reason
    ``oauth_service.redirect_uri`` gives.
    """
    base = (config.public_base_url or "").rstrip("/")
    return f"{base}/connected-accounts/{provider}/callback"


def default_return_url(config: GatewayConfig) -> str:
    """Where a flow started without a ``return_url`` lands: a page on this deployment."""
    base = (config.public_base_url or "").rstrip("/")
    return f"{base}/#/connections"


def validate_return_url(url: str) -> None:
    """Refuse a ``return_url`` a browser could be sent to unsafely.

    Supplied by the authenticated application, not the browser, so this is
    the same trust as a redirect URI in config. Still: https only, except
    http on localhost for development, and no fragment (it would swallow the
    query Otari appends).

    Raises:
        ConnectedAccountReturnUrlError: If the URL is not acceptable.

    """
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    if parts.scheme == "https" and host:
        ok = True
    elif parts.scheme == "http" and host in {"localhost", "127.0.0.1", "::1"}:
        ok = True
    else:
        ok = False
    if not ok or parts.fragment:
        raise ConnectedAccountReturnUrlError(url)


def with_query(url: str, **params: str) -> str:
    """``url`` with ``params`` appended as query parameters.

    For a hash-routed page (``…/#/connections``, the default return URL) the
    parameters go inside the fragment, which is the only part a hash router
    reads; a ``return_url`` an application supplies may carry no fragment
    (:func:`validate_return_url`), so its parameters go in the real query.
    """
    query = urlencode({key: value for key, value in params.items() if value})
    if not query:
        return url
    parts = urlsplit(url)
    if parts.fragment:
        separator = "&" if "?" in parts.fragment else "?"
        return f"{url}{separator}{query}"
    return f"{url}&{query}" if parts.query else f"{url}?{query}"


def provider_config(config: GatewayConfig, provider: str, scopes: list[str] | None = None) -> ProviderConfig:
    """apron-auth's ``ProviderConfig`` for ``provider`` on this deployment.

    Raises:
        ConnectedAppNotConfiguredError: For an unknown app, one without client
            credentials, or a deployment with no ``public_base_url``.

    """
    entry = config.connected_app(provider)
    if entry is None or provider not in _PRESETS:
        raise ConnectedAppNotConfiguredError(provider)
    kwargs: dict[str, Any] = {
        "client_id": entry["client_id"],
        "client_secret": entry["client_secret"],
        "scopes": list(scopes if scopes is not None else entry.get("scopes") or []),
        "redirect_uri": redirect_uri(config, provider),
    }
    if provider == "slack":
        kwargs["user_scopes"] = list(entry.get("user_scopes") or [])
    built, _revocation = _PRESETS[provider](**kwargs)
    return built  # type: ignore[no-any-return]


def _revocation_handler(config: GatewayConfig, provider: str) -> RevocationHandler | None:
    entry = config.connected_app(provider)
    if entry is None:
        return None
    kwargs: dict[str, Any] = {"client_id": entry["client_id"], "client_secret": entry["client_secret"], "scopes": []}
    if provider == "slack":
        kwargs["user_scopes"] = []
    _built, revocation = _PRESETS[provider](**kwargs)
    return revocation  # type: ignore[no-any-return]


class DatabaseStateStore:
    """apron-auth's ``StateStore`` over ``connected_account_oauth_states``.

    Bound to the end user the flow is for and the provider it is with, both
    recorded when the application started it. ``consume`` refuses a state that
    does not match, and reports it the same way as an unknown one so a stolen
    state cannot be told apart from a stale one.
    """

    def __init__(
        self, db: AsyncSession, *, end_user_id: uuid.UUID, provider: str, return_url: str | None = None
    ) -> None:
        self._db = db
        self._end_user_id = end_user_id
        self._provider = provider
        self._return_url = return_url

    async def save(self, state: OAuthPendingState) -> None:
        now = datetime.now(UTC)
        # Opportunistic sweep of stale rows, so the table does not grow with
        # every abandoned consent screen.
        await self._db.execute(
            delete(ConnectedAccountOAuthState).where(ConnectedAccountOAuthState.created_at < now - OAUTH_STATE_TTL)
        )
        try:
            verifier = encrypt_secret(state.code_verifier) if state.code_verifier else None
        except SecretBoxUnavailableError as error:
            raise SecretBoxUnavailableTenancyError() from error
        scopes = state.metadata.get("scopes")
        self._db.add(
            ConnectedAccountOAuthState(
                state=state.state,
                end_user_id=self._end_user_id,
                provider=self._provider,
                redirect_uri=state.redirect_uri,
                return_url=self._return_url,
                encrypted_code_verifier=verifier,
                requested_scopes=list(scopes) if isinstance(scopes, list) else None,
                created_at=datetime.fromtimestamp(state.created_at, tz=UTC),
            )
        )
        await self._db.commit()

    async def consume(self, state_key: str) -> OAuthPendingState | None:
        now = datetime.now(UTC)
        # One UPDATE decides: unconsumed, unexpired, this user's, this provider's.
        result = await self._db.execute(
            update(ConnectedAccountOAuthState)
            .where(
                ConnectedAccountOAuthState.state == state_key,
                ConnectedAccountOAuthState.end_user_id == self._end_user_id,
                ConnectedAccountOAuthState.provider == self._provider,
                ConnectedAccountOAuthState.consumed_at.is_(None),
                ConnectedAccountOAuthState.created_at >= now - OAUTH_STATE_TTL,
            )
            .values(consumed_at=now)
        )
        await self._db.commit()
        consumed = getattr(result, "rowcount", 0)
        if consumed != 1:
            return None
        row = (
            await self._db.execute(
                select(ConnectedAccountOAuthState).where(ConnectedAccountOAuthState.state == state_key)
            )
        ).scalar_one()
        verifier = decrypt_secret(row.encrypted_code_verifier) if row.encrypted_code_verifier else None
        return OAuthPendingState(
            state=row.state,
            redirect_uri=row.redirect_uri,
            code_verifier=verifier,
            created_at=row.created_at.timestamp(),
            metadata={"scopes": row.requested_scopes} if row.requested_scopes is not None else {},
        )


# ---------------------------------------------------------------------------
# The service
# ---------------------------------------------------------------------------


class ConnectedAccountService:
    """Connect, list, refresh and disconnect an application's users' third-party accounts.

    Every method that acts for a user takes the workspace (the API key's) and
    the application's ``user`` string; the :class:`EndUser` row is created on
    first use. The workspace is the isolation boundary: two applications
    naming a user ``alice`` never see each other's grants.
    """

    def __init__(self, db: AsyncSession, config: GatewayConfig) -> None:
        self._db = db
        self._config = config

    # -- wiring (overridable in tests) ----------------------------------------

    def _client(
        self,
        provider: str,
        *,
        end_user_id: uuid.UUID,
        scopes: list[str] | None = None,
        return_url: str | None = None,
    ) -> OAuthClient:
        built = provider_config(self._config, provider, scopes)
        return OAuthClient(
            built,
            state_store=DatabaseStateStore(self._db, end_user_id=end_user_id, provider=provider, return_url=return_url),
            revocation_handler=_revocation_handler(self._config, provider),
            identity_handler=_identity_handler(provider, built),
        )

    # -- end users -------------------------------------------------------------

    async def end_user(self, workspace_id: uuid.UUID, user: str, *, create: bool) -> EndUser | None:
        """The end user row for ``user`` in ``workspace_id``, created on first use when ``create``."""
        row = (
            await self._db.execute(
                select(EndUser).where(EndUser.workspace_id == workspace_id, EndUser.external_id == user)
            )
        ).scalar_one_or_none()
        if row is None and create:
            row = EndUser(workspace_id=workspace_id, external_id=user)
            self._db.add(row)
            await self._db.commit()
            await self._db.refresh(row)
        return row

    # -- reads -----------------------------------------------------------------

    async def list_apps(self, workspace_id: uuid.UUID, user: str | None = None) -> list[ConnectedAppPublic]:
        counts: dict[str, int] = {}
        end_user = await self.end_user(workspace_id, user, create=False) if user else None
        if end_user is not None:
            grouped = await self._db.execute(
                select(ConnectedAccount.provider, func.count())
                .where(ConnectedAccount.end_user_id == end_user.id)
                .group_by(ConnectedAccount.provider)
            )
            counts = {str(provider): int(count) for provider, count in grouped.all()}
        apps = []
        for provider in self._config.connected_app_providers:
            built = provider_config(self._config, provider)
            apps.append(
                ConnectedAppPublic(
                    provider=provider,
                    label=_PROVIDER_LABELS.get(provider, provider),
                    scopes=list(built.scopes),
                    scope_details=[
                        ScopePublic(
                            scope=meta.scope,
                            label=meta.label,
                            description=meta.description,
                            access_type=meta.access_type,
                            required=meta.required,
                        )
                        for meta in built.scope_metadata
                    ],
                    connected_accounts=counts.get(provider, 0),
                )
            )
        return apps

    async def list_accounts(
        self, workspace_id: uuid.UUID, user: str, provider: str | None = None
    ) -> ConnectedAccountsPublic:
        end_user = await self.end_user(workspace_id, user, create=False)
        if end_user is None:
            return ConnectedAccountsPublic(data=[], count=0)
        statement = select(ConnectedAccount).where(ConnectedAccount.end_user_id == end_user.id)
        if provider:
            statement = statement.where(ConnectedAccount.provider == provider)
        rows = (
            await self._db.execute(statement.order_by(ConnectedAccount.provider, ConnectedAccount.created_at))
        ).scalars()
        data = [_public(row, user) for row in rows]
        return ConnectedAccountsPublic(data=data, count=len(data))

    async def get_account(self, workspace_id: uuid.UUID, user: str, account_id: uuid.UUID) -> ConnectedAccountPublic:
        return _public(await self._row(workspace_id, user, account_id), user)

    # -- the flow --------------------------------------------------------------

    async def authorize(
        self,
        workspace_id: uuid.UUID,
        user: str,
        provider: str,
        *,
        scopes: list[str] | None = None,
        return_url: str | None = None,
    ) -> AuthorizePublic:
        """Start a flow for ``user``: the consent URL to send their browser to.

        Raises:
            ConnectedAppNotConfiguredError: If the app is not configured here.
            ConnectedAccountReturnUrlError: If ``return_url`` is not acceptable.
            ConnectedAccountLimitReachedError: If the user holds too many accounts.
            SecretBoxUnavailableTenancyError: If tokens could not be stored anyway.

        """
        provider_config(self._config, provider)  # refuse an unconfigured app before writing anything
        if return_url is not None:
            validate_return_url(return_url)
        if not _secret_box_ready():
            raise SecretBoxUnavailableTenancyError()
        end_user = await self.end_user(workspace_id, user, create=True)
        assert end_user is not None
        count = (
            await self._db.execute(select(func.count()).where(ConnectedAccount.end_user_id == end_user.id))
        ).scalar_one()
        if count >= MAX_CONNECTED_ACCOUNTS_PER_USER:
            raise ConnectedAccountLimitReachedError(MAX_CONNECTED_ACCOUNTS_PER_USER)
        client = self._client(provider, end_user_id=end_user.id, scopes=scopes, return_url=return_url)
        url, pending = await client.get_authorization_url(metadata={"scopes": scopes} if scopes else None)
        return AuthorizePublic(
            authorization_url=url,
            expires_at=datetime.fromtimestamp(pending.created_at, tz=UTC) + OAUTH_STATE_TTL,
        )

    async def complete(self, *, state: str, code: str) -> tuple[ConnectedAccountPublic, str]:
        """Finish a flow from the hosted callback: exchange the code and store the grant.

        Returns the account and the URL to send the browser to. Trusts nothing
        but ``state``, which was minted here and names the end user, the
        provider and the return URL.

        Raises:
            ConnectedAccountStateInvalidError: If ``state`` is unknown, used or expired.
            ConnectedAccountExchangeError: If the provider refuses the exchange.

        """
        pending = (
            await self._db.execute(
                select(ConnectedAccountOAuthState).where(
                    ConnectedAccountOAuthState.state == state,
                    ConnectedAccountOAuthState.consumed_at.is_(None),
                    ConnectedAccountOAuthState.created_at >= datetime.now(UTC) - OAUTH_STATE_TTL,
                )
            )
        ).scalar_one_or_none()
        if pending is None:
            raise ConnectedAccountStateInvalidError()
        end_user = (await self._db.execute(select(EndUser).where(EndUser.id == pending.end_user_id))).scalar_one()
        provider = pending.provider
        client = self._client(provider, end_user_id=end_user.id, scopes=pending.requested_scopes)
        try:
            tokens = await client.exchange_code(code, state=state)
        except OAuthError as error:
            if _looks_like_missing_state(error):
                raise ConnectedAccountStateInvalidError() from error
            logger.warning("connection exchange with %s failed: %s", provider, type(error).__name__)
            raise ConnectedAccountExchangeError(provider, "code exchange") from error
        identifier, account_label, metadata = await self._identity(client, provider, tokens)
        row = await self._upsert(end_user.id, provider, tokens, identifier, account_label, metadata)
        return _public(row, end_user.external_id), pending.return_url or default_return_url(self._config)

    async def return_url_for_state(self, state: str) -> str:
        """Where to send a browser whose flow failed, from the state it carries, else the default page."""
        stored = (
            await self._db.execute(
                select(ConnectedAccountOAuthState.return_url).where(ConnectedAccountOAuthState.state == state)
            )
        ).scalar_one_or_none()
        return stored or default_return_url(self._config)

    async def update(
        self, workspace_id: uuid.UUID, user: str, account_id: uuid.UUID, body: ConnectedAccountUpdate
    ) -> ConnectedAccountPublic:
        row = await self._row(workspace_id, user, account_id)
        row.label = body.label
        await self._db.commit()
        await self._db.refresh(row)
        return _public(row, user)

    async def disconnect(self, workspace_id: uuid.UUID, user: str, account_id: uuid.UUID) -> None:
        """Delete the grant, revoking it at the provider first when that is possible.

        Revocation is best effort: a provider that is down should not keep a
        user from removing a credential they no longer want this deployment to
        hold. The row is deleted either way.
        """
        row = await self._row(workspace_id, user, account_id)
        try:
            token = decrypt_secret(row.encrypted_access_token)
        except SecretDecryptionError:
            token = None
        if token is not None:
            try:
                await self._client(row.provider, end_user_id=row.end_user_id).revoke_token(token)
            except (OAuthError, ConnectedAppNotConfiguredError) as error:
                logger.info("revocation at %s skipped: %s", row.provider, type(error).__name__)
        await self._db.delete(row)
        await self._db.commit()

    # -- credentials ---------------------------------------------------------------

    async def access_token(self, workspace_id: uuid.UUID, user: str, account_id: uuid.UUID) -> AccessToken:
        """A live token for ``account_id``, refreshed first if it is about to expire.

        Raises:
            ConnectedAccountNotFoundError: If the account is not this user's.
            ConnectedAccountExchangeError: If a needed refresh fails.

        """
        row = await self._row(workspace_id, user, account_id)
        return await self._live_token(row)

    async def access_token_for_provider(self, workspace_id: uuid.UUID, user: str, provider: str) -> AccessToken | None:
        """The token for the user's account with ``provider``, or None when they have none.

        With several accounts for one provider the oldest wins; a caller that
        needs a specific one names it by id.
        """
        end_user = await self.end_user(workspace_id, user, create=False)
        if end_user is None:
            return None
        row = (
            await self._db.execute(
                select(ConnectedAccount)
                .where(ConnectedAccount.end_user_id == end_user.id, ConnectedAccount.provider == provider)
                .order_by(ConnectedAccount.created_at)
                .limit(1)
            )
        ).scalar_one_or_none()
        return None if row is None else await self._live_token(row)

    # -- internals ---------------------------------------------------------------

    async def _live_token(self, row: ConnectedAccount) -> AccessToken:
        expiring = row.expires_at is not None and row.expires_at <= datetime.now(UTC) + REFRESH_LEEWAY
        if expiring and row.encrypted_refresh_token:
            try:
                tokens = await self._client(row.provider, end_user_id=row.end_user_id).refresh_token(
                    decrypt_secret(row.encrypted_refresh_token)
                )
            except OAuthError as error:
                raise ConnectedAccountExchangeError(row.provider, "token refresh") from error
            _apply_tokens(row, tokens, keep_refresh=True)
            await self._db.commit()
            await self._db.refresh(row)
        extra: dict[str, str] = (
            json.loads(decrypt_secret(row.encrypted_extra_tokens)) if row.encrypted_extra_tokens else {}
        )
        return AccessToken(
            provider=row.provider,
            token=decrypt_secret(row.encrypted_access_token),
            token_type=row.token_type,
            expires_at=row.expires_at,
            extra=extra,
        )

    async def _row(self, workspace_id: uuid.UUID, user: str, account_id: uuid.UUID) -> ConnectedAccount:
        row = (
            await self._db.execute(
                select(ConnectedAccount)
                .join(EndUser, EndUser.id == ConnectedAccount.end_user_id)
                .where(
                    ConnectedAccount.id == account_id,
                    EndUser.workspace_id == workspace_id,
                    EndUser.external_id == user,
                )
            )
        ).scalar_one_or_none()
        if row is None:
            raise ConnectedAccountNotFoundError(account_id)
        return row

    async def _identity(
        self, client: OAuthClient, provider: str, tokens: TokenSet
    ) -> tuple[str | None, str | None, dict[str, Any]]:
        """Who the account is, from the provider's own identity endpoint. Best effort."""
        try:
            profile = await client.fetch_identity(tokens)
        except Exception as error:  # noqa: BLE001 - identity is a nicety; the grant is what matters
            logger.info("identity fetch at %s skipped: %s", provider, type(error).__name__)
            return None, None, {}
        tenancy = profile.tenancies[0] if profile.tenancies else None
        tenancy_id = getattr(tenancy, "id", None) if tenancy is not None else None
        tenancy_name = getattr(tenancy, "name", None) if tenancy is not None else None
        identifier = profile.email or profile.username or profile.subject or tenancy_id
        label = profile.name or profile.email or profile.username or tenancy_name
        if tenancy_name and label and tenancy_name != label:
            label = f"{label} ({tenancy_name})"
        metadata: dict[str, Any] = {
            key: value
            for key, value in {
                "subject": profile.subject,
                "email": profile.email,
                "name": profile.name,
                "username": profile.username,
                "tenancy_id": tenancy_id,
                "tenancy_name": tenancy_name,
            }.items()
            if value
        }
        return (str(identifier) if identifier else None), (str(label) if label else None), metadata

    async def _upsert(
        self,
        end_user_id: uuid.UUID,
        provider: str,
        tokens: TokenSet,
        identifier: str | None,
        account_label: str | None,
        metadata: dict[str, Any],
    ) -> ConnectedAccount:
        row: ConnectedAccount | None = None
        if identifier is not None:
            row = (
                await self._db.execute(
                    select(ConnectedAccount).where(
                        ConnectedAccount.end_user_id == end_user_id,
                        ConnectedAccount.provider == provider,
                        ConnectedAccount.account_identifier == identifier,
                    )
                )
            ).scalar_one_or_none()
        if row is None:
            row = ConnectedAccount(
                end_user_id=end_user_id, provider=provider, account_identifier=identifier, encrypted_access_token=""
            )
            self._db.add(row)
        row.account_label = account_label
        row.account_metadata = metadata or None
        try:
            _apply_tokens(row, tokens, keep_refresh=False)
        except SecretBoxUnavailableError as error:
            raise SecretBoxUnavailableTenancyError() from error
        await self._db.commit()
        await self._db.refresh(row)
        return row


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _identity_handler(provider: str, built: ProviderConfig) -> Any:
    module = {
        "atlassian": apron_atlassian,
        "github": apron_github,
        "google": apron_google,
        "hubspot": apron_hubspot,
        "linear": apron_linear,
        "microsoft": apron_microsoft,
        "notion": apron_notion,
        "salesforce": apron_salesforce,
        "slack": apron_slack,
        "typeform": apron_typeform,
    }[provider]
    return module.maybe_identity_handler(built)


def _secret_box_ready() -> bool:
    try:
        encrypt_secret("probe")
    except SecretBoxUnavailableError:
        return False
    return True


def _looks_like_missing_state(error: OAuthError) -> bool:
    text = str(error).lower()
    return "state" in text and ("not found" in text or "invalid" in text or "expired" in text or "unknown" in text)


def _expires_at(tokens: TokenSet) -> datetime | None:
    if tokens.expires_at is not None:
        return datetime.fromtimestamp(tokens.expires_at, tz=UTC)
    if tokens.expires_in is not None:
        return datetime.fromtimestamp(time.time() + tokens.expires_in, tz=UTC)
    return None


def _extra_tokens(tokens: TokenSet) -> dict[str, str]:
    """Secondary credentials some providers issue next to the primary one.

    Slack's ``oauth.v2.access`` answers with the bot token as ``access_token``
    and the user's own under ``authed_user``; both are kept so a consumer can
    act as either.
    """
    extra: dict[str, str] = {}
    authed_user = tokens.metadata.get("authed_user")
    if isinstance(authed_user, dict) and isinstance(authed_user.get("access_token"), str):
        extra["user"] = authed_user["access_token"]
    return extra


def _apply_tokens(row: ConnectedAccount, tokens: TokenSet, *, keep_refresh: bool) -> None:
    row.encrypted_access_token = encrypt_secret(tokens.access_token)
    if tokens.refresh_token:
        row.encrypted_refresh_token = encrypt_secret(tokens.refresh_token)
    elif not keep_refresh:
        row.encrypted_refresh_token = None
    extra = _extra_tokens(tokens)
    if extra:
        row.encrypted_extra_tokens = encrypt_secret(json.dumps(extra))
    elif not keep_refresh:
        row.encrypted_extra_tokens = None
    row.token_type = tokens.token_type or "Bearer"
    row.expires_at = _expires_at(tokens)
    if tokens.scope:
        row.scopes = tokens.scope.replace(",", " ").split()


def _public(row: ConnectedAccount, user: str) -> ConnectedAccountPublic:
    return ConnectedAccountPublic(
        id=row.id,
        user=user,
        provider=row.provider,
        account_identifier=row.account_identifier,
        account_label=row.account_label,
        label=row.label,
        scopes=list(row.scopes or []),
        expires_at=row.expires_at,
        has_refresh_token=row.encrypted_refresh_token is not None,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def provider_label(provider: str) -> str:
    return _PROVIDER_LABELS.get(provider, provider)
