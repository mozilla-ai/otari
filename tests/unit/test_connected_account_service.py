"""Connections: config, the database state store, and the service, against an
in-memory database with apron-auth's client replaced by a scripted stand-in."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import Awaitable, Callable, Iterator
from datetime import UTC, datetime, timedelta
from typing import Any, TypeVar

import pytest
from apron_auth.errors import OAuthError
from apron_auth.models import IdentityProfile, OAuthPendingState, TenancyContext, TokenSet
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel

import gateway.models  # noqa: F401  (registers every table on the shared metadata)
from gateway.core.config import GatewayConfig
from gateway.models.entities import ConnectedAccount, ConnectedAccountOAuthState, EndUser
from gateway.models.tenancy import Organization, Workspace
from gateway.services.secret_box import generate_secret_key
from gateway.services.tenancy import connected_account_service as svc
from gateway.services.tenancy.connected_account_service import (
    ConnectedAccountService,
    ConnectedAccountUpdate,
    DatabaseStateStore,
    provider_config,
    redirect_uri,
    validate_return_url,
    with_query,
)
from gateway.services.tenancy.errors import (
    ConnectedAccountExchangeError,
    ConnectedAccountNotFoundError,
    ConnectedAccountReturnUrlError,
    ConnectedAccountStateInvalidError,
    ConnectedAppNotConfiguredError,
    SecretBoxUnavailableTenancyError,
)

T = TypeVar("T")
ALICE, BOB = "alice@acme.test", "bob@acme.test"


@pytest.fixture(autouse=True)
def _secret_key(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    yield


def configured(**apps: Any) -> GatewayConfig:
    entries = apps or {
        "slack": {"client_id": "slack-id", "client_secret": "slack-secret", "user_scopes": ["channels:read"]},
        "github": {"client_id": "gh-id", "client_secret": "gh-secret", "scopes": ["repo"]},
    }
    return GatewayConfig(public_base_url="https://otari.example.com", connected_apps=entries)


def run(scenario: Callable[[AsyncSession], Awaitable[T]]) -> T:
    async def main() -> T:
        engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        try:
            async with engine.begin() as conn:
                await conn.run_sync(SQLModel.metadata.create_all)
            async with async_sessionmaker(engine, expire_on_commit=False)() as session:
                return await scenario(session)
        finally:
            await engine.dispose()

    return asyncio.run(main())


async def make_workspace(session: AsyncSession) -> uuid.UUID:
    slug = uuid.uuid4().hex[:8]
    organization = Organization(name=f"Acme {slug}", slug=f"acme-{slug}")
    session.add(organization)
    await session.commit()
    await session.refresh(organization)
    workspace = Workspace(name=f"ws-{slug}", organization_id=organization.id)
    session.add(workspace)
    await session.commit()
    await session.refresh(workspace)
    return workspace.id


class FakeOAuthClient:
    """Scripted stand-in for apron-auth's client: same surface, no network."""

    def __init__(self, store: DatabaseStateStore, *, tokens: TokenSet, identity: IdentityProfile | None) -> None:
        self.store = store
        self.tokens = tokens
        self.identity = identity
        self.revoked: list[str] = []
        self.refreshed: list[str] = []

    async def get_authorization_url(
        self, redirect_uri: str | None = None, metadata: dict[str, Any] | None = None
    ) -> tuple[str, OAuthPendingState]:
        pending = OAuthPendingState(
            state=uuid.uuid4().hex,
            redirect_uri=redirect_uri or "https://otari.example.com/cb",
            code_verifier="verifier-123",
            created_at=time.time(),
            metadata=metadata or {},
        )
        await self.store.save(pending)
        return f"https://provider.test/authorize?state={pending.state}", pending

    async def exchange_code(self, code: str, state: str | None = None, **_: Any) -> TokenSet:
        assert state is not None
        pending = await self.store.consume(state)
        if pending is None:
            msg = "State not found or expired"
            raise OAuthError(msg)
        if code == "bad-code":
            msg = "invalid_grant"
            raise OAuthError(msg)
        assert pending.code_verifier == "verifier-123"
        return self.tokens

    async def fetch_identity(self, tokens: TokenSet) -> IdentityProfile:
        if self.identity is None:
            msg = "no identity"
            raise OAuthError(msg)
        return self.identity

    async def refresh_token(self, refresh_token: str) -> TokenSet:
        self.refreshed.append(refresh_token)
        return TokenSet(access_token="refreshed-access", refresh_token=None, expires_in=3600, scope=self.tokens.scope)

    async def revoke_token(self, token: str) -> bool:
        self.revoked.append(token)
        return True


class Service(ConnectedAccountService):
    """The real service with the apron client swapped for the fake."""

    def __init__(
        self, db: AsyncSession, config: GatewayConfig, *, tokens: TokenSet, identity: IdentityProfile | None
    ) -> None:
        super().__init__(db, config)
        self.fake: FakeOAuthClient | None = None
        self._fake_tokens, self._fake_identity = tokens, identity

    def _client(
        self,
        provider: str,
        *,
        end_user_id: uuid.UUID,
        scopes: list[str] | None = None,
        return_url: str | None = None,
    ) -> Any:
        provider_config(self._config, provider, scopes)  # still refuses an unconfigured app
        self.fake = FakeOAuthClient(
            DatabaseStateStore(self._db, end_user_id=end_user_id, provider=provider, return_url=return_url),
            tokens=self._fake_tokens,
            identity=self._fake_identity,
        )
        return self.fake


SLACK_TOKENS = TokenSet(
    access_token="xoxb-bot",
    refresh_token="refresh-1",
    expires_in=3600,
    scope="chat:write,channels:read",
    metadata={"authed_user": {"id": "U1", "access_token": "xoxp-user"}},
)
SLACK_IDENTITY = IdentityProfile(
    provider="slack", subject="U1", name="Alice", tenancies=(TenancyContext(id="T001", name="Acme Corp"),)
)


async def _authorize_and_complete(
    service: Service, workspace_id: uuid.UUID, user: str, provider: str, **kw: Any
) -> Any:
    started = await service.authorize(workspace_id, user, provider, **kw)
    state = started.authorization_url.rsplit("state=", 1)[1]
    account, _return = await service.complete(state=state, code="good-code")
    return account


# -- config and helpers ----------------------------------------------------------


def test_config_validates_entries() -> None:
    with pytest.raises(ValueError, match="not a supported app"):
        GatewayConfig(connected_apps={"myspace": {"client_id": "a", "client_secret": "b"}})
    with pytest.raises(ValueError, match="client_secret is required"):
        GatewayConfig(connected_apps={"slack": {"client_id": "a"}})
    with pytest.raises(ValueError, match="user_scopes is only meaningful for slack"):
        GatewayConfig(connected_apps={"github": {"client_id": "a", "client_secret": "b", "user_scopes": []}})
    assert configured().connected_app_providers == ("github", "slack")


def test_an_app_without_public_base_url_is_not_on_offer() -> None:
    config = GatewayConfig(connected_apps={"github": {"client_id": "a", "client_secret": "b"}})
    assert config.connected_app_providers == ()
    with pytest.raises(ConnectedAppNotConfiguredError):
        provider_config(config, "github")


def test_provider_config_uses_presets_and_derived_redirect() -> None:
    config = configured()
    built = provider_config(config, "github")
    assert built.client_id == "gh-id"
    assert "repo" in built.scopes and "read:user" in built.scopes  # preset base scopes are kept
    assert built.redirect_uri == "https://otari.example.com/connected-accounts/github/callback"
    assert redirect_uri(config, "slack").endswith("/connected-accounts/slack/callback")
    assert provider_config(config, "slack").authorize_url.startswith("https://slack.com/oauth")


def test_return_url_rules() -> None:
    validate_return_url("https://myapp.example.com/settings?tab=apps")
    validate_return_url("http://localhost:3000/settings")
    for bad in ("http://myapp.example.com/x", "https://myapp.example.com/x#frag", "javascript:alert(1)", "/relative"):
        with pytest.raises(ConnectedAccountReturnUrlError):
            validate_return_url(bad)
    assert (
        with_query("https://a.test/p", connection="ok", provider="slack")
        == "https://a.test/p?connection=ok&provider=slack"
    )
    assert with_query("https://a.test/p?x=1", connection="ok") == "https://a.test/p?x=1&connection=ok"
    assert with_query("https://a.test/#/connections", connection="ok") == "https://a.test/#/connections?connection=ok"


# -- the flow ------------------------------------------------------------------


def test_connect_flow_creates_end_user_and_stores_encrypted_tokens() -> None:
    async def scenario(session: AsyncSession) -> None:
        workspace = await make_workspace(session)
        service = Service(session, configured(), tokens=SLACK_TOKENS, identity=SLACK_IDENTITY)
        started = await service.authorize(workspace, ALICE, "slack", return_url="https://myapp.test/settings")
        assert started.authorization_url.startswith("https://provider.test/authorize")
        assert started.expires_at > datetime.now(UTC) + timedelta(minutes=9)
        end_user = (await session.execute(select(EndUser))).scalar_one()
        assert (end_user.workspace_id, end_user.external_id) == (workspace, ALICE)
        pending = (await session.execute(select(ConnectedAccountOAuthState))).scalar_one()
        assert pending.end_user_id == end_user.id and pending.return_url == "https://myapp.test/settings"
        assert pending.encrypted_code_verifier not in (None, "verifier-123")

        state = started.authorization_url.rsplit("state=", 1)[1]
        account, return_to = await service.complete(state=state, code="good-code")
        assert return_to == "https://myapp.test/settings"
        assert (account.user, account.provider, account.account_identifier) == (ALICE, "slack", "U1")
        assert account.account_label == "Alice (Acme Corp)"
        assert account.scopes == ["chat:write", "channels:read"] and account.has_refresh_token

        row = (await session.execute(select(ConnectedAccount))).scalar_one()
        assert "xoxb-bot" not in (row.encrypted_access_token + (row.encrypted_extra_tokens or ""))
        token = await service.access_token(workspace, ALICE, account.id)
        assert token.token == "xoxb-bot" and token.extra == {"user": "xoxp-user"}
        apps = await service.list_apps(workspace, ALICE)
        assert {app.provider: app.connected_accounts for app in apps} == {"github": 0, "slack": 1}

        with pytest.raises(ConnectedAccountStateInvalidError):  # single use
            await service.complete(state=state, code="good-code")

    run(scenario)


def test_default_return_url_when_none_given() -> None:
    async def scenario(session: AsyncSession) -> None:
        workspace = await make_workspace(session)
        service = Service(session, configured(), tokens=SLACK_TOKENS, identity=SLACK_IDENTITY)
        started = await service.authorize(workspace, ALICE, "slack")
        state = started.authorization_url.rsplit("state=", 1)[1]
        _, return_to = await service.complete(state=state, code="c")
        assert return_to == "https://otari.example.com/#/connections"
        assert await service.return_url_for_state("unknown") == "https://otari.example.com/#/connections"

    run(scenario)


def test_workspaces_isolate_users_with_the_same_name() -> None:
    async def scenario(session: AsyncSession) -> None:
        first, second = await make_workspace(session), await make_workspace(session)
        service = Service(session, configured(), tokens=SLACK_TOKENS, identity=SLACK_IDENTITY)
        account = await _authorize_and_complete(service, first, ALICE, "slack")
        assert (await service.list_accounts(second, ALICE)).count == 0
        with pytest.raises(ConnectedAccountNotFoundError):
            await service.get_account(second, ALICE, account.id)
        with pytest.raises(ConnectedAccountNotFoundError):
            await service.get_account(first, BOB, account.id)
        assert await service.access_token_for_provider(second, ALICE, "slack") is None
        assert (await service.list_accounts(first, ALICE, provider="slack")).count == 1

    run(scenario)


def test_reconnecting_the_same_account_updates_the_row() -> None:
    async def scenario(session: AsyncSession) -> None:
        workspace = await make_workspace(session)
        service = Service(session, configured(), tokens=SLACK_TOKENS, identity=SLACK_IDENTITY)
        first = await _authorize_and_complete(service, workspace, ALICE, "slack")
        service._fake_tokens = TokenSet(access_token="xoxb-new", scope="chat:write")
        second = await _authorize_and_complete(service, workspace, ALICE, "slack")
        assert first.id == second.id
        assert (await service.list_accounts(workspace, ALICE)).count == 1
        assert (await service.access_token(workspace, ALICE, first.id)).token == "xoxb-new"

    run(scenario)


def test_expired_state_is_refused() -> None:
    async def scenario(session: AsyncSession) -> None:
        workspace = await make_workspace(session)
        service = Service(session, configured(), tokens=SLACK_TOKENS, identity=SLACK_IDENTITY)
        started = await service.authorize(workspace, ALICE, "slack")
        row = (await session.execute(select(ConnectedAccountOAuthState))).scalar_one()
        row.created_at = datetime.now(UTC) - timedelta(minutes=11)
        await session.commit()
        with pytest.raises(ConnectedAccountStateInvalidError):
            await service.complete(state=started.authorization_url.rsplit("state=", 1)[1], code="c")

    run(scenario)


def test_provider_refusal_is_a_502_and_missing_identity_is_tolerated() -> None:
    async def scenario(session: AsyncSession) -> None:
        workspace = await make_workspace(session)
        service = Service(session, configured(), tokens=TokenSet(access_token="gho_x", scope="repo"), identity=None)
        started = await service.authorize(workspace, ALICE, "github")
        with pytest.raises(ConnectedAccountExchangeError):
            await service.complete(state=started.authorization_url.rsplit("state=", 1)[1], code="bad-code")
        account = await _authorize_and_complete(service, workspace, ALICE, "github")
        assert account.account_identifier is None and account.account_label is None
        assert (await service.access_token(workspace, ALICE, account.id)).token == "gho_x"

    run(scenario)


def test_access_token_refreshes_when_about_to_expire() -> None:
    async def scenario(session: AsyncSession) -> None:
        workspace = await make_workspace(session)
        service = Service(session, configured(), tokens=SLACK_TOKENS, identity=SLACK_IDENTITY)
        account = await _authorize_and_complete(service, workspace, ALICE, "slack")
        row = (await session.execute(select(ConnectedAccount))).scalar_one()
        row.expires_at = datetime.now(UTC) + timedelta(seconds=10)
        await session.commit()
        token = await service.access_token_for_provider(workspace, ALICE, "slack")
        assert token is not None and token.token == "refreshed-access"
        assert service.fake is not None and service.fake.refreshed == ["refresh-1"]
        assert (await service.get_account(workspace, ALICE, account.id)).has_refresh_token is True

    run(scenario)


def test_update_label_and_disconnect_revokes() -> None:
    async def scenario(session: AsyncSession) -> None:
        workspace = await make_workspace(session)
        service = Service(session, configured(), tokens=SLACK_TOKENS, identity=SLACK_IDENTITY)
        account = await _authorize_and_complete(service, workspace, ALICE, "slack")
        updated = await service.update(workspace, ALICE, account.id, ConnectedAccountUpdate(label="Work Slack"))
        assert updated.label == "Work Slack"
        await service.disconnect(workspace, ALICE, account.id)
        assert service.fake is not None and service.fake.revoked == ["xoxb-bot"]
        assert (await service.list_accounts(workspace, ALICE)).count == 0

    run(scenario)


def test_unconfigured_app_bad_return_url_and_missing_secret_key_are_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario(session: AsyncSession) -> None:
        workspace = await make_workspace(session)
        service = Service(session, configured(), tokens=SLACK_TOKENS, identity=SLACK_IDENTITY)
        with pytest.raises(ConnectedAppNotConfiguredError):
            await service.authorize(workspace, ALICE, "notion")
        with pytest.raises(ConnectedAccountReturnUrlError):
            await service.authorize(workspace, ALICE, "slack", return_url="http://evil.test/")
        assert (await session.execute(select(EndUser))).scalar_one_or_none() is None  # nothing written on refusal
        monkeypatch.delenv("OTARI_SECRET_KEY")
        with pytest.raises(SecretBoxUnavailableTenancyError):
            await service.authorize(workspace, ALICE, "slack")

    run(scenario)


def test_token_helpers() -> None:
    assert svc._extra_tokens(SLACK_TOKENS) == {"user": "xoxp-user"}
    assert svc._extra_tokens(TokenSet(access_token="a")) == {}
    row = ConnectedAccount(end_user_id=uuid.uuid4(), provider="slack", encrypted_access_token="")
    svc._apply_tokens(row, TokenSet(access_token="a", scope="x y", expires_in=60), keep_refresh=False)
    assert row.scopes == ["x", "y"] and row.expires_at is not None and row.encrypted_refresh_token is None
    assert json.loads('{"user": "u"}') == {"user": "u"}
