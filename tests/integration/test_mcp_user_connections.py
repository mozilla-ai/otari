"""Per-user MCP connections: the sign-in, the token a request carries, its renewal, and every way one ends.

Exercised at the service layer, as `test_workspace_mcp_servers.py` is, because the cases that matter
need identities and users the one bootstrap operator cannot be. The wire is stood in for at the
functions of `_mcp_oauth`, whose own behavior `tests/unit/test_mcp_oauth.py` covers.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncGenerator, AsyncIterator, Iterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from urllib.parse import parse_qs, urlsplit

import pytest
import pytest_asyncio
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from gateway.core.config import API_ROOT, GatewayConfig
from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions.organizations_exceptions import NotAuthorizedError
from gateway.exceptions.shared_exceptions import SecretBoxUnavailableTenancyError
from gateway.exceptions.tools_exceptions import (
    McpConnectionFlowError,
    McpConnectionModeError,
    McpConnectionNotFoundError,
    McpConnectionSetupError,
    McpConnectionUnavailableError,
    McpConnectionUpstreamError,
    McpSignInRequiredError,
    WorkspaceMcpServerNotFoundError,
    WorkspaceMcpServerUnsafeUrlError,
)
from gateway.models.mcp import ResolvedMcpServer
from gateway.models.tenancy import Organization, User, Workspace
from gateway.models.tools import McpCredentialMode, McpOAuthFlow, McpUserConnection, WorkspaceMcpServer
from gateway.models.users import User as BillingUser
from gateway.repositories.tenancy import (
    OrganizationMemberRepository,
    OrganizationRepository,
    UserRepository,
    WorkspaceMemberRepository,
    WorkspaceRepository,
)
from gateway.repositories.tools import (
    McpOAuthFlowRepository,
    McpUserConnectionRepository,
    WorkspaceMcpServerRepository,
)
from gateway.services import mcp_stateless
from gateway.services.mcp_stateless import ServerCheck
from gateway.services.model_access import McpAccess, revoke_user_mcp_grants
from gateway.services.secret_box import decrypt_secret, encrypt_secret, generate_secret_key
from gateway.services.tenancy.workspace_mcp_server_service import (
    WorkspaceMcpServerCheckRequest,
    WorkspaceMcpServerCreate,
    WorkspaceMcpServerPublic,
    WorkspaceMcpServers,
    WorkspaceMcpServerService,
    WorkspaceMcpServerUpdate,
)
from gateway.services.tools import (
    McpConnectionService,
    McpGrantMembershipListener,
    McpUserTokens,
    _mcp_oauth,
    stored_oauth_client,
)
from gateway.services.tools._mcp_connections import checked_return_to
from gateway.services.tools._mcp_oauth import (
    AUTH_METHOD_POST,
    McpOAuthClient,
    McpOAuthMetadata,
    McpOAuthRegistrationError,
    McpOAuthTokenError,
    McpOAuthTokens,
)

from .conftest import _to_async_url

pytestmark = pytest.mark.asyncio

PUBLIC_URL = "https://93.184.216.34/mcp"
OTHER_PUBLIC_URL = "https://93.184.216.35/mcp"
ISSUER = "https://93.184.216.36"
ORIGIN = "https://otari.example.com"
CALLBACK = f"{ORIGIN}{API_ROOT}/mcp-connections/callback"


class _Wire:
    """Stands in for the authorization server: what it publishes, registers and issues, and what it was sent."""

    def __init__(self) -> None:
        self.metadata = McpOAuthMetadata(
            authorization_endpoint=f"{ISSUER}/authorize",
            token_endpoint=f"{ISSUER}/token",
            registration_endpoint=f"{ISSUER}/register",
            revocation_endpoint=f"{ISSUER}/revoke",
            scope="files:read",
            resource=PUBLIC_URL,
        )
        self.registrations: list[str] = []
        self.exchanges: list[dict[str, object]] = []
        self.refreshes: list[tuple[McpOAuthClient, str]] = []
        self.revoked: list[str] = []
        self.issue = McpOAuthTokens(access_token="access-1", refresh_token="refresh-1", expires_in=3600)
        self.refresh_error: McpOAuthTokenError | None = None
        self.refresh_gate: asyncio.Event | None = None

    async def discover(self, server_url: str) -> McpOAuthMetadata:
        return self.metadata

    async def register_client(self, registration_endpoint: str, redirect_uri: str) -> McpOAuthClient:
        self.registrations.append(redirect_uri)
        return McpOAuthClient(
            client_id=f"dynamic-{len(self.registrations)}", client_secret="dynamic-secret", auth_method=AUTH_METHOD_POST
        )

    async def exchange_code(
        self,
        token_endpoint: str,
        client: McpOAuthClient,
        *,
        code: str,
        redirect_uri: str,
        verifier: str,
        resource: str | None,
    ) -> McpOAuthTokens:
        if code == "refused":
            raise McpOAuthTokenError("The authorization server refused the code exchange", status_code=400)
        self.exchanges.append(
            {"client": client, "code": code, "redirect_uri": redirect_uri, "verifier": verifier, "resource": resource}
        )
        return self.issue

    async def refresh(
        self, token_endpoint: str, client: McpOAuthClient, *, refresh_token: str, resource: str | None
    ) -> McpOAuthTokens:
        self.refreshes.append((client, refresh_token))
        if self.refresh_gate is not None:
            await self.refresh_gate.wait()
        if self.refresh_error is not None:
            raise self.refresh_error
        count = len(self.refreshes)
        return McpOAuthTokens(access_token=f"renewed-{count}", refresh_token=f"rotated-{count}", expires_in=3600)

    async def revoke(self, revocation_endpoint: str, client: McpOAuthClient, *, token: str) -> bool:
        self.revoked.append(token)
        return True


@pytest.fixture(autouse=True)
def wire(monkeypatch: pytest.MonkeyPatch) -> _Wire:
    fake = _Wire()
    for name in ("discover", "register_client", "exchange_code", "refresh", "revoke"):
        monkeypatch.setattr(_mcp_oauth, name, getattr(fake, name))
    return fake


@pytest.fixture(autouse=True)
def _secret_key(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    yield


@pytest.fixture(autouse=True)
def checks(monkeypatch: pytest.MonkeyPatch) -> list[ResolvedMcpServer]:
    """Stands in for the connection check, and keeps the server each one was run with."""
    seen: list[ResolvedMcpServer] = []

    async def check(server: ResolvedMcpServer) -> ServerCheck:
        seen.append(server)
        return ServerCheck(tool_names=["list_files"])

    monkeypatch.setattr(mcp_stateless, "check_stored_server", check)
    return seen


@pytest.fixture
def config(test_config: GatewayConfig, monkeypatch: pytest.MonkeyPatch) -> GatewayConfig:
    monkeypatch.setattr(test_config, "public_base_url", ORIGIN)
    return test_config


@pytest_asyncio.fixture
async def tokens(async_db: AsyncSession, postgres_url: str) -> AsyncGenerator[McpUserTokens, None]:
    """The request path's token reader, renewing in a session of its own as it does in a running gateway."""
    engine = create_async_engine(_to_async_url(postgres_url))
    factory = async_sessionmaker(engine, expire_on_commit=False)

    @asynccontextmanager
    async def unit_of_work() -> AsyncIterator[UnitOfWork]:
        async with factory() as session:
            yield UnitOfWork(session)

    try:
        yield McpUserTokens(McpUserConnectionRepository(async_db), unit_of_work=unit_of_work)
    finally:
        await engine.dispose()


def _connections(db: AsyncSession, config: GatewayConfig) -> McpConnectionService:
    uow = UnitOfWork(db)
    return McpConnectionService(
        uow, WorkspaceMcpServerRepository(uow), McpUserConnectionRepository(uow), McpOAuthFlowRepository(uow), config
    )


def _service(db: AsyncSession, config: GatewayConfig) -> WorkspaceMcpServerService:
    return WorkspaceMcpServerService(db, connections=_connections(db, config), config=config)


async def _organization(db: AsyncSession) -> Organization:
    return await OrganizationRepository(db).create_organization(name="Acme", slug="acme", created_by_user_id=None)


async def _member(db: AsyncSession, organization: Organization, *, role: str, name: str) -> User:
    user = await UserRepository(db).create_local_identity(full_name=name, active_organization_id=organization.id)
    await OrganizationMemberRepository(db).create_membership(
        organization_id=organization.id, user_id=user.id, role=role
    )
    return user


class _World:
    """One workspace with an owner, a plain member, and a per-user server."""

    def __init__(self, workspace: Workspace, owner: User, member: User, server: WorkspaceMcpServerPublic) -> None:
        self.workspace = workspace
        self.owner = owner
        self.member = member
        self.server = server


async def _world(db: AsyncSession, config: GatewayConfig, **server: object) -> _World:
    organization = await _organization(db)
    owner = await _member(db, organization, role="owner", name="Owner")
    member = await _member(db, organization, role="member", name="Member")
    workspace = await WorkspaceRepository(db).create_workspace(
        name="Default", organization_id=organization.id, created_by_user_id=owner.id
    )
    await WorkspaceMemberRepository(db).create(workspace_id=workspace.id, user_id=owner.id, role="owner")
    await WorkspaceMemberRepository(db).create(workspace_id=workspace.id, user_id=member.id, role="member")
    fields: dict[str, object] = {"name": "drive", "url": PUBLIC_URL, "credential_mode": "per_user"}
    fields.update(server)
    created = await _service(db, config).create_server(
        user=owner,
        workspace_id=workspace.id,
        request=WorkspaceMcpServerCreate(**fields),  # type: ignore[arg-type]
    )
    return _World(workspace, owner, member, created)


def _state(authorization_url: str) -> str:
    return parse_qs(urlsplit(authorization_url).query)["state"][0]


async def _sign_in(db: AsyncSession, config: GatewayConfig, world: _World, user: User) -> None:
    service = _service(db, config)
    started = await service.start_connection(
        user=user, workspace_id=world.workspace.id, server_id=world.server.id, return_to=None
    )
    await service.complete_connection(user=user, state=_state(started.authorization_url), code="code")


async def _row(db: AsyncSession, server_id: uuid.UUID) -> WorkspaceMcpServer:
    row = await db.get(WorkspaceMcpServer, server_id, populate_existing=True)
    assert row is not None
    return row


async def _connection(db: AsyncSession, server_id: uuid.UUID, user_id: str) -> McpUserConnection | None:
    return (
        await db.execute(
            select(McpUserConnection)
            .where(McpUserConnection.server_id == server_id, McpUserConnection.user_id == user_id)
            .execution_options(populate_existing=True)
        )
    ).scalar_one_or_none()


async def _expire(db: AsyncSession, server_id: uuid.UUID, *, ago: timedelta = timedelta(minutes=1)) -> None:
    await db.execute(
        update(McpUserConnection)
        .where(McpUserConnection.server_id == server_id)
        .values(expires_at=datetime.now(UTC) - ago)
    )
    await db.commit()


# --------------------------------------------------------------------------- #
# The credential mode
# --------------------------------------------------------------------------- #


async def test_a_server_is_shared_unless_it_says_otherwise(async_db: AsyncSession, config: GatewayConfig) -> None:
    world = await _world(async_db, config, credential_mode="shared", authorization_token="shared-token")

    assert world.server.credential_mode == "shared"
    assert world.server.has_token is True
    assert (await _row(async_db, world.server.id)).credential_mode == "shared"
    with pytest.raises(McpConnectionModeError):
        await _service(async_db, config).start_connection(
            user=world.owner, workspace_id=world.workspace.id, server_id=world.server.id, return_to=None
        )


async def test_a_per_user_server_takes_no_shared_token_and_needs_https(
    async_db: AsyncSession, config: GatewayConfig
) -> None:
    with pytest.raises(ValueError, match="takes no authorization_token"):
        WorkspaceMcpServerCreate(
            name="x", url=PUBLIC_URL, credential_mode=McpCredentialMode.PER_USER, authorization_token="t"
        )
    with pytest.raises(ValueError, match="per_user server only"):
        WorkspaceMcpServerCreate(name="x", url=PUBLIC_URL, oauth_client_id="client")

    with pytest.raises(WorkspaceMcpServerUnsafeUrlError):
        await _world(async_db, config, url="http://93.184.216.34/mcp")


async def test_per_user_mode_needs_a_public_base_url_and_a_secret_key(
    async_db: AsyncSession, config: GatewayConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "public_base_url", None)
    with pytest.raises(McpConnectionSetupError, match="public_base_url"):
        await _world(async_db, config)
    await async_db.rollback()

    monkeypatch.setattr(config, "public_base_url", ORIGIN)
    monkeypatch.delenv("OTARI_SECRET_KEY")
    with pytest.raises(SecretBoxUnavailableTenancyError):
        await _world(async_db, config)


async def test_a_per_user_server_is_checked_with_the_callers_own_connection(
    async_db: AsyncSession, config: GatewayConfig, checks: list[ResolvedMcpServer]
) -> None:
    world = await _world(async_db, config)
    service = _service(async_db, config)

    assert world.server.health is None, "nobody has signed in yet, so there is nothing to check with"
    assert checks == []
    with pytest.raises(McpSignInRequiredError):
        await service.check_server(user=world.owner, workspace_id=world.workspace.id, server_id=world.server.id)
    with pytest.raises(McpConnectionSetupError):
        await service.check_connection(
            user=world.owner,
            workspace_id=world.workspace.id,
            request=WorkspaceMcpServerCheckRequest(url=PUBLIC_URL, server_id=world.server.id),
        )

    await _sign_in(async_db, config, world, world.owner)
    checked = await service.check_server(user=world.owner, workspace_id=world.workspace.id, server_id=world.server.id)

    assert checked.health is not None and checked.health.status == "healthy"
    assert checks[-1].authorization_token == "access-1"


# --------------------------------------------------------------------------- #
# Signing in
# --------------------------------------------------------------------------- #


async def test_a_sign_in_registers_a_client_once_and_stores_only_ciphertext(
    async_db: AsyncSession, config: GatewayConfig, wire: _Wire
) -> None:
    world = await _world(async_db, config)
    service = _service(async_db, config)

    started = await service.start_connection(
        user=world.member, workspace_id=world.workspace.id, server_id=world.server.id, return_to="/tools/mcp-servers"
    )

    query = {key: value[0] for key, value in parse_qs(urlsplit(started.authorization_url).query).items()}
    assert started.authorization_url.startswith(f"{ISSUER}/authorize?")
    assert query["redirect_uri"] == CALLBACK
    assert query["client_id"] == "dynamic-1"
    assert query["code_challenge_method"] == "S256"
    assert query["resource"] == PUBLIC_URL
    flow = (await async_db.execute(select(McpOAuthFlow))).scalar_one()
    assert query["state"] not in (flow.state_hash, flow.encrypted_code_verifier)
    assert _mcp_oauth.code_challenge(decrypt_secret(flow.encrypted_code_verifier)) == query["code_challenge"]
    assert (flow.user_id, flow.identity_id, flow.return_to) == (
        str(world.member.id),
        world.member.id,
        "/tools/mcp-servers",
    )
    client = stored_oauth_client(await _row(async_db, world.server.id))
    assert client is not None and client.dynamic and client.redirect_uri == CALLBACK
    assert client.encrypted_secret is not None and decrypt_secret(client.encrypted_secret) == "dynamic-secret"

    done = await service.complete_connection(user=world.member, state=query["state"], code="the-code")

    assert (done.server_id, done.workspace_id, done.server_name, done.return_to) == (
        world.server.id,
        world.workspace.id,
        "drive",
        "/tools/mcp-servers",
    )
    assert wire.exchanges[0]["code"] == "the-code"
    assert wire.exchanges[0]["resource"] == PUBLIC_URL
    assert wire.exchanges[0]["client"] == McpOAuthClient("dynamic-1", "dynamic-secret", AUTH_METHOD_POST)
    connection = await _connection(async_db, world.server.id, str(world.member.id))
    assert connection is not None
    assert "access-1" not in connection.encrypted_access_token
    assert decrypt_secret(connection.encrypted_access_token) == "access-1"
    assert connection.encrypted_refresh_token is not None
    assert decrypt_secret(connection.encrypted_refresh_token) == "refresh-1"
    assert connection.expires_at is not None and connection.expires_at > datetime.now(UTC) + timedelta(minutes=50)
    assert (await async_db.execute(select(McpOAuthFlow))).first() is None

    await _sign_in(async_db, config, world, world.owner)
    assert wire.registrations == [CALLBACK], "the second person signs in through the client the first registered"


async def test_the_public_shape_never_carries_a_secret(async_db: AsyncSession, config: GatewayConfig) -> None:
    world = await _world(async_db, config, oauth_client_id="my-client", oauth_client_secret="my-secret")
    service = _service(async_db, config)
    await _sign_in(async_db, config, world, world.owner)

    server = await service.get_server(user=world.owner, workspace_id=world.workspace.id, server_id=world.server.id)
    mine = await service.my_connections(user=world.owner, workspace_id=world.workspace.id)
    theirs = await service.server_connections(
        user=world.owner, workspace_id=world.workspace.id, server_id=world.server.id
    )

    assert (server.oauth_client_id, server.has_oauth_client_secret) == ("my-client", True)
    assert [connection.server_id for connection in mine.data] == [world.server.id]
    assert mine.data[0].renewable is True
    dumped = server.model_dump_json() + mine.model_dump_json() + theirs.model_dump_json()
    for secret in ("my-secret", "access-1", "refresh-1"):
        assert secret not in dumped


async def test_a_managers_own_client_is_used_and_none_is_registered(
    async_db: AsyncSession, config: GatewayConfig, wire: _Wire
) -> None:
    world = await _world(async_db, config, oauth_client_id="my-client", oauth_client_secret="my-secret")

    await _sign_in(async_db, config, world, world.member)

    assert wire.registrations == []
    assert wire.exchanges[0]["client"] == McpOAuthClient("my-client", "my-secret", AUTH_METHOD_POST)


async def test_a_server_that_registers_no_client_asks_for_a_client_id(
    async_db: AsyncSession, config: GatewayConfig, wire: _Wire, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = await _world(async_db, config)
    service = _service(async_db, config)

    async def refuse(registration_endpoint: str, redirect_uri: str) -> McpOAuthClient:
        raise McpOAuthRegistrationError("refused")

    monkeypatch.setattr(_mcp_oauth, "register_client", refuse)
    with pytest.raises(McpConnectionSetupError, match="OAuth client ID"):
        await service.start_connection(
            user=world.owner, workspace_id=world.workspace.id, server_id=world.server.id, return_to=None
        )

    wire.metadata = McpOAuthMetadata(authorization_endpoint=f"{ISSUER}/authorize", token_endpoint=f"{ISSUER}/token")
    with pytest.raises(McpConnectionSetupError, match="OAuth client ID"):
        await service.start_connection(
            user=world.owner, workspace_id=world.workspace.id, server_id=world.server.id, return_to=None
        )
    assert (await async_db.execute(select(McpOAuthFlow))).first() is None


async def test_a_state_finishes_one_sign_in_only(async_db: AsyncSession, config: GatewayConfig, wire: _Wire) -> None:
    world = await _world(async_db, config)
    service = _service(async_db, config)
    started = await service.start_connection(
        user=world.owner, workspace_id=world.workspace.id, server_id=world.server.id, return_to=None
    )
    state = _state(started.authorization_url)

    await service.complete_connection(user=world.owner, state=state, code="code")
    with pytest.raises(McpConnectionFlowError, match="unknown or was already finished"):
        await service.complete_connection(user=world.owner, state=state, code="code")
    with pytest.raises(McpConnectionFlowError):
        await service.complete_connection(user=world.owner, state="never-issued", code="code")

    assert len(wire.exchanges) == 1


async def test_a_sign_in_finishes_only_for_the_person_who_started_it(
    async_db: AsyncSession, config: GatewayConfig, wire: _Wire
) -> None:
    world = await _world(async_db, config)
    service = _service(async_db, config)
    started = await service.start_connection(
        user=world.member, workspace_id=world.workspace.id, server_id=world.server.id, return_to=None
    )
    state = _state(started.authorization_url)

    with pytest.raises(McpConnectionFlowError, match="another person"):
        await service.complete_connection(user=world.owner, state=state, code="code")
    with pytest.raises(McpConnectionFlowError, match="unknown or was already finished"):
        await service.complete_connection(user=world.member, state=state, code="code")

    # With no session at all, as the bare callback has: the state alone finishes nothing a session started.
    started = await service.start_connection(
        user=world.member, workspace_id=world.workspace.id, server_id=world.server.id, return_to=None
    )
    with pytest.raises(McpConnectionFlowError, match="another person"):
        await _connections(async_db, config).complete(state=_state(started.authorization_url), code="code")

    assert wire.exchanges == []
    assert await _connection(async_db, world.server.id, str(world.owner.id)) is None
    assert await _connection(async_db, world.server.id, str(world.member.id)) is None


async def test_an_expired_or_refused_sign_in_connects_nothing(async_db: AsyncSession, config: GatewayConfig) -> None:
    world = await _world(async_db, config)
    service = _service(async_db, config)
    started = await service.start_connection(
        user=world.owner, workspace_id=world.workspace.id, server_id=world.server.id, return_to=None
    )
    await async_db.execute(update(McpOAuthFlow).values(expires_at=datetime.now(UTC) - timedelta(seconds=1)))
    await async_db.commit()

    with pytest.raises(McpConnectionFlowError, match="too long"):
        await service.complete_connection(user=world.owner, state=_state(started.authorization_url), code="code")

    started = await service.start_connection(
        user=world.owner, workspace_id=world.workspace.id, server_id=world.server.id, return_to=None
    )
    with pytest.raises(McpConnectionUpstreamError):
        await service.complete_connection(user=world.owner, state=_state(started.authorization_url), code="refused")

    assert await _connection(async_db, world.server.id, str(world.owner.id)) is None


async def test_a_sign_in_started_outside_a_session_finishes_on_its_state_alone(
    async_db: AsyncSession, config: GatewayConfig
) -> None:
    """The entry point a later change mints a sign-in link through: a user, a server, and nothing of a session."""
    world = await _world(async_db, config)
    async_db.add(BillingUser(user_id="alice"))
    await async_db.commit()
    connections = _connections(async_db, config)

    url = await connections.start(server_id=world.server.id, user_id="alice", return_to="/playground")
    landing = await connections.landing(_state(url))
    done = await connections.complete(state=_state(url), code="code")

    assert (landing.known, landing.needs_dashboard) == (True, False)
    assert done.return_to == "/playground"
    assert await _connection(async_db, world.server.id, "alice") is not None
    assert (await connections.landing("never-issued")).known is False
    with pytest.raises(McpConnectionSetupError, match="default"):
        await connections.start(server_id=world.server.id, user_id="default")


@pytest.mark.parametrize(
    "return_to", ["https://evil.example/", "//evil.example/", "tools", "/\\evil.example", "/a\nb", "/" + "a" * 2048]
)
async def test_a_return_link_outside_the_dashboard_is_refused(return_to: str) -> None:
    with pytest.raises(McpConnectionSetupError):
        checked_return_to(return_to)


async def test_a_dashboard_path_is_kept_as_the_return_link() -> None:
    assert checked_return_to("/tools/mcp-servers/1?tab=access") == "/tools/mcp-servers/1?tab=access"
    assert checked_return_to(None) is None
    assert checked_return_to("") is None


async def test_only_a_member_who_sees_the_workspace_may_sign_in(async_db: AsyncSession, config: GatewayConfig) -> None:
    world = await _world(async_db, config)
    other = await OrganizationRepository(async_db).create_organization(
        name="Other", slug="other", created_by_user_id=None
    )
    outsider = await _member(async_db, other, role="owner", name="Outsider")
    service = _service(async_db, config)

    with pytest.raises(Exception) as raised:  # noqa: PT011
        await service.start_connection(
            user=outsider, workspace_id=world.workspace.id, server_id=world.server.id, return_to=None
        )
    assert getattr(raised.value, "status_code", None) == 404
    with pytest.raises(NotAuthorizedError):
        await service.server_connections(user=world.member, workspace_id=world.workspace.id, server_id=world.server.id)
    with pytest.raises(NotAuthorizedError):
        await service.remove_connection(
            user=world.member, workspace_id=world.workspace.id, server_id=world.server.id, user_id=str(world.owner.id)
        )


# --------------------------------------------------------------------------- #
# What a request carries
# --------------------------------------------------------------------------- #


def _access(user_id: str | None) -> McpAccess:
    return McpAccess(user_id=user_id)


async def test_a_request_carries_its_users_own_token(async_db: AsyncSession, config: GatewayConfig) -> None:
    world = await _world(async_db, config)
    await _sign_in(async_db, config, world, world.member)
    servers = WorkspaceMcpServers(async_db)

    [resolved] = await servers.resolve_many(world.workspace.id, [world.server.id], _access(str(world.member.id)))
    one = await servers.resolve_one(world.workspace.id, world.server.id, _access(str(world.member.id)))

    assert resolved.authorization_token == "access-1"
    assert one is not None and one.authorization_token == "access-1"


async def test_a_user_who_has_not_signed_in_gets_a_result_of_its_own_naming_the_server(
    async_db: AsyncSession, config: GatewayConfig
) -> None:
    world = await _world(async_db, config)
    await _sign_in(async_db, config, world, world.member)
    servers = WorkspaceMcpServers(async_db)
    owner_id = str(world.owner.id)

    with pytest.raises(McpSignInRequiredError) as many:
        await servers.resolve_many(world.workspace.id, [world.server.id], _access(owner_id))
    with pytest.raises(McpSignInRequiredError) as one:
        await servers.resolve_one(world.workspace.id, world.server.id, _access(owner_id))

    for raised in (many.value, one.value):
        assert (raised.mcp_server_id, raised.name, raised.user_id) == (world.server.id, "drive", owner_id)
        assert raised.error_code == "mcp_sign_in_required"
        assert raised.status_code == 403
        assert "drive" in raised.message
    listed = await servers.list_reachable(world.workspace.id, _access(owner_id))
    assert [(server.id, server.sign_in_required) for server in listed] == [(world.server.id, True)]
    listed = await servers.list_reachable(world.workspace.id, _access(str(world.member.id)))
    assert [(server.id, server.sign_in_required) for server in listed] == [(world.server.id, False)]


async def test_a_key_with_no_user_cannot_use_a_per_user_server(async_db: AsyncSession, config: GatewayConfig) -> None:
    world = await _world(async_db, config)
    await _sign_in(async_db, config, world, world.member)
    servers = WorkspaceMcpServers(async_db)

    for access in (_access(None), McpAccess(unrestricted=True)):
        with pytest.raises(WorkspaceMcpServerNotFoundError):
            await servers.resolve_many(world.workspace.id, [world.server.id], access)
        assert await servers.resolve_one(world.workspace.id, world.server.id, access) is None
        assert await servers.list_reachable(world.workspace.id, access) == []


async def test_a_caller_with_no_access_is_told_the_server_does_not_exist_signed_in_or_not(
    async_db: AsyncSession, config: GatewayConfig
) -> None:
    world = await _world(async_db, config, everyone_tools=[])
    servers = WorkspaceMcpServers(async_db)

    for user in (world.owner, world.member):
        with pytest.raises(WorkspaceMcpServerNotFoundError) as raised:
            await servers.resolve_many(world.workspace.id, [world.server.id], _access(str(user.id)))
        assert not isinstance(raised.value, McpSignInRequiredError)
        with pytest.raises(WorkspaceMcpServerNotFoundError):
            await servers.resolve_one(world.workspace.id, world.server.id, _access(str(user.id)))
        assert await servers.list_reachable(world.workspace.id, _access(str(user.id))) == []


async def test_a_shared_server_is_untouched_by_who_is_calling(async_db: AsyncSession, config: GatewayConfig) -> None:
    world = await _world(async_db, config, credential_mode="shared", authorization_token="shared-token")
    servers = WorkspaceMcpServers(async_db)

    for access in (_access(None), _access("anyone"), McpAccess(unrestricted=True)):
        [resolved] = await servers.resolve_many(world.workspace.id, [world.server.id], access)
        assert resolved.authorization_token == "shared-token"
        [listed] = await servers.list_reachable(world.workspace.id, access)
        assert listed.sign_in_required is False


# --------------------------------------------------------------------------- #
# Renewal
# --------------------------------------------------------------------------- #


async def test_an_expiring_token_is_renewed_and_the_rotated_refresh_token_kept(
    async_db: AsyncSession, config: GatewayConfig, wire: _Wire, tokens: McpUserTokens
) -> None:
    world = await _world(async_db, config)
    await _sign_in(async_db, config, world, world.member)
    user_id = str(world.member.id)
    row = await _row(async_db, world.server.id)
    servers = WorkspaceMcpServers(async_db, tokens)
    before = await servers.resolve_one(world.workspace.id, world.server.id, _access(user_id))

    assert await tokens.token_for(row, user_id) == "access-1"
    assert wire.refreshes == [], "a token well inside its lifetime is used as it is"

    await _expire(async_db, world.server.id, ago=timedelta(minutes=-2))
    assert await tokens.token_for(row, user_id) == "renewed-1"

    assert wire.refreshes == [(McpOAuthClient("dynamic-1", "dynamic-secret", AUTH_METHOD_POST), "refresh-1")]
    connection = await _connection(async_db, world.server.id, user_id)
    assert connection is not None and connection.encrypted_refresh_token is not None
    assert decrypt_secret(connection.encrypted_access_token) == "renewed-1"
    assert decrypt_secret(connection.encrypted_refresh_token) == "rotated-1"
    assert connection.expires_at is not None and connection.expires_at > datetime.now(UTC) + timedelta(minutes=50)
    after = await servers.resolve_one(world.workspace.id, world.server.id, _access(user_id))
    assert before is not None and after is not None
    assert after.authorization_token == "renewed-1"
    assert after.revision == before.revision, "a renewal is not a change to the server's configuration"


async def test_concurrent_requests_spend_a_refresh_token_once(
    async_db: AsyncSession, config: GatewayConfig, wire: _Wire, tokens: McpUserTokens, postgres_url: str
) -> None:
    world = await _world(async_db, config)
    await _sign_in(async_db, config, world, world.member)
    await _expire(async_db, world.server.id)
    user_id = str(world.member.id)
    row = await _row(async_db, world.server.id)
    wire.refresh_gate = asyncio.Event()
    # A second reader on a session of its own, as a second request has.
    engine = create_async_engine(_to_async_url(postgres_url))
    try:
        async with async_sessionmaker(engine, expire_on_commit=False)() as other_session:
            other = McpUserTokens(McpUserConnectionRepository(other_session), unit_of_work=tokens._unit_of_work)
            first = asyncio.create_task(tokens.token_for(row, user_id))
            second = asyncio.create_task(other.token_for(row, user_id))
            while not wire.refreshes:
                await asyncio.sleep(0.01)
            await asyncio.sleep(0.05)
            wire.refresh_gate.set()
            results = await asyncio.gather(first, second)
    finally:
        await engine.dispose()

    assert list(results) == ["renewed-1", "renewed-1"]
    assert len(wire.refreshes) == 1


async def test_a_dead_grant_ends_the_connection_and_asks_for_a_new_sign_in(
    async_db: AsyncSession, config: GatewayConfig, wire: _Wire, tokens: McpUserTokens
) -> None:
    world = await _world(async_db, config)
    await _sign_in(async_db, config, world, world.member)
    await _expire(async_db, world.server.id)
    user_id = str(world.member.id)
    wire.refresh_error = McpOAuthTokenError("refused", status_code=400, error_code="invalid_grant")

    with pytest.raises(McpSignInRequiredError):
        await tokens.token_for(await _row(async_db, world.server.id), user_id)

    assert await _connection(async_db, world.server.id, user_id) is None
    with pytest.raises(McpSignInRequiredError):
        await tokens.token_for(await _row(async_db, world.server.id), user_id)
    assert len(wire.refreshes) == 1, "a connection that is gone is not renewed again"


async def test_a_renewal_that_fails_for_now_keeps_the_connection(
    async_db: AsyncSession, config: GatewayConfig, wire: _Wire, tokens: McpUserTokens
) -> None:
    world = await _world(async_db, config)
    await _sign_in(async_db, config, world, world.member)
    user_id = str(world.member.id)
    row = await _row(async_db, world.server.id)
    wire.refresh_error = McpOAuthTokenError("unavailable", status_code=503)

    await _expire(async_db, world.server.id, ago=timedelta(minutes=-2))
    assert await tokens.token_for(row, user_id) == "access-1", "still inside its lifetime, so it is used"

    await _expire(async_db, world.server.id)
    with pytest.raises(McpConnectionUnavailableError):
        await tokens.token_for(row, user_id)

    assert await _connection(async_db, world.server.id, user_id) is not None
    wire.refresh_error = None
    assert await tokens.token_for(row, user_id) == "renewed-3"


async def test_an_expired_token_with_no_refresh_token_asks_for_a_new_sign_in(
    async_db: AsyncSession, config: GatewayConfig, wire: _Wire, tokens: McpUserTokens
) -> None:
    world = await _world(async_db, config)
    wire.issue = McpOAuthTokens(access_token="access-only", refresh_token=None, expires_in=None)
    await _sign_in(async_db, config, world, world.member)
    user_id = str(world.member.id)
    row = await _row(async_db, world.server.id)

    assert await tokens.token_for(row, user_id) == "access-only", "no stated lifetime: used until refused"
    await _expire(async_db, world.server.id)
    with pytest.raises(McpSignInRequiredError):
        await tokens.token_for(row, user_id)

    assert wire.refreshes == []
    assert await _connection(async_db, world.server.id, user_id) is None


# --------------------------------------------------------------------------- #
# Ending a connection
# --------------------------------------------------------------------------- #


async def test_a_user_disconnects_and_the_tokens_are_revoked(
    async_db: AsyncSession, config: GatewayConfig, wire: _Wire
) -> None:
    world = await _world(async_db, config)
    await _sign_in(async_db, config, world, world.member)
    await _sign_in(async_db, config, world, world.owner)
    service = _service(async_db, config)

    await service.disconnect(user=world.member, workspace_id=world.workspace.id, server_id=world.server.id)

    assert wire.revoked == ["refresh-1"]
    assert await _connection(async_db, world.server.id, str(world.member.id)) is None
    assert await _connection(async_db, world.server.id, str(world.owner.id)) is not None
    with pytest.raises(McpConnectionNotFoundError):
        await service.disconnect(user=world.member, workspace_id=world.workspace.id, server_id=world.server.id)


async def test_a_manager_sees_who_is_connected_and_removes_one(
    async_db: AsyncSession, config: GatewayConfig, wire: _Wire
) -> None:
    world = await _world(async_db, config)
    await _sign_in(async_db, config, world, world.member)
    async_db.add(BillingUser(user_id="service-user"))
    await async_db.commit()
    connections = _connections(async_db, config)
    url = await connections.start(server_id=world.server.id, user_id="service-user")
    await connections.complete(state=_state(url), code="code")
    service = _service(async_db, config)

    listed = await service.server_connections(
        user=world.owner, workspace_id=world.workspace.id, server_id=world.server.id
    )
    assert [(entry.user_id, entry.full_name) for entry in listed.data] == [
        (str(world.member.id), "Member"),
        ("service-user", None),
    ]

    await service.remove_connection(
        user=world.owner, workspace_id=world.workspace.id, server_id=world.server.id, user_id=str(world.member.id)
    )

    assert await _connection(async_db, world.server.id, str(world.member.id)) is None
    assert await _connection(async_db, world.server.id, "service-user") is not None
    assert wire.revoked == ["refresh-1"]


async def test_deleting_the_server_ends_every_connection(async_db: AsyncSession, config: GatewayConfig) -> None:
    world = await _world(async_db, config)
    await _sign_in(async_db, config, world, world.member)
    service = _service(async_db, config)
    await service.start_connection(
        user=world.owner, workspace_id=world.workspace.id, server_id=world.server.id, return_to=None
    )

    await service.delete_server(user=world.owner, workspace_id=world.workspace.id, server_id=world.server.id)

    assert (await async_db.execute(select(McpUserConnection))).first() is None
    assert (await async_db.execute(select(McpOAuthFlow))).first() is None


async def test_deleting_a_user_ends_their_connections_and_pending_sign_ins(
    async_db: AsyncSession, config: GatewayConfig
) -> None:
    world = await _world(async_db, config)
    await _sign_in(async_db, config, world, world.member)
    await _sign_in(async_db, config, world, world.owner)
    service = _service(async_db, config)
    await service.start_connection(
        user=world.member, workspace_id=world.workspace.id, server_id=world.server.id, return_to=None
    )

    await revoke_user_mcp_grants(async_db, str(world.member.id))
    await async_db.commit()

    assert await _connection(async_db, world.server.id, str(world.member.id)) is None
    assert await _connection(async_db, world.server.id, str(world.owner.id)) is not None
    assert (await async_db.execute(select(McpOAuthFlow))).first() is None


async def test_leaving_the_workspace_ends_a_members_connections_to_its_servers(
    async_db: AsyncSession, config: GatewayConfig
) -> None:
    world = await _world(async_db, config)
    await _sign_in(async_db, config, world, world.member)
    await _sign_in(async_db, config, world, world.owner)
    await _service(async_db, config).start_connection(
        user=world.member, workspace_id=world.workspace.id, server_id=world.server.id, return_to=None
    )
    uow = UnitOfWork(async_db)

    async with uow:
        await McpGrantMembershipListener(uow).member_lost_workspaces(world.member.id, [world.workspace.id])

    assert await _connection(async_db, world.server.id, str(world.member.id)) is None
    assert await _connection(async_db, world.server.id, str(world.owner.id)) is not None
    assert (await async_db.execute(select(McpOAuthFlow))).first() is None


@pytest.mark.parametrize(
    "change",
    [
        WorkspaceMcpServerUpdate(url=OTHER_PUBLIC_URL),
        WorkspaceMcpServerUpdate(credential_mode=McpCredentialMode.SHARED),
        WorkspaceMcpServerUpdate(oauth_client_id="another-client"),
    ],
    ids=["url", "mode", "client"],
)
async def test_a_change_the_tokens_do_not_fit_ends_every_connection(
    async_db: AsyncSession, config: GatewayConfig, change: WorkspaceMcpServerUpdate
) -> None:
    world = await _world(async_db, config)
    await _sign_in(async_db, config, world, world.member)
    service = _service(async_db, config)

    await service.update_server(
        user=world.owner, workspace_id=world.workspace.id, server_id=world.server.id, request=change
    )

    assert await _connection(async_db, world.server.id, str(world.member.id)) is None
    client = stored_oauth_client(await _row(async_db, world.server.id))
    assert client is None or not client.dynamic, "the client Otari registered went with the tokens"


async def test_a_change_that_leaves_the_tokens_valid_keeps_the_connections(
    async_db: AsyncSession, config: GatewayConfig
) -> None:
    world = await _world(async_db, config, oauth_client_id="my-client", oauth_client_secret="old-secret")
    await _sign_in(async_db, config, world, world.member)
    service = _service(async_db, config)

    for change in (
        WorkspaceMcpServerUpdate(name="renamed", enabled=False),
        WorkspaceMcpServerUpdate(oauth_client_secret="new-secret"),
        WorkspaceMcpServerUpdate(oauth_client_id="my-client"),
    ):
        await service.update_server(
            user=world.owner, workspace_id=world.workspace.id, server_id=world.server.id, request=change
        )

    assert await _connection(async_db, world.server.id, str(world.member.id)) is not None
    client = stored_oauth_client(await _row(async_db, world.server.id))
    assert client is not None and client.encrypted_secret is not None
    assert (client.client_id, decrypt_secret(client.encrypted_secret)) == ("my-client", "new-secret")


async def test_moving_a_shared_server_to_per_user_drops_its_shared_token(
    async_db: AsyncSession, config: GatewayConfig
) -> None:
    world = await _world(async_db, config, credential_mode="shared", authorization_token="shared-token")
    service = _service(async_db, config)
    workspace_id, owner_id = world.workspace.id, str(world.owner.id)

    with pytest.raises(McpConnectionSetupError, match="takes no authorization_token"):
        await service.update_server(
            user=world.owner,
            workspace_id=workspace_id,
            server_id=world.server.id,
            request=WorkspaceMcpServerUpdate(credential_mode=McpCredentialMode.PER_USER, authorization_token="another"),
        )
    updated = await service.update_server(
        user=world.owner,
        workspace_id=workspace_id,
        server_id=world.server.id,
        request=WorkspaceMcpServerUpdate(credential_mode=McpCredentialMode.PER_USER),
    )

    assert (updated.credential_mode, updated.has_token) == ("per_user", False)
    assert (await _row(async_db, world.server.id)).encrypted_token is None
    with pytest.raises(McpSignInRequiredError):
        await WorkspaceMcpServers(async_db).resolve_many(workspace_id, [world.server.id], _access(owner_id))


async def test_a_connection_whose_client_the_server_no_longer_holds_cannot_be_renewed(
    async_db: AsyncSession, config: GatewayConfig, wire: _Wire, tokens: McpUserTokens
) -> None:
    """A client change ends connections, and one that survived it is still not renewed with another client."""
    world = await _world(async_db, config)
    await _sign_in(async_db, config, world, world.member)
    await _expire(async_db, world.server.id)
    row = await _row(async_db, world.server.id)
    row.oauth_client = {"client_id": "another", "encrypted_secret": encrypt_secret("s"), "dynamic": False}
    await async_db.commit()

    with pytest.raises(McpSignInRequiredError):
        await tokens.token_for(row, str(world.member.id))

    assert wire.refreshes == []
