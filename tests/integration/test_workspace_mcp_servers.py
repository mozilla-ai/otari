"""Workspace-scoped MCP servers: CRUD, encryption at rest, and request-path resolution.

Exercised at the service layer, matching `test_org_provider_keys.py` and
`test_workspace_member_budget_policies.py`: the API can only ever act as the
one bootstrap operator identity a standalone deployment has, who is always an
owner, so the authorization rules that matter (a plain member refused, another
organization's workspace invisible) are only reachable by calling the service
with identities built at whatever role a case needs.

URLs here are IP literals in public ranges, or are rejected before any lookup
happens (a bad scheme, a token on ``http://``). ``validate_mcp_url`` resolves a
hostname through DNS, so a test naming one would pass or fail on whether the
runner has egress.
"""

import time
import uuid
from collections.abc import Iterator

import pytest
from fastapi import HTTPException, Response
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.routes import chat
from gateway.api.routes._pipeline import RequestContext, prepare_gateway_tools
from gateway.api.routes.chat import ChatCompletionRequest
from gateway.core.config import GatewayConfig
from gateway.models.entities import WorkspaceMcpServer
from gateway.models.mcp import MAX_MCP_SERVER_IDS, McpServerConfig
from gateway.models.tenancy import Organization, User, Workspace
from gateway.repositories.tenancy import (
    OrganizationMemberRepository,
    OrganizationRepository,
    UserRepository,
    WorkspaceMemberRepository,
    WorkspaceRepository,
)
from gateway.services.secret_box import decrypt_secret, generate_secret_key
from gateway.services.tenancy.errors import (
    NotAuthorizedError,
    WorkspaceMcpServerAlreadyExistsError,
    WorkspaceMcpServerLimitReachedError,
    WorkspaceMcpServerNotFoundError,
    WorkspaceMcpServerUnsafeUrlError,
    WorkspaceNotFoundError,
)
from gateway.services.tenancy.workspace_mcp_server_service import (
    MAX_ALLOWED_TOOLS,
    MAX_MCP_SERVERS_PER_WORKSPACE,
    WorkspaceMcpServerCreate,
    WorkspaceMcpServerService,
    WorkspaceMcpServerUpdate,
    resolve_workspace_mcp_servers,
)

pytestmark = pytest.mark.asyncio

# A public IP literal, so the safety check never reaches a DNS resolver.
PUBLIC_URL = "https://93.184.216.34/mcp"
OTHER_PUBLIC_URL = "https://93.184.216.35/mcp"


async def _organization(db: AsyncSession, *, slug: str = "acme") -> Organization:
    return await OrganizationRepository(db).create_organization(name=slug.title(), slug=slug, created_by_user_id=None)


async def _member(db: AsyncSession, organization: Organization, *, role: str, full_name: str) -> User:
    user = await UserRepository(db).create_local_identity(
        full_name=full_name,
        active_organization_id=organization.id,
    )
    await OrganizationMemberRepository(db).create_membership(
        organization_id=organization.id, user_id=user.id, role=role
    )
    return user


async def _workspace(
    db: AsyncSession, organization: Organization, *, name: str = "Default", owner: User | None = None
) -> Workspace:
    workspace = await WorkspaceRepository(db).create_workspace(
        name=name, organization_id=organization.id, created_by_user_id=owner.id if owner else None
    )
    if owner is not None:
        await WorkspaceMemberRepository(db).create(workspace_id=workspace.id, user_id=owner.id, role="owner")
    return workspace


@pytest.fixture(autouse=True)
def _secret_key(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    yield


def _create(**overrides: object) -> WorkspaceMcpServerCreate:
    fields: dict[str, object] = {"name": "github", "url": PUBLIC_URL}
    fields.update(overrides)
    return WorkspaceMcpServerCreate(**fields)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# CRUD
# --------------------------------------------------------------------------- #


async def test_crud_round_trip(async_db: AsyncSession) -> None:
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    workspace = await _workspace(async_db, organization, owner=owner)
    service = WorkspaceMcpServerService(async_db)

    created = await service.create_server(
        user=owner,
        workspace_id=workspace.id,
        request=_create(purpose_hint="Repository lookups", allowed_tools=["list_issues"]),
    )
    assert created.workspace_id == workspace.id
    assert created.has_token is False
    assert created.enabled is True
    assert created.allowed_tools == ["list_issues"]

    listed = await service.list_servers(user=owner, workspace_id=workspace.id)
    assert listed.count == 1
    assert [server.id for server in listed.data] == [created.id]

    updated = await service.update_server(
        user=owner,
        workspace_id=workspace.id,
        server_id=created.id,
        request=WorkspaceMcpServerUpdate(name="github-enterprise", enabled=False),
    )
    assert updated.name == "github-enterprise"
    assert updated.enabled is False
    assert updated.url == PUBLIC_URL, "an omitted field is left in place"

    await service.delete_server(user=owner, workspace_id=workspace.id, server_id=created.id)
    assert (await service.list_servers(user=owner, workspace_id=workspace.id)).count == 0


async def test_unknown_server_id_is_not_found(async_db: AsyncSession) -> None:
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    workspace = await _workspace(async_db, organization, owner=owner)
    service = WorkspaceMcpServerService(async_db)

    with pytest.raises(WorkspaceMcpServerNotFoundError):
        await service.delete_server(user=owner, workspace_id=workspace.id, server_id=uuid.uuid4())


async def test_another_workspaces_server_is_not_found(async_db: AsyncSession) -> None:
    """A server id is scoped to its workspace on every read, not only on the list."""
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    first = await _workspace(async_db, organization, name="First", owner=owner)
    second = await _workspace(async_db, organization, name="Second", owner=owner)
    service = WorkspaceMcpServerService(async_db)

    created = await service.create_server(user=owner, workspace_id=first.id, request=_create())

    with pytest.raises(WorkspaceMcpServerNotFoundError):
        await service.update_server(
            user=owner,
            workspace_id=second.id,
            server_id=created.id,
            request=WorkspaceMcpServerUpdate(name="renamed"),
        )


async def test_server_count_is_capped_per_workspace(async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch) -> None:
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    workspace = await _workspace(async_db, organization, owner=owner)
    service = WorkspaceMcpServerService(async_db)

    for index in range(3):
        await service.create_server(user=owner, workspace_id=workspace.id, request=_create(name=f"server-{index}"))

    monkeypatch.setattr(
        "gateway.services.tenancy.workspace_mcp_server_service.MAX_MCP_SERVERS_PER_WORKSPACE",
        3,
    )
    with pytest.raises(WorkspaceMcpServerLimitReachedError):
        await service.create_server(user=owner, workspace_id=workspace.id, request=_create(name="server-4"))

    assert MAX_MCP_SERVERS_PER_WORKSPACE >= 3, "the shipped cap is not what this test pins"


async def test_creating_a_server_locks_the_workspace(async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch) -> None:
    """The cap is a read-decide-write, so it takes the workspace row lock before counting.

    A count that is not serialized against a concurrent create is a check two
    callers can both pass at the ceiling. Asserted through the lock rather than
    by racing two sessions, because the race is precisely what the lock makes
    unobservable.
    """
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    workspace = await _workspace(async_db, organization, owner=owner)

    locked: list[uuid.UUID] = []
    original = WorkspaceRepository.lock

    async def recording_lock(self: WorkspaceRepository, workspace_id: uuid.UUID) -> None:
        locked.append(workspace_id)
        await original(self, workspace_id)

    monkeypatch.setattr(WorkspaceRepository, "lock", recording_lock)
    await WorkspaceMcpServerService(async_db).create_server(
        user=owner, workspace_id=workspace.id, request=_create()
    )

    assert locked == [workspace.id]


# --------------------------------------------------------------------------- #
# Duplicate names
# --------------------------------------------------------------------------- #


async def test_duplicate_name_in_one_workspace_is_rejected(async_db: AsyncSession) -> None:
    """otari#658's third Definition-of-Done item: rejected, never silently collapsed."""
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    workspace = await _workspace(async_db, organization, owner=owner)
    # Read before the refused create: the rollback that maps the IntegrityError
    # expires every loaded instance, and reading one back is synchronous IO
    # this session cannot do.
    workspace_id = workspace.id
    service = WorkspaceMcpServerService(async_db)

    await service.create_server(user=owner, workspace_id=workspace_id, request=_create(name="github"))
    with pytest.raises(WorkspaceMcpServerAlreadyExistsError):
        await service.create_server(
            user=owner, workspace_id=workspace_id, request=_create(name="github", url=OTHER_PUBLIC_URL)
        )

    await async_db.refresh(owner)
    listed = await service.list_servers(user=owner, workspace_id=workspace_id)
    assert listed.count == 1
    assert listed.data[0].url == PUBLIC_URL, "the first server survives the refused create untouched"


async def test_rename_onto_an_existing_name_is_rejected(async_db: AsyncSession) -> None:
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    workspace = await _workspace(async_db, organization, owner=owner)
    workspace_id = workspace.id  # see the note in the create-side test above
    service = WorkspaceMcpServerService(async_db)

    await service.create_server(user=owner, workspace_id=workspace_id, request=_create(name="github"))
    second = await service.create_server(
        user=owner, workspace_id=workspace_id, request=_create(name="jira", url=OTHER_PUBLIC_URL)
    )

    with pytest.raises(WorkspaceMcpServerAlreadyExistsError):
        await service.update_server(
            user=owner,
            workspace_id=workspace_id,
            server_id=second.id,
            request=WorkspaceMcpServerUpdate(name="github"),
        )

    await async_db.refresh(owner)
    listed = await service.list_servers(user=owner, workspace_id=workspace_id)
    assert sorted(server.name for server in listed.data) == ["github", "jira"]


async def test_the_same_name_in_two_workspaces_is_fine(async_db: AsyncSession) -> None:
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    first = await _workspace(async_db, organization, name="First", owner=owner)
    second = await _workspace(async_db, organization, name="Second", owner=owner)
    service = WorkspaceMcpServerService(async_db)

    await service.create_server(user=owner, workspace_id=first.id, request=_create(name="github"))
    await service.create_server(user=owner, workspace_id=second.id, request=_create(name="github"))

    assert (await service.list_servers(user=owner, workspace_id=first.id)).count == 1
    assert (await service.list_servers(user=owner, workspace_id=second.id)).count == 1


# --------------------------------------------------------------------------- #
# The token: encrypted at rest, never returned
# --------------------------------------------------------------------------- #


async def test_token_is_encrypted_at_rest_and_never_returned(async_db: AsyncSession) -> None:
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    workspace = await _workspace(async_db, organization, owner=owner)
    service = WorkspaceMcpServerService(async_db)

    created = await service.create_server(
        user=owner, workspace_id=workspace.id, request=_create(authorization_token="ghp_secret_value")
    )
    assert created.has_token is True
    assert "ghp_secret_value" not in created.model_dump_json()

    listed = await service.list_servers(user=owner, workspace_id=workspace.id)
    assert "ghp_secret_value" not in listed.model_dump_json()

    stored = (
        await async_db.execute(select(WorkspaceMcpServer).where(WorkspaceMcpServer.id == created.id))
    ).scalar_one()
    assert stored.encrypted_token is not None
    assert stored.encrypted_token != "ghp_secret_value"
    assert decrypt_secret(stored.encrypted_token) == "ghp_secret_value"


async def test_token_update_semantics(async_db: AsyncSession) -> None:
    """Omitted leaves it, an empty string clears it, a value rotates it."""
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    workspace = await _workspace(async_db, organization, owner=owner)
    service = WorkspaceMcpServerService(async_db)

    created = await service.create_server(
        user=owner, workspace_id=workspace.id, request=_create(authorization_token="first")
    )

    unchanged = await service.update_server(
        user=owner,
        workspace_id=workspace.id,
        server_id=created.id,
        request=WorkspaceMcpServerUpdate(purpose_hint="Now with a hint"),
    )
    assert unchanged.has_token is True
    stored = await async_db.get(WorkspaceMcpServer, created.id)
    assert stored is not None and stored.encrypted_token is not None
    assert decrypt_secret(stored.encrypted_token) == "first"

    await service.update_server(
        user=owner,
        workspace_id=workspace.id,
        server_id=created.id,
        request=WorkspaceMcpServerUpdate(authorization_token="second"),
    )
    stored = await async_db.get(WorkspaceMcpServer, created.id)
    assert stored is not None and stored.encrypted_token is not None
    assert decrypt_secret(stored.encrypted_token) == "second"

    cleared = await service.update_server(
        user=owner,
        workspace_id=workspace.id,
        server_id=created.id,
        request=WorkspaceMcpServerUpdate(authorization_token=""),
    )
    assert cleared.has_token is False
    stored = await async_db.get(WorkspaceMcpServer, created.id)
    assert stored is not None and stored.encrypted_token is None


# --------------------------------------------------------------------------- #
# URL validation
# --------------------------------------------------------------------------- #


async def test_a_token_on_a_plain_http_url_is_rejected(async_db: AsyncSession) -> None:
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    workspace = await _workspace(async_db, organization, owner=owner)
    service = WorkspaceMcpServerService(async_db)

    with pytest.raises(WorkspaceMcpServerUnsafeUrlError):
        await service.create_server(
            user=owner,
            workspace_id=workspace.id,
            request=_create(url="http://93.184.216.34/mcp", authorization_token="secret"),
        )


async def test_a_non_http_scheme_is_rejected(async_db: AsyncSession) -> None:
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    workspace = await _workspace(async_db, organization, owner=owner)
    service = WorkspaceMcpServerService(async_db)

    with pytest.raises(WorkspaceMcpServerUnsafeUrlError):
        await service.create_server(user=owner, workspace_id=workspace.id, request=_create(url="ftp://example/mcp"))


async def test_adding_a_token_to_a_stored_http_url_is_rejected(async_db: AsyncSession) -> None:
    """The check runs against the effective merged state, not only against what the PATCH carried."""
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    workspace = await _workspace(async_db, organization, owner=owner)
    service = WorkspaceMcpServerService(async_db)

    created = await service.create_server(
        user=owner, workspace_id=workspace.id, request=_create(url="http://93.184.216.34/mcp")
    )
    with pytest.raises(WorkspaceMcpServerUnsafeUrlError):
        await service.update_server(
            user=owner,
            workspace_id=workspace.id,
            server_id=created.id,
            request=WorkspaceMcpServerUpdate(authorization_token="secret"),
        )


async def test_a_metadata_only_update_does_not_revalidate_the_url(async_db: AsyncSession) -> None:
    """Disabling a server must not depend on its host still passing the SSRF check."""
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    workspace = await _workspace(async_db, organization, owner=owner)
    service = WorkspaceMcpServerService(async_db)

    created = await service.create_server(
        user=owner, workspace_id=workspace.id, request=_create(url="http://93.184.216.34/mcp")
    )
    disabled = await service.update_server(
        user=owner,
        workspace_id=workspace.id,
        server_id=created.id,
        request=WorkspaceMcpServerUpdate(enabled=False),
    )
    assert disabled.enabled is False


# --------------------------------------------------------------------------- #
# The role gate
# --------------------------------------------------------------------------- #


async def test_a_plain_member_may_neither_read_nor_write(async_db: AsyncSession) -> None:
    """Reads are gated too: these rows name the endpoints the gateway connects to."""
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    workspace = await _workspace(async_db, organization, owner=owner)
    member = await _member(async_db, organization, role="member", full_name="Member")
    await WorkspaceMemberRepository(async_db).create(workspace_id=workspace.id, user_id=member.id, role="member")
    service = WorkspaceMcpServerService(async_db)

    with pytest.raises(NotAuthorizedError):
        await service.list_servers(user=member, workspace_id=workspace.id)
    with pytest.raises(NotAuthorizedError):
        await service.create_server(user=member, workspace_id=workspace.id, request=_create())


async def test_a_workspace_admin_may_manage_without_organization_admin(async_db: AsyncSession) -> None:
    """This repository's own rule, and a deliberate widening of the platform's org-admin-only gate."""
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    workspace = await _workspace(async_db, organization, owner=owner)
    member = await _member(async_db, organization, role="member", full_name="Workspace Admin")
    await WorkspaceMemberRepository(async_db).create(workspace_id=workspace.id, user_id=member.id, role="admin")
    service = WorkspaceMcpServerService(async_db)

    created = await service.create_server(user=member, workspace_id=workspace.id, request=_create())
    assert created.name == "github"


async def test_another_organizations_workspace_is_invisible(async_db: AsyncSession) -> None:
    organization = await _organization(async_db, slug="acme")
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    other_organization = await _organization(async_db, slug="other")
    other_owner = await _member(async_db, other_organization, role="owner", full_name="Other Owner")
    other_workspace = await _workspace(async_db, other_organization, name="Theirs", owner=other_owner)
    service = WorkspaceMcpServerService(async_db)

    with pytest.raises(WorkspaceNotFoundError):
        await service.list_servers(user=owner, workspace_id=other_workspace.id)


# --------------------------------------------------------------------------- #
# Request-path resolution
# --------------------------------------------------------------------------- #


async def test_resolve_returns_decrypted_configs_in_request_order(async_db: AsyncSession) -> None:
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    workspace = await _workspace(async_db, organization, owner=owner)
    service = WorkspaceMcpServerService(async_db)

    first = await service.create_server(
        user=owner,
        workspace_id=workspace.id,
        request=_create(name="github", authorization_token="ghp_token", allowed_tools=["list_issues"]),
    )
    second = await service.create_server(
        user=owner,
        workspace_id=workspace.id,
        request=_create(name="jira", url=OTHER_PUBLIC_URL, purpose_hint="Tickets"),
    )

    resolved = await resolve_workspace_mcp_servers(
        async_db,
        workspace_id=workspace.id,
        server_ids=[second.id, first.id, second.id],
    )
    assert [server.name for server in resolved] == ["jira", "github"], "de-duplicated, request order preserved"
    assert resolved[0].purpose_hint == "Tickets"
    assert resolved[1].authorization_token == "ghp_token"
    assert resolved[1].allowed_tools == ["list_issues"]


async def test_resolve_skips_a_disabled_server(async_db: AsyncSession) -> None:
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    workspace = await _workspace(async_db, organization, owner=owner)
    service = WorkspaceMcpServerService(async_db)

    created = await service.create_server(
        user=owner, workspace_id=workspace.id, request=_create(enabled=False)
    )

    assert await resolve_workspace_mcp_servers(async_db, workspace_id=workspace.id, server_ids=[created.id]) == []


async def test_resolve_refuses_an_id_from_another_workspace(async_db: AsyncSession) -> None:
    """Not found, not forbidden: the answer must not tell one workspace what another one holds."""
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    first = await _workspace(async_db, organization, name="First", owner=owner)
    second = await _workspace(async_db, organization, name="Second", owner=owner)
    service = WorkspaceMcpServerService(async_db)

    created = await service.create_server(user=owner, workspace_id=first.id, request=_create())

    with pytest.raises(WorkspaceMcpServerNotFoundError):
        await resolve_workspace_mcp_servers(async_db, workspace_id=second.id, server_ids=[created.id])


async def test_resolve_on_a_workspace_that_configured_nothing(async_db: AsyncSession) -> None:
    """otari#678's zero-rows requirement: no rows changes nothing, and names nothing.

    A deployment that has configured no MCP servers behaves exactly as it did
    before this table existed: a request that names no ids resolves to no
    servers, and one that names an id is refused rather than served with the
    id quietly dropped.
    """
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    workspace = await _workspace(async_db, organization, owner=owner)

    assert await resolve_workspace_mcp_servers(async_db, workspace_id=workspace.id, server_ids=[]) == []

    with pytest.raises(WorkspaceMcpServerNotFoundError):
        await resolve_workspace_mcp_servers(async_db, workspace_id=workspace.id, server_ids=[uuid.uuid4()])


async def test_deleting_a_workspace_takes_its_servers(async_db: AsyncSession) -> None:
    """The FK cascades: a configuration row has no meaning once its workspace is gone."""
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    workspace = await _workspace(async_db, organization, owner=owner)
    service = WorkspaceMcpServerService(async_db)
    created = await service.create_server(user=owner, workspace_id=workspace.id, request=_create())

    stored_workspace = await async_db.get(Workspace, workspace.id)
    assert stored_workspace is not None
    await async_db.delete(stored_workspace)
    await async_db.commit()

    assert await async_db.get(WorkspaceMcpServer, created.id) is None


# --------------------------------------------------------------------------- #
# The request path
# --------------------------------------------------------------------------- #


def _request_context(
    db: AsyncSession, workspace_id: uuid.UUID, organization_id: uuid.UUID | None = None
) -> RequestContext:
    """A standalone request context of the shape the completion preamble builds.

    ``organization_id`` is part of that shape: the preamble derives it from the
    same workspace it resolves, and every gate in `prepare_gateway_tools` now
    fails closed without it (otari#654), so a context missing it would be
    refused before reaching what these cases are about.
    """
    return RequestContext(
        config=GatewayConfig(),
        db=db,
        log_writer=None,  # type: ignore[arg-type]
        hybrid_mode=False,
        route=None,
        user_token=None,
        api_key_id="key-1",
        user_id="user-1",
        rate_limit_info=None,
        reservation=None,
        started_at=time.monotonic(),
        workspace_id=workspace_id,
        organization_id=organization_id or uuid.uuid4(),
    )


async def test_prepare_gateway_tools_hands_the_tool_loop_the_workspaces_servers(async_db: AsyncSession) -> None:
    """otari#658's second Definition-of-Done item, at the seam otari#678 named.

    `prepare_gateway_tools` is where admission resolves tool policy, and
    `ToolContext` is what the tool loop reads, so a stored server reaching
    `mcp_server_configs` with its token decrypted is the tool loop reaching it.
    """
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    workspace = await _workspace(async_db, organization, owner=owner)
    service = WorkspaceMcpServerService(async_db)
    stored = await service.create_server(
        user=owner,
        workspace_id=workspace.id,
        request=_create(name="github", authorization_token="ghp_token"),
    )

    tool_ctx = await prepare_gateway_tools(
        adapter=chat._ADAPTER,
        ctx=_request_context(async_db, workspace.id, organization.id),
        response=Response(),
        guardrails=None,
        guardrail_text="",
        tools=None,
        mcp_servers=None,
        mcp_server_ids=[stored.id],
        max_tool_iterations=None,
        tools_header=None,
    )

    assert tool_ctx.use_tool_loop is True
    assert tool_ctx.mcp_server_configs is not None
    assert [server.name for server in tool_ctx.mcp_server_configs] == ["github"]
    assert tool_ctx.mcp_server_configs[0].authorization_token == "ghp_token"


async def test_prepare_gateway_tools_merges_stored_servers_after_inline_ones(async_db: AsyncSession) -> None:
    """Same order the hybrid path uses: the caller's inline servers first, then the resolved ones."""
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    workspace = await _workspace(async_db, organization, owner=owner)
    service = WorkspaceMcpServerService(async_db)
    stored = await service.create_server(user=owner, workspace_id=workspace.id, request=_create(name="stored"))

    tool_ctx = await prepare_gateway_tools(
        adapter=chat._ADAPTER,
        ctx=_request_context(async_db, workspace.id, organization.id),
        response=Response(),
        guardrails=None,
        guardrail_text="",
        tools=None,
        mcp_servers=[McpServerConfig(name="inline", url=OTHER_PUBLIC_URL)],
        mcp_server_ids=[stored.id],
        max_tool_iterations=None,
        tools_header=None,
    )

    assert tool_ctx.mcp_server_configs is not None
    assert [server.name for server in tool_ctx.mcp_server_configs] == ["inline", "stored"]


async def test_prepare_gateway_tools_is_unchanged_when_nothing_is_configured(async_db: AsyncSession) -> None:
    """otari#678's zero-rows requirement on the request path itself.

    A deployment that has configured no workspace MCP servers keeps serving an
    inline ``mcp_servers`` request exactly as it did, and a request that names
    no ids never touches the new table.
    """
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    workspace = await _workspace(async_db, organization, owner=owner)

    tool_ctx = await prepare_gateway_tools(
        adapter=chat._ADAPTER,
        ctx=_request_context(async_db, workspace.id, organization.id),
        response=Response(),
        guardrails=None,
        guardrail_text="",
        tools=None,
        mcp_servers=[McpServerConfig(name="inline", url=PUBLIC_URL)],
        mcp_server_ids=None,
        max_tool_iterations=None,
        tools_header=None,
    )

    assert tool_ctx.mcp_server_configs is not None
    assert [server.name for server in tool_ctx.mcp_server_configs] == ["inline"]


async def test_prepare_gateway_tools_refuses_an_unknown_id(async_db: AsyncSession) -> None:
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    workspace = await _workspace(async_db, organization, owner=owner)

    with pytest.raises(HTTPException) as exc_info:
        await prepare_gateway_tools(
            adapter=chat._ADAPTER,
            ctx=_request_context(async_db, workspace.id, organization.id),
            response=Response(),
            guardrails=None,
            guardrail_text="",
            tools=None,
            mcp_servers=None,
            mcp_server_ids=[uuid.uuid4()],
            max_tool_iterations=None,
            tools_header=None,
        )

    assert exc_info.value.status_code == 404


# --------------------------------------------------------------------------- #
# Request-shape bounds
# --------------------------------------------------------------------------- #


async def test_an_explicit_null_on_a_required_field_is_refused() -> None:
    """``None`` is this schema's "not sent" marker, so clearing a NOT NULL column is a 422.

    Without this the value reaches the flush and comes back as the same
    ``IntegrityError`` a duplicate name raises, which the service would then
    report as a name collision that never happened.
    """
    for field in ("name", "url", "enabled"):
        with pytest.raises(ValidationError):
            WorkspaceMcpServerUpdate(**{field: None})


async def test_the_published_schema_does_not_advertise_null_where_it_is_refused() -> None:
    """The generated client must not tell a caller that a payload the server 422s is valid.

    ``name``/``url``/``enabled`` back NOT NULL columns and the validator above
    refuses an explicit ``null`` for them, so they are omittable but not
    nullable. The three that really can be cleared stay nullable.
    """
    properties = WorkspaceMcpServerUpdate.model_json_schema()["properties"]

    for field, json_type in (("name", "string"), ("url", "string"), ("enabled", "boolean")):
        assert properties[field].get("type") == json_type, field
        assert "anyOf" not in properties[field], f"{field} is advertised as nullable"

    for field in ("purpose_hint", "allowed_tools", "authorization_token"):
        assert {"type": "null"} in properties[field]["anyOf"], f"{field} should stay clearable"


async def test_a_nullable_field_can_still_be_cleared(async_db: AsyncSession) -> None:
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    workspace = await _workspace(async_db, organization, owner=owner)
    service = WorkspaceMcpServerService(async_db)
    created = await service.create_server(
        user=owner, workspace_id=workspace.id, request=_create(purpose_hint="Repository lookups")
    )

    cleared = await service.update_server(
        user=owner,
        workspace_id=workspace.id,
        server_id=created.id,
        request=WorkspaceMcpServerUpdate(purpose_hint=None),
    )
    assert cleared.purpose_hint is None


async def test_the_tool_allow_list_is_bounded() -> None:
    """The one field that is arbitrary JSON, and so the one that could otherwise be unbounded."""
    with pytest.raises(ValidationError):
        WorkspaceMcpServerCreate(name="github", url=PUBLIC_URL, allowed_tools=["t"] * (MAX_ALLOWED_TOOLS + 1))
    with pytest.raises(ValidationError):
        WorkspaceMcpServerCreate(name="github", url=PUBLIC_URL, allowed_tools=["t" * 300])


async def test_mcp_server_ids_is_bounded_on_the_request() -> None:
    """A workspace holds at most 50 servers in either mode, so a longer list can only be waste."""
    with pytest.raises(ValidationError):
        ChatCompletionRequest(
            model="openai:gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
            mcp_server_ids=[uuid.uuid4() for _ in range(MAX_MCP_SERVER_IDS + 1)],
        )
