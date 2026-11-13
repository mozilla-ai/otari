"""A workspace's MCP server logos: what each server shows, and an admin's own logo and tile choice.

Exercised at the service layer, as ``test_workspace_mcp_servers.py`` is, because the
rules that matter (a plain member refused, another organization's workspace not
found) need identities at roles the API's one operator identity never has.
"""

import io
import uuid
from datetime import UTC, datetime

import pytest
from PIL import Image
from sqlalchemy import Engine, event
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.config import API_ROOT, GatewayConfig
from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions.organizations_exceptions import NotAuthorizedError
from gateway.exceptions.tools_exceptions import McpLogoInvalidError, WorkspaceMcpServerNotFoundError
from gateway.models.tenancy import Organization, User, Workspace
from gateway.models.tools import WorkspaceMcpServer
from gateway.repositories.mcp_catalog import McpCatalogServerRepository, McpCatalogSyncStateRepository
from gateway.repositories.tenancy import (
    OrganizationMemberRepository,
    OrganizationRepository,
    UserRepository,
    WorkspaceMemberRepository,
    WorkspaceRepository,
)
from gateway.repositories.tools import WorkspaceMcpServerLogoRepository, WorkspaceMcpServerRepository
from gateway.schemas.mcp_catalog import McpCatalogEntry
from gateway.services.mcp_catalog import Logo, LogoLookBudget, McpCatalogService
from gateway.services.tenancy import OrganizationService
from gateway.services.tenancy.authorization import WorkspaceAccess
from gateway.services.tools import McpServerLogoService

pytestmark = pytest.mark.asyncio


def _png(color: tuple[int, int, int, int], size: int = 16) -> bytes:
    buffer = io.BytesIO()
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    # A mark in the middle on a transparent field, so its ink is measured.
    for x in range(4, size - 4):
        for y in range(4, size - 4):
            image.putpixel((x, y), color)
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class _Fetcher:
    def __init__(self, images: dict[str, bytes] | None = None) -> None:
        self.images = images or {}
        self.asked: list[str] = []

    def __call__(self) -> "_Fetcher":
        return self

    async def fetch(self, url: str) -> Logo | None:
        self.asked.append(url)
        content = self.images.get(url)
        return Logo.of(content, "image/png") if content is not None else None

    async def aclose(self) -> None:
        return None


async def _organization(db: AsyncSession, slug: str = "acme") -> Organization:
    return await OrganizationRepository(db).create_organization(name=slug.title(), slug=slug, created_by_user_id=None)


async def _member(db: AsyncSession, organization: Organization, role: str) -> User:
    user = await UserRepository(db).create_local_identity(full_name=role, active_organization_id=organization.id)
    await OrganizationMemberRepository(db).create_membership(
        organization_id=organization.id, user_id=user.id, role=role
    )
    return user


async def _workspace(db: AsyncSession, organization: Organization, *members: tuple[User, str]) -> Workspace:
    workspace = await WorkspaceRepository(db).create_workspace(
        name="Default", organization_id=organization.id, created_by_user_id=None
    )
    for user, role in members:
        await WorkspaceMemberRepository(db).create(workspace_id=workspace.id, user_id=user.id, role=role)
    await db.commit()
    return workspace


async def _server(db: AsyncSession, workspace: Workspace, url: str, name: str = "server") -> uuid.UUID:
    server = WorkspaceMcpServer(workspace_id=workspace.id, name=name, url=url)
    db.add(server)
    await db.commit()
    return server.id


def _service(db: AsyncSession, fetcher: _Fetcher) -> McpServerLogoService:
    uow = UnitOfWork(db)
    workspaces = WorkspaceAccess(db, OrganizationService(db, membership_listener=None))
    catalog = McpCatalogService(
        uow,
        McpCatalogServerRepository(uow),
        McpCatalogSyncStateRepository(uow),
        workspaces,
        GatewayConfig(),
        fetcher=fetcher,
        look_budget=LogoLookBudget(),
    )
    return McpServerLogoService(
        uow, WorkspaceMcpServerRepository(uow), WorkspaceMcpServerLogoRepository(uow), workspaces, catalog
    )


async def _setup(db: AsyncSession) -> tuple[Workspace, User, User]:
    organization = await _organization(db)
    owner = await _member(db, organization, "owner")
    member = await _member(db, organization, "member")
    workspace = await _workspace(db, organization, (owner, "owner"), (member, "member"))
    return workspace, owner, member


async def test_a_server_added_by_url_shows_its_hosts_logo(async_db: AsyncSession) -> None:
    workspace, _, member = await _setup(async_db)
    dark = _png((10, 10, 10, 255))
    fetcher = _Fetcher({"https://example.com/apple-touch-icon.png": dark})
    service = _service(async_db, fetcher)
    server_id = await _server(async_db, workspace, "https://mcp.example.com/mcp")
    path = f"{API_ROOT}/workspaces/{workspace.id}/mcp-servers/{server_id}/logo"

    before = await service.list_logos(user=member, workspace_id=workspace.id)
    logo = await service.logo(user=member, workspace_id=workspace.id, server_id=server_id)
    after = await service.list_logos(user=member, workspace_id=workspace.id)

    # Listing reached nothing; the first request for the image did, once.
    assert before.data[0].logo_url == path
    assert logo is not None and logo.content == dark
    assert fetcher.asked == ["https://example.com/apple-touch-icon.png"]
    entry = after.data[0]
    assert entry.logo_url == f"{path}?v={logo.etag}"
    # Dark ink needs a light tile on a dark background.
    assert (entry.logo_tone, entry.logo_tile, entry.has_logo_override) == ("dark", "auto", False)


async def test_a_server_from_the_catalog_shows_the_catalogs_logo(async_db: AsyncSession) -> None:
    workspace, owner, _ = await _setup(async_db)
    fetcher = _Fetcher()
    service = _service(async_db, fetcher)
    async with UnitOfWork(async_db) as uow:
        await McpCatalogServerRepository(uow).upsert(
            [
                McpCatalogEntry(
                    name="com.stripe/mcp",
                    title="Stripe",
                    description=None,
                    url="https://mcp.stripe.com",
                    requires_token=False,
                    website_url=None,
                    version=None,
                    source="registry",
                )
            ],
            seen_at=datetime.now(UTC),
        )
    server_id = await _server(async_db, workspace, "https://mcp.stripe.com/")

    logo = await service.logo(user=owner, workspace_id=workspace.id, server_id=server_id)

    assert logo is not None and logo.content_type == "image/svg+xml"
    assert fetcher.asked == []


async def test_an_admins_upload_and_tile_win_and_removing_the_upload_keeps_the_tile(async_db: AsyncSession) -> None:
    workspace, owner, member = await _setup(async_db)
    fetcher = _Fetcher({"https://example.com/apple-touch-icon.png": _png((10, 10, 10, 255))})
    service = _service(async_db, fetcher)
    server_id = await _server(async_db, workspace, "https://mcp.example.com/mcp")
    upload = _png((250, 250, 250, 255))

    saved = await service.set_logo(
        user=owner, workspace_id=workspace.id, server_id=server_id, tile="always", content=upload
    )
    served = await service.logo(user=member, workspace_id=workspace.id, server_id=server_id)

    assert saved.has_logo_override is True
    assert (saved.logo_tile, saved.logo_tone) == ("always", "tile")
    assert served is not None and served.content == upload
    # The upload never waits on the server's own logo.
    assert fetcher.asked == []

    removed = await service.remove_upload(user=owner, workspace_id=workspace.id, server_id=server_id)
    assert (removed.has_logo_override, removed.logo_tile) == (False, "always")
    served = await service.logo(user=member, workspace_id=workspace.id, server_id=server_id)
    assert served is not None and served.content != upload


async def test_only_a_png_is_accepted_as_an_upload(async_db: AsyncSession) -> None:
    workspace, owner, _ = await _setup(async_db)
    service = _service(async_db, _Fetcher())
    server_id = await _server(async_db, workspace, "https://mcp.example.com/mcp")
    svg = b'<svg xmlns="http://www.w3.org/2000/svg"><rect width="1" height="1"/></svg>'

    for content in (svg, b"not an image", b"\x89PNG\r\n\x1a\n" + b"\x00" * 32):
        with pytest.raises(McpLogoInvalidError):
            await service.set_logo(
                user=owner, workspace_id=workspace.id, server_id=server_id, tile="auto", content=content
            )
    with pytest.raises(McpLogoInvalidError):
        await service.set_logo(
            user=owner, workspace_id=workspace.id, server_id=server_id, tile="sometimes", content=None
        )


async def test_a_png_whose_pixels_do_not_decode_is_refused(async_db: AsyncSession) -> None:
    """Its signature and header read fine, so only decoding it finds the damage."""
    workspace, owner, _ = await _setup(async_db)
    service = _service(async_db, _Fetcher())
    server_id = await _server(async_db, workspace, "https://mcp.example.com/mcp")
    whole = _png((0, 0, 0, 255), size=64)
    truncated = whole[: len(whole) // 2]
    assert Logo.sniff(truncated) == "image/png"

    with pytest.raises(McpLogoInvalidError):
        await service.set_logo(
            user=owner, workspace_id=workspace.id, server_id=server_id, tile="auto", content=truncated
        )


async def test_a_member_reads_logos_but_may_not_change_one(async_db: AsyncSession) -> None:
    workspace, _, member = await _setup(async_db)
    service = _service(async_db, _Fetcher())
    server_id = await _server(async_db, workspace, "https://mcp.example.com/mcp")

    assert len((await service.list_logos(user=member, workspace_id=workspace.id)).data) == 1
    with pytest.raises(NotAuthorizedError):
        await service.set_logo(user=member, workspace_id=workspace.id, server_id=server_id, tile="never", content=None)
    with pytest.raises(NotAuthorizedError):
        await service.remove_upload(user=member, workspace_id=workspace.id, server_id=server_id)


async def test_another_workspaces_server_is_not_found(async_db: AsyncSession) -> None:
    workspace, owner, _ = await _setup(async_db)
    other_organization = await _organization(async_db, "other")
    other_owner = await _member(async_db, other_organization, "owner")
    other = await _workspace(async_db, other_organization, (other_owner, "owner"))
    fetcher = _Fetcher()
    service = _service(async_db, fetcher)
    foreign = await _server(async_db, other, "https://mcp.example.com/mcp")

    # Named under the caller's own workspace, the other workspace's server does not exist. Ids are read
    # first: the refusal rolls the shared session back, which expires the objects the test holds.
    workspace_id = workspace.id
    with pytest.raises(WorkspaceMcpServerNotFoundError):
        await service.logo(user=owner, workspace_id=workspace_id, server_id=foreign)
    await async_db.refresh(owner)
    with pytest.raises(WorkspaceMcpServerNotFoundError):
        await service.set_logo(user=owner, workspace_id=workspace_id, server_id=foreign, tile="never", content=None)
    assert fetcher.asked == []


async def test_deleting_a_server_takes_its_logo(async_db: AsyncSession) -> None:
    workspace, owner, _ = await _setup(async_db)
    service = _service(async_db, _Fetcher())
    server_id = await _server(async_db, workspace, "https://mcp.example.com/mcp")
    await service.set_logo(
        user=owner, workspace_id=workspace.id, server_id=server_id, tile="auto", content=_png((0, 0, 0, 255))
    )

    server = await async_db.get(WorkspaceMcpServer, server_id)
    await async_db.delete(server)
    await async_db.commit()

    async with UnitOfWork(async_db) as uow:
        assert await WorkspaceMcpServerLogoRepository(uow).find(server_id) is None


async def test_the_list_shows_an_upload_without_reading_its_image(async_db: AsyncSession) -> None:
    workspace, owner, member = await _setup(async_db)
    service = _service(async_db, _Fetcher())
    server_id = await _server(async_db, workspace, "https://mcp.example.com/mcp")
    saved = await service.set_logo(
        user=owner, workspace_id=workspace.id, server_id=server_id, tile="never", content=_png((250, 250, 250, 255))
    )
    statements: list[str] = []

    def record(_conn: object, _cursor: object, statement: str, *_args: object) -> None:
        statements.append(statement)

    event.listen(Engine, "before_cursor_execute", record)
    try:
        listed = await service.list_logos(user=member, workspace_id=workspace.id)
    finally:
        event.remove(Engine, "before_cursor_execute", record)

    (row,) = listed.data
    assert (row.logo_url, row.has_logo_override, row.logo_tile) == (saved.logo_url, True, "never")
    logo_reads = [statement for statement in statements if "workspace_mcp_server_logos" in statement]
    # Whether an image is held is asked; the image itself is never selected.
    selected = [s.replace("workspace_mcp_server_logos.content IS NOT NULL", "") for s in logo_reads]
    assert logo_reads and not any("workspace_mcp_server_logos.content" in s for s in selected)
