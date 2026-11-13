"""How a workspace shows its MCP servers: each server's logo, and an admin's own logo and tile choice.

A server's logo is an admin's upload when there is one. Otherwise it is the one
the mcp-catalog domain finds for the server's URL, which this service asks for
through ``ServerLogoSource`` rather than importing it. The upload is kept apart
from what the catalog fetches, so a fetch or a sync never replaces it.

Any member who can see the workspace reads the logos, as they read the servers.
Only someone who may manage the workspace changes one.
"""

import uuid
from collections.abc import Sequence
from typing import Protocol, cast

from gateway.core.config import API_ROOT
from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions.tools_exceptions import McpLogoInvalidError, WorkspaceMcpServerNotFoundError
from gateway.log_config import logger
from gateway.models.tenancy import User, Workspace
from gateway.models.tools import WorkspaceMcpServerLogo
from gateway.repositories.tools import (
    WorkspaceMcpServerLogoFacts,
    WorkspaceMcpServerLogoRepository,
    WorkspaceMcpServerRepository,
)
from gateway.schemas.tools import WorkspaceMcpServerLogoPublic, WorkspaceMcpServerLogosPublic
from gateway.services.mcp_catalog import Logo, LogoRef, LogoTile

_TILES: frozenset[str] = frozenset({"auto", "always", "never"})


class WorkspaceChecks(Protocol):
    """What this service needs of the tenancy domain: whether a caller sees or manages a workspace."""

    async def resolve_visible_workspace(self, *, user: User, workspace_id: uuid.UUID) -> Workspace: ...

    async def require_workspace_management_access(self, *, user: User, workspace: Workspace) -> None: ...


class ServerLogoSource(Protocol):
    """Where a server's own logo comes from when no admin uploaded one: the mcp-catalog domain."""

    async def logo_refs(self, urls: Sequence[str]) -> dict[str, LogoRef]: ...

    async def logo_for_url(self, url: str) -> Logo | None: ...


def server_logo_path(workspace_id: uuid.UUID, server_id: uuid.UUID) -> str:
    return f"{API_ROOT}/workspaces/{workspace_id}/mcp-servers/{server_id}/logo"


def _uploaded(row: WorkspaceMcpServerLogo | None) -> Logo | None:
    if row is None or row.content is None or row.etag is None:
        return None
    return Logo(row.content, "image/png", row.etag, row.luminance)


def _facts(row: WorkspaceMcpServerLogo | None) -> WorkspaceMcpServerLogoFacts | None:
    if row is None:
        return None
    return WorkspaceMcpServerLogoFacts(
        tile=row.tile, etag=row.etag, luminance=row.luminance, has_content=row.content is not None
    )


def _public(
    workspace_id: uuid.UUID, server_id: uuid.UUID, row: WorkspaceMcpServerLogoFacts | None, ref: LogoRef | None
) -> WorkspaceMcpServerLogoPublic:
    tile = cast(LogoTile, row.tile if row is not None and row.tile in _TILES else "auto")
    path = server_logo_path(workspace_id, server_id)
    uploaded = row is not None and row.has_content and row.etag is not None
    if row is not None and uploaded:
        url: str | None = f"{path}?v={row.etag}"
        luminance = row.luminance
    else:
        url = ref.url(path) if ref is not None else None
        luminance = ref.luminance if ref is not None else None
    return WorkspaceMcpServerLogoPublic(
        server_id=server_id,
        logo_url=url,
        logo_tone=Logo.tone(luminance, tile) if url is not None else None,
        logo_tile=tile,
        has_logo_override=uploaded,
    )


class McpServerLogoService:
    """Reads each server's logo, and sets or removes an admin's own."""

    def __init__(
        self,
        uow: UnitOfWork,
        servers: WorkspaceMcpServerRepository,
        logos: WorkspaceMcpServerLogoRepository,
        workspaces: WorkspaceChecks,
        source: ServerLogoSource,
    ) -> None:
        self._uow = uow
        self._servers = servers
        self._logos = logos
        self._workspaces = workspaces
        self._source = source

    async def _server_url(self, workspace_id: uuid.UUID, server_id: uuid.UUID) -> str:
        """The URL of a server in this workspace. Inside a block. A server of another workspace is not found."""
        server = await self._servers.find(server_id)
        if server is None or server.workspace_id != workspace_id:
            raise WorkspaceMcpServerNotFoundError(server_id)
        return server.url

    async def list_logos(self, *, user: User, workspace_id: uuid.UUID) -> WorkspaceMcpServerLogosPublic:
        """The logo of each of the workspace's servers. Starts no look for one."""
        workspace = await self._workspaces.resolve_visible_workspace(user=user, workspace_id=workspace_id)
        async with self._uow:
            urls = await self._servers.urls_in_workspace(workspace.id)
            rows = await self._logos.facts_for_servers(list(urls))
        refs = await self._source.logo_refs(list(urls.values()))
        return WorkspaceMcpServerLogosPublic(
            data=[
                _public(workspace.id, server_id, rows.get(server_id), refs.get(url)) for server_id, url in urls.items()
            ]
        )

    async def logo(self, *, user: User, workspace_id: uuid.UUID, server_id: uuid.UUID) -> Logo | None:
        """The image a server shows: the admin's upload, else its own, looked for first if a look is due."""
        workspace = await self._workspaces.resolve_visible_workspace(user=user, workspace_id=workspace_id)
        async with self._uow:
            url = await self._server_url(workspace.id, server_id)
            uploaded = _uploaded(await self._logos.find(server_id))
        return uploaded or await self._source.logo_for_url(url)

    async def set_logo(
        self, *, user: User, workspace_id: uuid.UUID, server_id: uuid.UUID, tile: str, content: bytes | None
    ) -> WorkspaceMcpServerLogoPublic:
        """Set the tile choice, and replace the uploaded logo when ``content`` is given.

        Raises:
            McpLogoInvalidError: ``content`` is not a PNG the gateway can read, or ``tile`` is not a choice.
        """
        if tile not in _TILES:
            raise McpLogoInvalidError
        uploaded: Logo | None = None
        if content is not None:
            if len(content) > Logo.MAX_UPLOAD_BYTES or Logo.sniff(content) != "image/png":
                raise McpLogoInvalidError
            uploaded = await Logo.measured(content, "image/png")
        workspace = await self._workspaces.resolve_visible_workspace(user=user, workspace_id=workspace_id)
        await self._workspaces.require_workspace_management_access(user=user, workspace=workspace)
        async with self._uow:
            url = await self._server_url(workspace.id, server_id)
            row = await self._logos.save(
                server_id,
                tile=tile,
                content=uploaded.content if uploaded else None,
                etag=uploaded.etag if uploaded else None,
                luminance=uploaded.luminance if uploaded else None,
                replace_content=uploaded is not None,
            )
        logger.info("MCP server logo saved server_id=%s uploaded=%s tile=%s", server_id, uploaded is not None, tile)
        return await self._describe(workspace.id, server_id, row, url)

    async def remove_upload(
        self, *, user: User, workspace_id: uuid.UUID, server_id: uuid.UUID
    ) -> WorkspaceMcpServerLogoPublic:
        """Remove the uploaded logo, so the server shows its own again. The tile choice stays."""
        workspace = await self._workspaces.resolve_visible_workspace(user=user, workspace_id=workspace_id)
        await self._workspaces.require_workspace_management_access(user=user, workspace=workspace)
        async with self._uow:
            url = await self._server_url(workspace.id, server_id)
            row = await self._logos.find(server_id)
            if row is not None:
                row = await self._logos.save(
                    server_id, tile=row.tile, content=None, etag=None, luminance=None, replace_content=True
                )
        logger.info("MCP server logo removed server_id=%s", server_id)
        return await self._describe(workspace.id, server_id, row, url)

    async def _describe(
        self, workspace_id: uuid.UUID, server_id: uuid.UUID, row: WorkspaceMcpServerLogo | None, url: str
    ) -> WorkspaceMcpServerLogoPublic:
        refs = await self._source.logo_refs([url])
        return _public(workspace_id, server_id, _facts(row), refs.get(url))
