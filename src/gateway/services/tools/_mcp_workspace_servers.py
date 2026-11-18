"""How one workspace sees a stored MCP server: its own, or one of its organization's under the workspace's settings."""

import uuid
from collections.abc import Collection, Sequence
from dataclasses import dataclass

from gateway.models.tools import WorkspaceMcpServer, WorkspaceMcpServerOverride
from gateway.repositories.tools import (
    McpServerAliasRepository,
    WorkspaceMcpServerOverrideRepository,
    WorkspaceMcpServerRepository,
)
from gateway.services.model_access import (
    McpAccess,
    clip_mcp_tool_patterns,
    expand_mcp_tool_patterns,
    mcp_read_tools,
)


@dataclass(frozen=True, slots=True)
class McpServerInWorkspace:
    """A stored server as one workspace reaches it.

    A workspace's own server reads as its row does. An organization's server reads through the
    workspace's override, where there is one: the workspace's ``everyone_tools`` and
    ``attach_by_default`` stand in for the server's, and every caller's tools are clipped to the
    organization's limit for the workspace, so no rule set in the workspace reaches past it.
    """

    row: WorkspaceMcpServer
    override: WorkspaceMcpServerOverride | None = None

    @property
    def id(self) -> uuid.UUID:
        return self.row.id

    @property
    def of_organization(self) -> bool:
        """Whether the server is its organization's rather than one workspace's own."""
        return self.row.workspace_id is None

    @property
    def visible(self) -> bool:
        """Whether the workspace reaches the server at all. An organization's server can be turned off three ways."""
        if not self.of_organization:
            return True
        override = self.override
        return self.row.enabled and (override is None or (override.enabled and override.organization_tools != []))

    @property
    def enabled(self) -> bool:
        return self.row.enabled and self.visible

    @property
    def everyone_tools(self) -> list[str]:
        if self.override is not None and self.override.everyone_tools is not None:
            return list(self.override.everyone_tools)
        return list(self.row.everyone_tools)

    @property
    def attach_by_default(self) -> bool:
        if self.override is not None and self.override.attach_by_default is not None:
            return self.override.attach_by_default
        return self.row.attach_by_default

    @property
    def read_tools(self) -> list[str]:
        return mcp_read_tools(self.row.read_tools, self.row.read_tools_hinted)

    @property
    def tool_limit(self) -> list[str] | None:
        """The organization's limit on this workspace's callers, ``@read`` expanded; ``None`` sets none."""
        if self.override is None or self.override.organization_tools is None:
            return None
        return expand_mcp_tool_patterns(self.override.organization_tools, self.read_tools)

    def refused_by(self, access: McpAccess) -> str:
        """What decided that a caller holds no tool here: the organization's limit, or the caller's own rule."""
        if self.tool_limit is not None and access.patterns_for(self.id, self.everyone_tools, self.read_tools):
            return "organization"
        return access.decide(self.id, self.everyone_tools)[0]

    def patterns_for(self, access: McpAccess) -> list[str]:
        """The tool patterns a caller holds on the server here, ``@read`` expanded; empty means it cannot reach it."""
        granted = access.patterns_for(self.id, self.everyone_tools, self.read_tools)
        limit = self.tool_limit
        return granted if limit is None else clip_mcp_tool_patterns(granted, limit)


class McpWorkspaceServers:
    """Reads the stored servers a workspace reaches, its own and its organization's, as ``McpServerInWorkspace``."""

    def __init__(
        self,
        servers: WorkspaceMcpServerRepository,
        overrides: WorkspaceMcpServerOverrideRepository,
        aliases: McpServerAliasRepository,
    ) -> None:
        self._servers = servers
        self._overrides = overrides
        self._aliases = aliases

    async def seen_from(
        self, workspace_id: uuid.UUID, rows: Sequence[WorkspaceMcpServer]
    ) -> list[McpServerInWorkspace]:
        """These rows as the workspace sees them, in their order, leaving out what it does not reach."""
        shared = [row.id for row in rows if row.workspace_id is None]
        overrides = await self._overrides.for_workspace(workspace_id, shared) if shared else {}
        views = [McpServerInWorkspace(row, overrides.get(row.id)) for row in rows]
        return [view for view in views if view.visible]

    async def reaches(self, workspace_id: uuid.UUID, row: WorkspaceMcpServer) -> bool:
        """Whether the workspace reaches a server: its own always, its organization's unless turned off there."""
        return bool(await self.seen_from(workspace_id, [row]))

    async def for_request(
        self,
        workspace_id: uuid.UUID,
        organization_id: uuid.UUID,
        server_ids: Collection[uuid.UUID],
        *,
        attached: bool,
    ) -> tuple[list[McpServerInWorkspace], dict[uuid.UUID, uuid.UUID]]:
        """The servers these ids name in the workspace, and with ``attached`` every one it might attach, oldest first.

        Also returns the id each given id names: itself, or for an old id of a folded workspace copy the
        organization's server it was folded into. Callers check :attr:`McpServerInWorkspace.attach_by_default`.
        """
        named = list(dict.fromkeys(server_ids))
        rows = await self._servers.for_request(workspace_id, organization_id, named, attached=attached)
        found = {row.id for row in rows}
        missing = [server_id for server_id in named if server_id not in found]
        # Only an id naming nothing is looked up, so a request that names live ids reads no alias.
        targets = await self._aliases.targets(workspace_id, missing) if missing else {}
        if targets:
            swapped = list(dict.fromkeys(targets.get(server_id, server_id) for server_id in named))
            rows = await self._servers.for_request(workspace_id, organization_id, swapped, attached=attached)
        return await self.seen_from(workspace_id, rows), {
            server_id: targets.get(server_id, server_id) for server_id in named
        }

    async def swap_aliases(self, workspace_id: uuid.UUID, server_ids: Collection[uuid.UUID]) -> list[uuid.UUID]:
        """These ids in their order, each old id of a folded copy swapped for the server it names in the workspace."""
        ids = list(dict.fromkeys(server_ids))
        targets = await self._aliases.targets(workspace_id, ids) if ids else {}
        return list(dict.fromkeys(targets.get(server_id, server_id) for server_id in ids))

    async def every(self, workspace_id: uuid.UUID, organization_id: uuid.UUID) -> list[McpServerInWorkspace]:
        """Every server the workspace reaches, a disabled one of its own included, oldest first."""
        return await self.seen_from(workspace_id, await self._servers.in_workspace(workspace_id, organization_id))

    async def enabled(self, workspace_id: uuid.UUID, organization_id: uuid.UUID) -> list[McpServerInWorkspace]:
        """Every enabled server the workspace reaches, oldest first."""
        rows = await self._servers.enabled_in_workspace(workspace_id, organization_id)
        return [view for view in await self.seen_from(workspace_id, rows) if view.enabled]

    async def any_organization_server_attached(self, workspace_id: uuid.UUID, organization_id: uuid.UUID) -> bool:
        """Whether the workspace attaches one of its organization's enabled servers to every request."""
        shared = await self._servers.enabled_of_organization(organization_id)
        return any(view.attach_by_default for view in await self.seen_from(workspace_id, shared))
