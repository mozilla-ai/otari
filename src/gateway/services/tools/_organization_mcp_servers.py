"""An organization's MCP server across its workspaces: where it is on, and the limit the organization sets in each."""

import uuid
from collections.abc import Awaitable, Callable

from pydantic import BaseModel, Field, model_validator

from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions.organizations_exceptions import WorkspaceNotFoundError
from gateway.exceptions.tools_exceptions import WorkspaceMcpServerNotFoundError
from gateway.models.tenancy import User
from gateway.models.tools import WorkspaceMcpServer, WorkspaceMcpServerOverride
from gateway.repositories.tools import WorkspaceMcpServerOverrideRepository, WorkspaceMcpServerRepository
from gateway.services.model_access import validate_mcp_tool_patterns
from gateway.services.tenancy import OrganizationService

# Workspaces read per page while listing an organization's, which holds few.
_WORKSPACE_PAGE = 200


class OrganizationMcpServerLimit(BaseModel):
    """The organization's limit on one workspace's callers of one of its servers."""

    tools: list[str] = Field(
        description="Tool patterns ('*', a name, 'prefix*', '@read') the workspace's callers may reach at most; "
        "[] turns the server off for the workspace, and its admins cannot turn it back on"
    )

    @model_validator(mode="after")
    def _check(self) -> "OrganizationMcpServerLimit":
        self.tools = validate_mcp_tool_patterns(self.tools)
        return self


class OrganizationMcpServerWorkspace(BaseModel):
    """One workspace of the organization, as one of its servers reaches it."""

    workspace_id: uuid.UUID
    name: str
    reached: bool = Field(
        description="The workspace's requests reach the server: neither the organization nor the workspace turned "
        "it off there, and the server itself is on"
    )
    organization_tools: list[str] | None = Field(
        description="The organization's limit for the workspace; null sets none, [] turns the server off there"
    )
    workspace_enabled: bool = Field(description="The workspace's own switch, which its admins set")
    narrowed: bool = Field(
        description="The workspace's admins changed something of their own: turned the server off, or set its "
        "default access or whether it is attached"
    )

    @classmethod
    def of(
        cls, workspace_id: uuid.UUID, name: str, server: WorkspaceMcpServer, override: WorkspaceMcpServerOverride | None
    ) -> "OrganizationMcpServerWorkspace":
        workspace_enabled = override is None or override.enabled
        limit = None if override is None else override.organization_tools
        return cls(
            workspace_id=workspace_id,
            name=name,
            reached=server.enabled and workspace_enabled and limit != [],
            organization_tools=None if limit is None else list(limit),
            workspace_enabled=workspace_enabled,
            narrowed=override is not None
            and (not override.enabled or override.everyone_tools is not None or override.attach_by_default is not None),
        )


class OrganizationMcpServerWorkspaces(BaseModel):
    """Every workspace of the organization, as one of its servers reaches each, by name."""

    data: list[OrganizationMcpServerWorkspace]
    count: int


class OrganizationMcpServerWorkspaceService:
    """Shows where an organization's server is on, and sets or clears the organization's limit in a workspace.

    Owners and admins of the organization only. The workspace's own settings on the server stay as its admins
    left them: a limit narrows them, and never widens anything a workspace set.
    """

    def __init__(
        self,
        uow: UnitOfWork,
        *,
        servers: WorkspaceMcpServerRepository,
        overrides: WorkspaceMcpServerOverrideRepository,
        organizations: OrganizationService,
        lock_workspace: Callable[[uuid.UUID], Awaitable[None]],
    ) -> None:
        """``lock_workspace`` takes a workspace's row lock, which serializes two writes of its settings."""
        self.uow = uow
        self.servers = servers
        self.overrides = overrides
        self.organizations = organizations
        self.lock_workspace = lock_workspace

    async def workspaces(self, *, user: User, server_id: uuid.UUID) -> OrganizationMcpServerWorkspaces:
        """Each workspace of the caller's organization, as the server reaches it, by name."""
        server = await self._organization_server(user, server_id)
        names = await self._workspace_names(user)
        async with self.uow:
            overrides = {override.workspace_id: override for override in await self.overrides.for_server(server.id)}
        data = [
            OrganizationMcpServerWorkspace.of(workspace_id, name, server, overrides.get(workspace_id))
            for workspace_id, name in sorted(names.items(), key=lambda item: item[1].casefold())
        ]
        return OrganizationMcpServerWorkspaces(data=data, count=len(data))

    async def set_limit(
        self, *, user: User, server_id: uuid.UUID, workspace_id: uuid.UUID, limit: OrganizationMcpServerLimit
    ) -> OrganizationMcpServerWorkspace:
        """Limit what a workspace's callers reach on the server, ``[]`` turning it off there."""
        return await self._write(user, server_id, workspace_id, limit.tools)

    async def clear_limit(
        self, *, user: User, server_id: uuid.UUID, workspace_id: uuid.UUID
    ) -> OrganizationMcpServerWorkspace:
        """Lift the organization's limit in a workspace. Idempotent."""
        return await self._write(user, server_id, workspace_id, None)

    async def _write(
        self, user: User, server_id: uuid.UUID, workspace_id: uuid.UUID, tools: list[str] | None
    ) -> OrganizationMcpServerWorkspace:
        server = await self._organization_server(user, server_id)
        names = await self._workspace_names(user)
        if workspace_id not in names:
            raise WorkspaceNotFoundError(workspace_id)
        async with self.uow:
            await self.lock_workspace(workspace_id)
            override = await self.overrides.find(workspace_id=workspace_id, server_id=server.id)
            # The workspace's own settings stay as its admins left them; only the limit changes.
            override = await self.overrides.save(
                override,
                workspace_id=workspace_id,
                server=server,
                organization_tools=tools,
                enabled=True if override is None else override.enabled,
                everyone_tools=None if override is None else override.everyone_tools,
                attach_by_default=None if override is None else override.attach_by_default,
            )
            return OrganizationMcpServerWorkspace.of(workspace_id, names[workspace_id], server, override)

    async def _organization_server(self, user: User, server_id: uuid.UUID) -> WorkspaceMcpServer:
        """One of the caller's organization's own servers, once the caller is confirmed one of its owners or admins."""
        organization = await self.organizations.get_active_organization_for_user(user)
        await self.organizations.require_active_organization_management_access(user=user, organization=organization)
        async with self.uow:
            server = await self.servers.find(server_id)
        if server is None or server.workspace_id is not None or server.organization_id != organization.id:
            raise WorkspaceMcpServerNotFoundError(server_id)
        return server

    async def _workspace_names(self, user: User) -> dict[uuid.UUID, str]:
        """Every workspace of the caller's organization, by id, which an owner or admin may list."""
        names: dict[uuid.UUID, str] = {}
        skip = 0
        while True:
            page = await self.organizations.list_app_workspaces(user=user, skip=skip, limit=_WORKSPACE_PAGE)
            names.update({workspace.id: workspace.name for workspace in page.data})
            skip += len(page.data)
            if not page.data or skip >= page.count:
                return names
