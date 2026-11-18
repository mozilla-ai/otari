"""Folding workspaces' own copies of a server into their organization's: what each keeps, and the fold itself.

A copy is a workspace's own server at the same URL as the organization's. Folding it moves its access rules
into the workspace's settings on the organization's server, keeps its id working there, and deletes it.

What happens to the people signed in to it:

- A personal sign-in made through the same OAuth client as the organization server's own (the same
  ``client_id``, which the gateway's client metadata document gives every server it signs in to through it)
  can be renewed and revoked through the organization server, so it is carried across.
- Any other sign-in is revoked once the fold commits: its client is the copy's, which goes with it. So is a
  carried sign-in's twin, where the person already holds one on the organization server, so nobody is moved
  to another upstream account without knowing.
- A shared account or a token is never carried: the organization server's own credential serves every
  workspace, and keeping a copy's would give one workspace another account.

The counts a fold step shows come from the same rule, before anyone confirms.
"""

import uuid
from collections.abc import Collection, Sequence
from dataclasses import dataclass

from pydantic import BaseModel, Field

from gateway.models.tools import McpCredentialMode, McpUserConnection, WorkspaceMcpServer
from gateway.repositories.tools import (
    McpOAuthFlowRepository,
    McpServerAliasRepository,
    McpServerRequestRepository,
    McpUserConnectionRepository,
    WorkspaceMcpServerGrantRepository,
    WorkspaceMcpServerOverrideRepository,
    WorkspaceMcpServerRepository,
)
from gateway.services.model_access import expand_mcp_tool_patterns, mcp_read_tools
from gateway.services.tools._mcp_connections import McpRevocation, revocations_for, stored_oauth_client


class McpServerFoldCandidate(BaseModel):
    """A workspace's own copy of a server, and what folding it into the organization's would do."""

    server_id: uuid.UUID
    workspace_id: uuid.UUID
    workspace_name: str
    name: str
    credential_mode: McpCredentialMode
    same_credential: bool = Field(
        description="It signs in the way the organization's server does and holds no token or shared account of "
        "its own, so folding it changes no account its calls use. A fold step ticks these by default"
    )
    signed_in: int = Field(description="People signed in to it with their own account")
    keep_sign_in: int = Field(description="Of those, how many keep their sign-in on the organization's server")
    sign_in_again: int = Field(description="Of those, how many must sign in again")
    replaces_shared_credential: bool = Field(
        description="It holds a token or a shared account, which the organization server's credential replaces"
    )


@dataclass(frozen=True, slots=True)
class _Plan:
    """One copy's sign-ins, split by what a fold into ``target_client_id`` does with them."""

    carried: list[McpUserConnection]
    revoked: list[McpUserConnection]


@dataclass(frozen=True, slots=True)
class McpServerFoldOutcome:
    """What a fold left to do once it commits, and what it did."""

    revocations: list[McpRevocation]
    carried: int
    revoked: int


class McpServerFold:
    """Weighs and folds workspaces' copies of a server into their organization's, in the caller's transaction."""

    def __init__(
        self,
        *,
        servers: WorkspaceMcpServerRepository,
        grants: WorkspaceMcpServerGrantRepository,
        overrides: WorkspaceMcpServerOverrideRepository,
        aliases: McpServerAliasRepository,
        connections: McpUserConnectionRepository,
        flows: McpOAuthFlowRepository,
        requests: McpServerRequestRepository,
    ) -> None:
        self._servers = servers
        self._grants = grants
        self._overrides = overrides
        self._aliases = aliases
        self._connections = connections
        self._flows = flows
        self._requests = requests

    async def copies(
        self, organization_id: uuid.UUID, url: str, *, except_id: uuid.UUID | None = None
    ) -> list[WorkspaceMcpServer]:
        """The organization's workspaces' own servers at ``url``, oldest first."""
        return await self._servers.copies_in_organization(organization_id, url, except_id=except_id)

    async def weigh(
        self,
        copies: Sequence[WorkspaceMcpServer],
        *,
        target: WorkspaceMcpServer | None,
        credential_mode: McpCredentialMode,
        workspace_names: dict[uuid.UUID, str],
    ) -> list[McpServerFoldCandidate]:
        """What folding each copy into ``target`` would do; with no ``target`` yet, a server of ``credential_mode``."""
        signed_in_there = await self._signed_in(target)
        client_id = _client_id(target)
        accounts = await self._connections.shared_for_servers([copy.id for copy in copies])
        weighed: list[McpServerFoldCandidate] = []
        for copy in copies:
            if copy.workspace_id is None:
                continue
            plan = _plan(await self._connections.every_for_server(copy.id), client_id, signed_in_there)
            own_credential = copy.encrypted_token is not None or copy.id in accounts
            weighed.append(
                McpServerFoldCandidate(
                    server_id=copy.id,
                    workspace_id=copy.workspace_id,
                    workspace_name=workspace_names.get(copy.workspace_id, ""),
                    name=copy.name,
                    credential_mode=McpCredentialMode(copy.credential_mode),
                    same_credential=copy.credential_mode == credential_mode and not own_credential,
                    signed_in=len(plan.carried) + len(plan.revoked),
                    keep_sign_in=len(plan.carried),
                    sign_in_again=len(plan.revoked),
                    replaces_shared_credential=own_credential,
                )
            )
            # A person carried from this copy holds a sign-in on the target from then on.
            signed_in_there |= {connection.user_id for connection in plan.carried}
        return weighed

    async def fold(self, target: WorkspaceMcpServer, copies: Sequence[WorkspaceMcpServer]) -> McpServerFoldOutcome:
        """Fold these copies into the organization's ``target``, staging every change; the caller commits.

        The returned revocations are for the caller to send once the fold has committed.
        """
        signed_in_there = await self._signed_in(target)
        client_id = _client_id(target)
        revocations: list[McpRevocation] = []
        carried = revoked = 0
        for copy in copies:
            workspace_id = copy.workspace_id
            if (
                workspace_id is None
                or target.workspace_id is not None
                or copy.organization_id != target.organization_id
            ):
                raise ValueError(f"MCP server {copy.id} is not a workspace's copy of an organization's server")
            plan = _plan(await self._connections.every_for_server(copy.id), client_id, signed_in_there)
            account = await self._connections.find_shared(copy.id)
            # Read while the copy still holds the client its tokens were issued to.
            revocations += revocations_for(copy, [*plan.revoked, *([account] if account is not None else [])])
            for connection in plan.carried:
                await self._connections.move(connection, target.id)
                signed_in_there.add(connection.user_id)
            carried += len(plan.carried)
            revoked += len(plan.revoked)
            await self._move_rules(copy, target, workspace_id)
            await self._aliases.add(copy.id, workspace_id=workspace_id, server=target)
            await self._requests.repoint(copy.id, target.id)
            # Staged as well as cascaded by the keys, so they go where a database enforces none.
            await self._connections.remove_for_server(copy.id)
            await self._connections.remove_shared_for_server(copy.id)
            await self._flows.remove_for_server(copy.id)
            await self._servers.delete(copy)
        return McpServerFoldOutcome(revocations=revocations, carried=carried, revoked=revoked)

    async def _move_rules(self, copy: WorkspaceMcpServer, target: WorkspaceMcpServer, workspace_id: uuid.UUID) -> None:
        """The copy's rules become its workspace's settings and grants on the target.

        ``@read`` is written out as the copy's own read tools, which the target's may not match, so no rule
        starts reaching more or fewer tools. A person or key with a grant on the target in the workspace keeps it.
        """
        read_tools = mcp_read_tools(copy.read_tools, copy.read_tools_hinted)
        everyone = expand_mcp_tool_patterns(copy.everyone_tools, read_tools)
        override = await self._overrides.find(workspace_id=workspace_id, server_id=target.id)
        await self._overrides.save(
            override,
            workspace_id=workspace_id,
            server=target,
            organization_tools=None if override is None else override.organization_tools,
            enabled=copy.enabled,
            everyone_tools=None if everyone == list(target.everyone_tools) else everyone,
            attach_by_default=None if copy.attach_by_default == target.attach_by_default else copy.attach_by_default,
        )
        held = {(grant.api_key_id, grant.user_id) for grant in await self._grants.for_server(target.id, workspace_id)}
        for grant in await self._grants.for_server(copy.id):
            if (grant.api_key_id, grant.user_id) in held:
                continue
            await self._grants.put(
                target,
                workspace_id=workspace_id,
                api_key_ids=[grant.api_key_id] if grant.api_key_id is not None else [],
                user_ids=[grant.user_id] if grant.user_id is not None else [],
                tools=expand_mcp_tool_patterns(grant.tools, read_tools),
            )

    async def _signed_in(self, target: WorkspaceMcpServer | None) -> set[str]:
        if target is None:
            return set()
        return {connection.user_id for connection in await self._connections.every_for_server(target.id)}


def _client_id(target: WorkspaceMcpServer | None) -> str | None:
    """The OAuth client a sign-in must have been made through to be carried to ``target``; ``None`` carries none."""
    client = stored_oauth_client(target) if target is not None else None
    return client.client_id if client is not None else None


def _plan(connections: Collection[McpUserConnection], client_id: str | None, signed_in_there: set[str]) -> _Plan:
    carried = [
        connection
        for connection in connections
        if client_id is not None and connection.client_id == client_id and connection.user_id not in signed_in_there
    ]
    kept = {connection.id for connection in carried}
    return _Plan(carried=carried, revoked=[connection for connection in connections if connection.id not in kept])
