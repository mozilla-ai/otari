"""Folding workspaces' own copies of a server into their organization's: what each keeps, and the fold itself.

A copy is a workspace's own server at the same URL as the organization's. Folding it moves its access rules
into the workspace's settings on the organization's server, keeps its id working there, and deletes it.

What happens to the people signed in to it:

- A personal sign-in is carried across where the organization server is personal too, sits at the very URL the
  sign-in was made for, and can renew it: the sign-in holds a refresh token, the same OAuth client issued it
  (the same ``client_id``, which the gateway's client metadata document gives every server it signs in to
  through it), the organization server holds the secret the sign-in was made with if it sent one, and that client
  may send its credentials to the token endpoint the sign-in recorded.
- Any other sign-in is revoked once the fold commits: its client is the copy's, which goes with it. So is a
  carried sign-in's twin, where the person already holds one on the organization server: they go on through
  that one, which may be another upstream account, and the fold step counts them apart so nobody is moved
  without the admin seeing it.
- A shared account or a token is never carried: the organization server's own credential serves every
  workspace, and keeping a copy's would give one workspace another account.

The counts a fold step shows come from the same rule, before anyone confirms.
"""

import uuid
from collections.abc import Collection, Sequence
from dataclasses import dataclass

from pydantic import BaseModel, Field

from gateway.models.tools import McpCredentialMode, McpSharedConnection, McpUserConnection, WorkspaceMcpServer
from gateway.repositories.tools import (
    McpOAuthFlowRepository,
    McpServerAliasRepository,
    McpServerRequestRepository,
    McpUserConnectionRepository,
    WorkspaceMcpServerGrantRepository,
    WorkspaceMcpServerOverrideRepository,
    WorkspaceMcpServerRepository,
)
from gateway.services.mcp_stateless import reads_as_pattern
from gateway.services.model_access import clip_mcp_tool_patterns, expand_mcp_tool_patterns, mcp_read_tools
from gateway.services.tools._mcp_connections import (
    MAX_BACKGROUND_REVOCATIONS,
    McpRevocation,
    renews_through,
    revocations_for,
)


class McpServerFoldCandidate(BaseModel):
    """A workspace's own copy of a server, and what folding it into the organization's would do."""

    server_id: uuid.UUID
    workspace_id: uuid.UUID
    workspace_name: str
    name: str
    credential_mode: McpCredentialMode
    enabled: bool = Field(
        default=True,
        description="Whether it is on. An organization server that is off cannot take the place of one that is on",
    )
    same_credential: bool = Field(
        description="It signs in the way the organization's server does and holds no token or shared account of "
        "its own, so folding it changes no account its calls use. A fold step ticks these by default, unless "
        "someone would have to sign in again or go on through another sign-in"
    )
    signed_in: int = Field(description="People signed in to it with their own account")
    keep_sign_in: int = Field(description="Of those, how many keep their sign-in on the organization's server")
    sign_in_again: int = Field(description="Of those, how many must sign in again")
    switch_account: int = Field(
        default=0,
        description="Of those, how many already hold a sign-in on the organization's server and go on through "
        "it, which may be another account upstream than the one they used here",
    )
    replaces_shared_credential: bool = Field(
        description="It holds a token or a shared account, which the organization server's credential replaces"
    )


@dataclass(frozen=True, slots=True)
class _Plan:
    """One copy's sign-ins, split by what a fold into its target does with them.

    ``revoked`` is every sign-in not carried; ``switched`` is those of them whose person holds one on the target.
    """

    carried: list[McpUserConnection]
    revoked: list[McpUserConnection]
    switched: int


@dataclass(frozen=True, slots=True)
class McpServerFoldOutcome:
    """What a fold left to do once it commits, and what it did.

    ``unread`` counts the ended sign-ins past what one background revocation sends, which are not prepared.
    """

    revocations: list[McpRevocation]
    carried: int
    revoked: int
    unread: int = 0


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
        self,
        organization_id: uuid.UUID,
        url: str,
        *,
        except_id: uuid.UUID | None = None,
        ids: Collection[uuid.UUID] | None = None,
        lock: bool = False,
    ) -> list[WorkspaceMcpServer]:
        """The organization's workspaces' own servers at ``url``, oldest first, the order a fold takes them in.

        ``ids`` keeps only those, and ``lock`` holds them until the caller commits, matched at ``url`` as they
        are once locked.
        """
        return await self._servers.copies_in_organization(organization_id, url, except_id=except_id, ids=ids, lock=lock)

    async def weigh(
        self,
        copies: Sequence[WorkspaceMcpServer],
        *,
        target: WorkspaceMcpServer | None,
        credential_mode: McpCredentialMode,
        workspace_names: dict[uuid.UUID, str],
        with_token: bool = False,
    ) -> list[McpServerFoldCandidate]:
        """What folding each copy into ``target`` would do.

        With no ``target`` yet, into a server of ``credential_mode``, holding a token of its own when ``with_token``.
        """
        signed_in_there = await self._signed_in(target)
        accounts = await self._connections.shared_for_servers([copy.id for copy in copies])
        # A copy with no credential of its own still changes account when the target brings one.
        if target is not None:
            with_token = target.encrypted_token is not None or bool(
                await self._connections.shared_for_servers([target.id])
            )
        weighed: list[McpServerFoldCandidate] = []
        for copy in copies:
            if copy.workspace_id is None:
                continue
            plan = _plan(await self._connections.every_for_server(copy.id), target, signed_in_there)
            own_credential = copy.encrypted_token is not None or copy.id in accounts
            weighed.append(
                McpServerFoldCandidate(
                    server_id=copy.id,
                    workspace_id=copy.workspace_id,
                    workspace_name=workspace_names.get(copy.workspace_id, ""),
                    name=copy.name,
                    credential_mode=McpCredentialMode(copy.credential_mode),
                    enabled=copy.enabled,
                    same_credential=copy.credential_mode == credential_mode and not own_credential and not with_token,
                    signed_in=len(plan.carried) + len(plan.revoked),
                    keep_sign_in=len(plan.carried),
                    sign_in_again=len(plan.revoked) - plan.switched,
                    switch_account=plan.switched,
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
        revocations: list[McpRevocation] = []
        carried = revoked = unread = 0
        for copy in copies:
            workspace_id = copy.workspace_id
            if (
                workspace_id is None
                or target.workspace_id is not None
                or copy.organization_id != target.organization_id
            ):
                raise ValueError(f"MCP server {copy.id} is not a workspace's copy of an organization's server")
            # Locked and read again, so a renewal in flight stores its token first and that is the one revoked.
            plan = _plan(await self._connections.every_for_server(copy.id, lock=True), target, signed_in_there)
            account = await self._connections.find_shared(copy.id, lock=True)
            # Read while the copy still holds the client its tokens were issued to, up to what one background
            # revocation sends.
            ending: list[McpUserConnection | McpSharedConnection] = [*plan.revoked, *([account] if account else [])]
            # A sign-in with nothing to revoke takes no room.
            read = 0
            while read < len(ending) and len(revocations) < MAX_BACKGROUND_REVOCATIONS:
                batch = ending[read : read + MAX_BACKGROUND_REVOCATIONS - len(revocations)]
                revocations += revocations_for(copy, batch)
                read += len(batch)
            unread += len(ending) - read
            await self._connections.move([connection.id for connection in plan.carried], target.id)
            signed_in_there |= {connection.user_id for connection in plan.carried}
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
        return McpServerFoldOutcome(revocations=revocations, carried=carried, revoked=revoked, unread=unread)

    async def _move_rules(self, copy: WorkspaceMcpServer, target: WorkspaceMcpServer, workspace_id: uuid.UUID) -> None:
        """The copy's rules become its workspace's settings and grants on the target.

        ``@read`` is written out as the copy's own read tools, which the target's may not match, so no rule
        starts reaching more or fewer tools. The copy's allowed tools stay the most its workspace reaches, as
        the organization's limit there. A person or key with a grant on the target in the workspace keeps it.
        """
        read_tools = mcp_read_tools(copy.read_tools, copy.read_tools_hinted)
        everyone = expand_mcp_tool_patterns(copy.everyone_tools, read_tools)
        override = await self._overrides.find(workspace_id=workspace_id, server_id=target.id)
        await self._overrides.save(
            override,
            workspace_id=workspace_id,
            server=target,
            organization_tools=_limit(copy, target, None if override is None else override.organization_tools),
            enabled=copy.enabled,
            # Both stored even where they equal the target's defaults, which can change later without the workspace.
            everyone_tools=everyone,
            attach_by_default=copy.attach_by_default,
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


def _limit(copy: WorkspaceMcpServer, target: WorkspaceMcpServer, held: list[str] | None) -> list[str] | None:
    """The organization's limit on the copy's workspace once folded: ``held``, narrowed to the copy's allowed tools.

    An unrestricted copy adds nothing. A restricted one is kept even where the target allows no more today, as
    the target's allow-list can widen later. An allowed tool is an exact name and a limit holds patterns, so a
    name that would read as one (``list*``, ``@read``) is left out: no tool so named is ever listed or called.
    """
    if copy.allowed_tools is None:
        return held
    allowed = [name for name in copy.allowed_tools if not reads_as_pattern(name)]
    if held is None:
        return allowed
    return clip_mcp_tool_patterns(
        allowed, expand_mcp_tool_patterns(held, mcp_read_tools(target.read_tools, target.read_tools_hinted))
    )


def _carries(target: WorkspaceMcpServer | None, connection: McpUserConnection) -> bool:
    """Whether ``target`` can take a copy's sign-in as its own, as the module docstring says."""
    return (
        target is not None
        and target.credential_mode == McpCredentialMode.PER_USER
        and connection.server_url == target.url
        and renews_through(target, connection)
    )


def _plan(
    connections: Collection[McpUserConnection], target: WorkspaceMcpServer | None, signed_in_there: set[str]
) -> _Plan:
    carried = [
        connection
        for connection in connections
        if _carries(target, connection) and connection.user_id not in signed_in_there
    ]
    kept = {connection.id for connection in carried}
    revoked = [connection for connection in connections if connection.id not in kept]
    switched = sum(1 for connection in revoked if connection.user_id in signed_in_there)
    return _Plan(carried=carried, revoked=revoked, switched=switched)
