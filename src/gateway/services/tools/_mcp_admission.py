"""Admit the MCP servers one request may reach: its own, and the stored ones its IDs name."""

from __future__ import annotations

import asyncio
import uuid
from collections import Counter

from gateway.exceptions.tools_exceptions import McpServerConfigurationError, McpServerDeclarationError
from gateway.log_config import logger
from gateway.models.mcp import McpServerConfig
from gateway.ports.mcp_server_port import McpServerPort, McpServerScope
from gateway.services.secret_box import SecretBoxUnavailableError, SecretDecryptionError
from gateway.services.url_safety import UnsafeURLError, validate_mcp_url

MCP_SERVER_TOKEN_UNREADABLE_DETAIL = "A configured MCP server's authorization token could not be read"
MCP_SERVER_URL_UNSAFE_DETAIL = "A configured MCP server's URL failed its safety check"
MCP_SERVER_NAME_COLLIDES_WITH_STORED_DETAIL = (
    "A request-supplied MCP server name collides with one of this workspace's stored servers"
)
MCP_SERVER_NAMES_NOT_UNIQUE_DETAIL = "Configured MCP servers do not have unique names"


def duplicate_mcp_server_name_detail(name: str) -> str:
    """400 detail for two entries in the caller's own ``mcp_servers`` list sharing a name."""
    return f"Duplicate MCP server name {name!r}. mcp_servers entries must have unique names."


async def _validate_mcp_server_urls(
    mcp_servers: list[McpServerConfig],
    *,
    stored: bool = False,
    workspace_id: uuid.UUID | None = None,
) -> None:
    """SSRF/scheme safety check for the MCP server URLs in this request.

    Called once per source rather than over the merged list, because a failure
    means different things for the two and the caller is owed a different
    answer:

    * A **request-body** server is the caller's own, so a rejection is their
      malformed request and the reason travels back to them, naming the URL they
      sent.
    * A **stored** server (resolved from ``mcp_server_ids``) is workspace
      configuration the caller can neither see nor fix, and the rejection names
      the host and the range it resolved into. ``routes/workspace_mcp_servers``
      gates even the *read* of those rows behind the master key, on the grounds
      that they name the endpoints this gateway connects to, so echoing one to
      any key holder gives away what that gate is there to withhold. It answers
      the way an unreadable stored token already does: a fixed 500 detail, with
      the reason and the workspace in the log, since a stored endpoint that
      fails its check is the operator's problem and not the caller's.

    Runs concurrently since each check does an independent DNS lookup;
    ``asyncio.gather`` (default ``return_exceptions=False``) propagates the
    first ``UnsafeURLError`` it sees as soon as it's raised. Note this does
    *not* cancel the other in-flight checks: they keep running in the
    background and are simply not awaited further; harmless here since
    ``validate_mcp_url`` has no side effects beyond a DNS lookup.

    It runs here rather than in a Pydantic validator because the DNS lookup must
    be awaited, so a rejected URL answers ``400`` rather than ``422``.
    """
    try:
        await asyncio.gather(
            *(
                validate_mcp_url(server.url, has_authorization_token=bool(server.authorization_token))
                for server in mcp_servers
            )
        )
    except UnsafeURLError as exc:
        if not stored:
            raise McpServerDeclarationError(str(exc)) from exc
        logger.error("Configured MCP server URL failed its safety check for workspace %s: %s", workspace_id, exc)
        raise McpServerConfigurationError(MCP_SERVER_URL_UNSAFE_DETAIL) from exc


async def admit_mcp_servers(
    inline_servers: list[McpServerConfig] | None,
    server_ids: list[uuid.UUID] | None,
    *,
    port: McpServerPort,
    scope: McpServerScope,
) -> list[McpServerConfig] | None:
    """The MCP servers the request may reach: its own, then the stored ones its IDs name.

    Every deployment refuses an unknown ID with a 404, so the status a caller
    sees does not change with the deployment it reached.

    Raises:
        McpServerDeclarationError: the request's own servers cannot be reached as declared.
        McpServerConfigurationError: the stored servers cannot be used, which is the operator's to fix.
        WorkspaceMcpServerNotFoundError: an ID names no server this workspace holds.
        McpServerResolutionFailedError: the peer that holds the servers could not answer.
    """
    mcp_servers = inline_servers
    # Each source is checked on its own, because a stored server's refusal carries a fixed detail.
    # A duplicate name collapses two servers into one client session, so each source refuses one.
    inline_names: set[str] = set()
    if mcp_servers:
        # Before the URL check, which resolves DNS for each server.
        for server in mcp_servers:
            if server.name in inline_names:
                raise McpServerDeclarationError(duplicate_mcp_server_name_detail(server.name))
            inline_names.add(server.name)
        await _validate_mcp_server_urls(mcp_servers)
    if server_ids:
        try:
            stored_servers = await port.resolve_many(scope, server_ids)
        except (SecretBoxUnavailableError, SecretDecryptionError) as exc:
            # The operator's problem, not the caller's, and the underlying message
            # names the environment variable, so it stays in the log.
            logger.error("MCP server token could not be decrypted for workspace %s: %s", scope.workspace_id, exc)
            raise McpServerConfigurationError(MCP_SERVER_TOKEN_UNREADABLE_DETAIL) from exc
        await _validate_mcp_server_urls(stored_servers, stored=True, workspace_id=scope.workspace_id)
        stored_name_counts = Counter(server.name for server in stored_servers)
        # Only a peer's answer can repeat a name. The caller cannot fix it, so the names go to the log.
        if len(stored_name_counts) != len(stored_servers):
            logger.error(
                "Stored MCP servers do not have unique names for workspace %s: %s",
                scope.workspace_id,
                sorted(name for name, count in stored_name_counts.items() if count > 1),
            )
            raise McpServerConfigurationError(MCP_SERVER_NAMES_NOT_UNIQUE_DETAIL)
        # Gotcha: the detail does not repeat a stored name, but a caller can still guess one by probing.
        if inline_names & stored_name_counts.keys():
            raise McpServerDeclarationError(MCP_SERVER_NAME_COLLIDES_WITH_STORED_DETAIL)
        mcp_servers = (mcp_servers or []) + stored_servers
    return mcp_servers
