"""Resolving workspace-scoped MCP server ids against this deployment's own rows."""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.models.mcp import MAX_MCP_SERVER_IDS
from gateway.services.tenancy import workspace_mcp_server_service
from gateway.services.tenancy.workspace_mcp_server_service import MAX_MCP_SERVERS_PER_WORKSPACE, WorkspaceMcpServers


def test_the_request_bound_admits_every_server_a_workspace_can_hold() -> None:
    """`MAX_MCP_SERVER_IDS` justifies itself by matching the per-workspace cap, so pin that.

    They are declared separately on purpose (a wire bound has to be checkable at
    parse time, with no database to consult), which is exactly what lets them
    drift. Raising the service cap alone would leave a workspace able to store
    more servers than one request may name, surfacing as a 422 with nothing
    pointing at the cause.
    """
    assert MAX_MCP_SERVER_IDS >= MAX_MCP_SERVERS_PER_WORKSPACE


@pytest.mark.asyncio
async def test_resolve_many_reads_the_workspaces_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    session = MagicMock()
    workspace_id = uuid.uuid4()
    ids = [uuid.uuid4(), uuid.uuid4()]
    stored: list[Any] = [MagicMock()]
    resolve = AsyncMock(return_value=stored)
    monkeypatch.setattr(workspace_mcp_server_service, "resolve_workspace_mcp_servers", resolve)

    out = await WorkspaceMcpServers(session).resolve_many(workspace_id, ids)

    assert out is stored
    resolve.assert_awaited_once_with(session, workspace_id=workspace_id, server_ids=ids)


@pytest.mark.asyncio
async def test_resolve_one_reads_the_workspaces_row(monkeypatch: pytest.MonkeyPatch) -> None:
    session = MagicMock()
    workspace_id = uuid.uuid4()
    server_id = uuid.uuid4()
    resolve = AsyncMock(return_value=None)
    monkeypatch.setattr(workspace_mcp_server_service, "resolve_workspace_mcp_server", resolve)

    out = await WorkspaceMcpServers(session).resolve_one(workspace_id, server_id)

    assert out is None
    resolve.assert_awaited_once_with(session, workspace_id=workspace_id, server_id=server_id)
