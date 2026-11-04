"""Resolving one stored MCP server for the stateless MCP routes.

The stored row is the authorization input for both stored-server endpoints.
"""

from __future__ import annotations

import uuid
from typing import Any, cast

import pytest

from gateway.api.routes.mcp import _Principal, _resolve_server
from gateway.models.mcp import ResolvedMcpServer
from gateway.services.mcp_stateless import (
    CODE_SERVER_NOT_FOUND,
    ExecutionState,
    McpExecutionError,
)
from gateway.services.tenancy.workspace_mcp_server_service import WorkspaceMcpServers

SERVER_ID = uuid.UUID("2c948a61-dc96-4cd8-96bb-8e1434bf424e")


class _Answering:
    """Servers that give ``resolve_one`` one fixed answer."""

    def __init__(self, answer: Any = None) -> None:
        self._answer = answer

    async def resolve_one(self, workspace_id: uuid.UUID, server_id: uuid.UUID) -> Any:
        return self._answer


def _servers(answer: Any = None) -> WorkspaceMcpServers:
    return cast(WorkspaceMcpServers, _Answering(answer))


@pytest.mark.asyncio
async def test_an_id_reaching_nothing_is_server_not_found() -> None:
    principal = _Principal(workspace_id=uuid.uuid4())

    with pytest.raises(McpExecutionError) as raised:
        await _resolve_server(principal, _servers(), SERVER_ID)

    assert raised.value.code == CODE_SERVER_NOT_FOUND
    assert raised.value.execution_state is ExecutionState.NOT_STARTED
    assert raised.value.status_code == 404


@pytest.mark.asyncio
async def test_a_disabled_server_is_indistinguishable_from_a_missing_one() -> None:
    server = ResolvedMcpServer(id=SERVER_ID, name="github", url="https://mcp.example.com/mcp", enabled=False)

    with pytest.raises(McpExecutionError) as raised:
        await _resolve_server(_Principal(workspace_id=uuid.uuid4()), _servers(server), SERVER_ID)

    assert raised.value.code == CODE_SERVER_NOT_FOUND
    assert raised.value.status_code == 404


@pytest.mark.asyncio
async def test_an_enabled_server_is_returned() -> None:
    server = ResolvedMcpServer(id=SERVER_ID, name="github", url="https://mcp.example.com/mcp", enabled=True)

    assert await _resolve_server(_Principal(workspace_id=uuid.uuid4()), _servers(server), SERVER_ID) is server
