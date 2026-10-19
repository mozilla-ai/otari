"""Resolving one stored MCP server for the stateless MCP routes.

The port's answer is the authorization input for both stored-server endpoints,
and a failure to read it fails closed.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest

from gateway.api.routes.mcp import _Principal, _resolve_server
from gateway.exceptions.tools_exceptions import McpResolutionFailure, McpServerResolutionFailedError
from gateway.models.mcp import ResolvedMcpServer
from gateway.ports.mcp_server_port import McpServerScope
from gateway.services.mcp_stateless import (
    CODE_RESOLUTION_FAILED,
    CODE_SERVER_NOT_FOUND,
    ExecutionState,
    McpExecutionError,
)

SERVER_ID = uuid.UUID("2c948a61-dc96-4cd8-96bb-8e1434bf424e")


class _PortAnswering:
    """A port that gives ``resolve_one`` one fixed answer."""

    def __init__(self, answer: Any = None, raises: Exception | None = None) -> None:
        self._answer = answer
        self._raises = raises

    async def resolve_many(self, scope: McpServerScope, server_ids: list[uuid.UUID]) -> Any:
        raise NotImplementedError

    async def resolve_one(self, scope: McpServerScope, server_id: uuid.UUID) -> Any:
        if self._raises is not None:
            raise self._raises
        return self._answer


@pytest.mark.asyncio
async def test_an_id_reaching_nothing_is_server_not_found() -> None:
    principal = _Principal(workspace_id=uuid.uuid4())

    with pytest.raises(McpExecutionError) as raised:
        await _resolve_server(principal, _PortAnswering(answer=None), SERVER_ID)

    assert raised.value.code == CODE_SERVER_NOT_FOUND
    assert raised.value.execution_state is ExecutionState.NOT_STARTED
    assert raised.value.status_code == 404


@pytest.mark.asyncio
async def test_an_unreadable_answer_is_the_resolution_failure_this_contract_publishes() -> None:
    """The port's failure reaches the caller as a 502, not as an unhandled 500."""
    principal = _Principal(workspace_id=None)
    port = _PortAnswering(raises=McpServerResolutionFailedError(McpResolutionFailure.NO_WORKSPACE))

    with pytest.raises(McpExecutionError) as raised:
        await _resolve_server(principal, port, SERVER_ID)

    assert raised.value.code == CODE_RESOLUTION_FAILED
    assert raised.value.execution_state is ExecutionState.NOT_STARTED
    assert raised.value.status_code == 502


@pytest.mark.asyncio
async def test_a_disabled_server_is_indistinguishable_from_a_missing_one() -> None:
    server = ResolvedMcpServer(id=SERVER_ID, name="github", url="https://mcp.example.com/mcp", enabled=False)

    with pytest.raises(McpExecutionError) as raised:
        await _resolve_server(_Principal(workspace_id=uuid.uuid4()), _PortAnswering(answer=server), SERVER_ID)

    assert raised.value.code == CODE_SERVER_NOT_FOUND
    assert raised.value.status_code == 404


@pytest.mark.asyncio
async def test_an_enabled_server_is_returned() -> None:
    server = ResolvedMcpServer(id=SERVER_ID, name="github", url="https://mcp.example.com/mcp", enabled=True)

    assert (
        await _resolve_server(_Principal(workspace_id=uuid.uuid4()), _PortAnswering(answer=server), SERVER_ID) is server
    )
