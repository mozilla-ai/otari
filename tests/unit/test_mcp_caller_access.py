"""What the server list tells a caller about its own access, matched against what execution admits."""

from __future__ import annotations

import uuid

import pytest

from gateway.models.tools import WorkspaceMcpServer
from gateway.services.tenancy.workspace_mcp_server_service import _caller_access


def _server(allowed_tools: list[str] | None, *, enabled: bool = True) -> WorkspaceMcpServer:
    return WorkspaceMcpServer(
        workspace_id=uuid.uuid4(),
        name="drive",
        url="https://93.184.216.34/mcp",
        enabled=enabled,
        allowed_tools=allowed_tools,
    )


@pytest.mark.parametrize(
    ("allowed_tools", "tools", "access"),
    [
        (None, ["*"], "all"),
        # An explicit deny-all on the server admits nothing, whatever the caller's rule says.
        ([], ["*"], "none"),
        (["list_files"], ["*"], "some"),
        (None, ["@read"], "read"),
        (None, [], "none"),
    ],
)
def test_the_summary_matches_the_servers_allow_list(
    allowed_tools: list[str] | None, tools: list[str], access: str
) -> None:
    assert _caller_access(_server(allowed_tools), tools) == access


def test_a_disabled_server_is_reached_by_nobody() -> None:
    assert _caller_access(_server(None, enabled=False), ["*"]) == "none"
