"""How a workspace sees an organization's MCP server, and clipping a caller's tools to the organization's limit."""

import uuid

import pytest

from gateway.models.tools import WorkspaceMcpServer, WorkspaceMcpServerOverride
from gateway.services.model_access import McpAccess, clip_mcp_tool_patterns, is_mcp_tool_allowed
from gateway.services.tools import McpServerInWorkspace

TOOLS = ["search_issues", "search_users", "list_issues", "delete_issue", "s"]


@pytest.mark.parametrize(
    ("patterns", "limit"),
    [
        (["*"], ["search*"]),
        (["search*"], ["*"]),
        (["s*"], ["search_*"]),
        (["search_*"], ["s*"]),
        (["search_issues", "delete_issue"], ["search*"]),
        (["list_issues"], ["search*"]),
        (["search*", "list_issues"], ["search_users", "list_*"]),
        (["*"], []),
        ([], ["*"]),
    ],
)
def test_a_clipped_rule_admits_exactly_the_tools_both_rules_admit(patterns: list[str], limit: list[str]) -> None:
    clipped = clip_mcp_tool_patterns(patterns, limit)

    for tool in TOOLS:
        both = is_mcp_tool_allowed(patterns, tool) and is_mcp_tool_allowed(limit, tool)
        assert is_mcp_tool_allowed(clipped, tool) is both, tool


def _server(**fields: object) -> WorkspaceMcpServer:
    values: dict[str, object] = {
        "id": uuid.uuid4(),
        "organization_id": uuid.uuid4(),
        "workspace_id": None,
        "name": "jira",
        "url": "https://jira.example/mcp",
        "enabled": True,
        "everyone_tools": ["*"],
        "attach_by_default": False,
        "read_tools": ["list_issues"],
    }
    values.update(fields)
    return WorkspaceMcpServer(**values)


def _override(server: WorkspaceMcpServer, **fields: object) -> WorkspaceMcpServerOverride:
    values: dict[str, object] = {"enabled": True, "organization_tools": None, "everyone_tools": None}
    values.update(fields)
    return WorkspaceMcpServerOverride(
        workspace_id=uuid.uuid4(), organization_id=server.organization_id, server_id=server.id, **values
    )


def test_a_workspace_s_own_server_reads_as_its_row() -> None:
    server = _server(workspace_id=uuid.uuid4(), enabled=False, everyone_tools=["search*"], attach_by_default=True)
    view = McpServerInWorkspace(server)

    assert (view.visible, view.enabled, view.everyone_tools, view.attach_by_default, view.tool_limit) == (
        True,
        False,
        ["search*"],
        True,
        None,
    )


def test_an_organization_server_reads_through_the_workspace_s_override() -> None:
    server = _server(everyone_tools=["list_issues"], attach_by_default=True)
    override = _override(server, everyone_tools=["*"], attach_by_default=False, organization_tools=["@read"])
    view = McpServerInWorkspace(server, override)

    assert (view.everyone_tools, view.attach_by_default, view.tool_limit) == (["*"], False, ["list_issues"])
    assert view.patterns_for(McpAccess()) == ["list_issues"]
    assert view.patterns_for(McpAccess(unrestricted=True)) == ["list_issues"], "not even the master key widens it"
    assert McpServerInWorkspace(server).everyone_tools == ["list_issues"], "no override: the server's own"


@pytest.mark.parametrize(
    ("server_fields", "override_fields"),
    [({"enabled": False}, None), ({}, {"enabled": False}), ({}, {"organization_tools": []})],
)
def test_an_organization_server_can_be_turned_off_for_a_workspace_three_ways(
    server_fields: dict[str, object], override_fields: dict[str, object] | None
) -> None:
    server = _server(**server_fields)
    override = _override(server, **override_fields) if override_fields is not None else None

    assert McpServerInWorkspace(server, override).visible is False
