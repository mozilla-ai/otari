"""The MCP access rules: tool patterns, who decides on a server, and admission."""

import uuid
from typing import Any, cast

import pytest

from gateway.exceptions.tools_exceptions import McpServerDeclarationError
from gateway.models.mcp import McpServerConfig, ResolvedMcpServer
from gateway.services.model_access import (
    MAX_MCP_TOOL_PATTERNS,
    McpAccess,
    is_mcp_tool_allowed,
    validate_mcp_tool_patterns,
)
from gateway.services.tenancy.workspace_mcp_server_service import WorkspaceMcpServers
from gateway.services.tools import admit_mcp_servers
from gateway.services.tools._mcp_admission import INLINE_MCP_SERVER_REFUSED_DETAIL

GITHUB = uuid.UUID(int=1)
JIRA = uuid.UUID(int=2)


def test_tool_patterns_match_exactly_by_prefix_or_everything() -> None:
    assert is_mcp_tool_allowed(["*"], "anything")
    assert is_mcp_tool_allowed(["get_*"], "get_issue")
    assert not is_mcp_tool_allowed(["get_*"], "list_issues")
    assert is_mcp_tool_allowed(["list_issues"], "list_issues")
    assert not is_mcp_tool_allowed(["list_issues"], "list_issues_v2")
    assert not is_mcp_tool_allowed([], "list_issues")


def test_validation_trims_collapses_duplicates_and_reduces_to_star() -> None:
    assert validate_mcp_tool_patterns([" get_issue ", "get_issue", "list_*"]) == ["get_issue", "list_*"]
    assert validate_mcp_tool_patterns(["get_issue", "*"]) == ["*"]
    assert validate_mcp_tool_patterns([]) == []


@pytest.mark.parametrize(
    "patterns",
    [None, "get_issue", [""], ["  "], [1], ["*_issue"], ["get_*_x"], ["x"] * (MAX_MCP_TOOL_PATTERNS + 1)],
)
def test_validation_refuses_a_bad_list(patterns: Any) -> None:
    with pytest.raises(ValueError):
        validate_mcp_tool_patterns(patterns)


def test_a_caller_without_grants_gets_each_servers_everyone_rule() -> None:
    access = McpAccess()
    assert access.decide(GITHUB, ["*"]) == ("everyone", ["*"])
    assert access.decide(JIRA, []) == ("everyone", [])


def test_a_keys_grant_decides_over_its_owners_and_both_over_everyone() -> None:
    access = McpAccess(key_grants={GITHUB: ("get_issue",)}, user_grants={GITHUB: ("*",), JIRA: ()})
    assert access.decide(GITHUB, ["*"]) == ("key", ["get_issue"])
    assert access.decide(JIRA, ["*"]) == ("user", [])
    assert access.patterns_for(uuid.UUID(int=3), ["list_*"]) == ["list_*"]


def test_a_grant_can_allow_what_everyone_is_denied() -> None:
    assert McpAccess(user_grants={GITHUB: ("*",)}).patterns_for(GITHUB, []) == ["*"]


def test_an_unrestricted_caller_reaches_everything() -> None:
    access = McpAccess(unrestricted=True, key_grants={GITHUB: ()})
    assert access.patterns_for(GITHUB, []) == ["*"]
    assert access.may_declare_servers


def test_the_workspace_switch_decides_whether_a_request_may_name_its_own_servers() -> None:
    assert McpAccess().may_declare_servers
    assert not McpAccess(request_servers_allowed=False).may_declare_servers
    assert McpAccess(unrestricted=True, request_servers_allowed=False).may_declare_servers


class _Servers:
    """Stored servers that record the access they were asked with."""

    def __init__(self) -> None:
        self.asked: list[McpAccess | None] = []

    async def resolve_many(
        self, workspace_id: uuid.UUID, server_ids: list[uuid.UUID], access: McpAccess | None = None
    ) -> list[McpServerConfig]:
        self.asked.append(access)
        return [McpServerConfig(name="github", url="https://93.184.216.34/mcp", caller_tool_patterns=["list_*"])]

    async def resolve_one(self, workspace_id: uuid.UUID, server_id: uuid.UUID) -> ResolvedMcpServer | None:
        return None


def _servers() -> tuple[_Servers, WorkspaceMcpServers]:
    fake = _Servers()
    return fake, cast(WorkspaceMcpServers, fake)


@pytest.mark.asyncio
async def test_admission_refuses_a_declared_server_where_the_workspace_turned_them_off() -> None:
    _, servers = _servers()

    with pytest.raises(McpServerDeclarationError) as raised:
        await admit_mcp_servers(
            [McpServerConfig(name="mine", url="https://93.184.216.34/mcp")],
            None,
            servers=servers,
            workspace_id=uuid.uuid4(),
            access=McpAccess(request_servers_allowed=False),
        )

    assert str(raised.value) == INLINE_MCP_SERVER_REFUSED_DETAIL


@pytest.mark.asyncio
async def test_admission_passes_the_callers_access_to_the_stored_servers_and_clears_inline_patterns() -> None:
    fake, servers = _servers()
    access = McpAccess(key_grants={GITHUB: ("list_*",)})
    inline = McpServerConfig(name="mine", url="https://93.184.216.34/mcp", caller_tool_patterns=["smuggled"])

    admitted = await admit_mcp_servers(
        [inline], [uuid.uuid4()], servers=servers, workspace_id=uuid.uuid4(), access=access
    )

    assert fake.asked == [access]
    assert admitted is not None
    assert [(server.name, server.caller_tool_patterns) for server in admitted] == [
        ("mine", None),
        ("github", ["list_*"]),
    ]


@pytest.mark.asyncio
async def test_a_callers_inline_servers_pass_where_the_workspace_allows_them() -> None:
    _, servers = _servers()

    admitted = await admit_mcp_servers(
        [McpServerConfig(name="mine", url="https://93.184.216.34/mcp")],
        None,
        servers=servers,
        workspace_id=uuid.uuid4(),
        access=McpAccess(),
    )

    assert admitted is not None
    assert [server.name for server in admitted] == ["mine"]


def test_the_callers_patterns_are_off_the_wire_the_schema_and_the_revision() -> None:
    config = McpServerConfig(name="s", url="https://93.184.216.34/mcp", caller_tool_patterns=["a*"])
    assert "caller_tool_patterns" not in config.model_dump()
    assert "caller_tool_patterns" not in McpServerConfig.model_json_schema()["properties"]

    plain = ResolvedMcpServer(id=uuid.UUID(int=1), name="s", url="https://93.184.216.34/mcp")
    other = ResolvedMcpServer(id=uuid.UUID(int=1), name="s", url="https://93.184.216.34/mcp", everyone_tools=[])
    assert plain.revision == other.revision
