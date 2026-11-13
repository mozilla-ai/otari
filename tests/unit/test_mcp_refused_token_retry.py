"""A per-user MCP server that refuses its token with a 401 gets one retry with a renewed one, on both call paths."""

from __future__ import annotations

import asyncio
import uuid
from contextlib import AsyncExitStack
from typing import Any, cast

import httpx2
import pytest
from mcp.types import ListToolsResult

from gateway.exceptions.tools_exceptions import McpSignInRequiredError
from gateway.models.mcp import McpServerConfig, ResolvedMcpServer
from gateway.services import mcp_stateless
from gateway.services.mcp_client import MCPClientPool
from gateway.services.mcp_stateless import CODE_SIGN_IN_REQUIRED, McpExecutionError

pytestmark = pytest.mark.asyncio

URL = "https://93.184.216.34/mcp"


def _refused(status: int = 401) -> BaseExceptionGroup[Exception]:
    request = httpx2.Request("POST", URL)
    return BaseExceptionGroup(
        "transport",
        [httpx2.HTTPStatusError("refused", request=request, response=httpx2.Response(status, request=request))],
    )


class _Renewer:
    def __init__(self, *, answer: str | None = "renewed") -> None:
        self.asked: list[str] = []
        self.answer = answer

    async def __call__(self, rejected: str) -> str:
        self.asked.append(rejected)
        if self.answer is None:
            raise McpSignInRequiredError(uuid.uuid4(), name="drive", user_id="u")
        return self.answer


def _server(renew: _Renewer | None) -> ResolvedMcpServer:
    return ResolvedMcpServer(id=uuid.uuid4(), name="drive", url=URL, authorization_token="stale", renew_token=renew)


def _stateless_network(monkeypatch: pytest.MonkeyPatch, *, refuse: set[str], status: int = 401) -> list[str]:
    """Stand in for connecting: a token in ``refuse`` is answered ``status``. Returns the tokens tried."""
    tried: list[str] = []

    async def enter(stack: AsyncExitStack, server: ResolvedMcpServer, *, screen_answers: bool) -> Any:
        tried.append(server.authorization_token or "")
        if server.authorization_token in refuse:
            raise _refused(status)
        return object()

    monkeypatch.setattr(mcp_stateless, "_enter_session", enter)
    return tried


async def test_a_refused_token_is_renewed_and_the_session_opened_once_more(monkeypatch: pytest.MonkeyPatch) -> None:
    tried = _stateless_network(monkeypatch, refuse={"stale"})
    renew = _Renewer()

    async with mcp_stateless.open_session(_server(renew)):
        pass

    assert (tried, renew.asked) == (["stale", "renewed"], ["stale"])


async def test_a_refusal_screened_as_a_check_screens_it_is_renewed_too(monkeypatch: pytest.MonkeyPatch) -> None:
    """A manual check screens answers, so its 401 arrives already named rather than as an HTTP error."""
    tried: list[str] = []

    async def enter(stack: AsyncExitStack, server: ResolvedMcpServer, *, screen_answers: bool) -> Any:
        assert screen_answers
        tried.append(server.authorization_token or "")
        if server.authorization_token == "stale":
            raise BaseExceptionGroup(
                "transport", [mcp_stateless._UnusableAnswer(mcp_stateless.CheckFailure.UNAUTHORIZED, 401)]
            )
        return object()

    monkeypatch.setattr(mcp_stateless, "_enter_session", enter)
    renew = _Renewer()

    async with mcp_stateless.open_session(_server(renew), screen_answers=True):
        pass

    assert (tried, renew.asked) == (["stale", "renewed"], ["stale"])


async def test_a_second_refusal_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    tried = _stateless_network(monkeypatch, refuse={"stale", "renewed"})

    with pytest.raises(BaseExceptionGroup):
        async with mcp_stateless.open_session(_server(_Renewer())):
            pass

    assert tried == ["stale", "renewed"]


@pytest.mark.parametrize("status", [403, 500])
async def test_another_failure_is_not_retried(monkeypatch: pytest.MonkeyPatch, status: int) -> None:
    tried = _stateless_network(monkeypatch, refuse={"stale"}, status=status)
    renew = _Renewer()

    with pytest.raises(BaseExceptionGroup):
        async with mcp_stateless.open_session(_server(renew)):
            pass

    assert (tried, renew.asked) == (["stale"], [])


async def test_a_shared_server_has_nothing_to_renew_and_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    tried = _stateless_network(monkeypatch, refuse={"stale"})

    with pytest.raises(BaseExceptionGroup):
        async with mcp_stateless.open_session(_server(None)):
            pass

    assert tried == ["stale"]


async def test_a_sign_in_that_ended_is_answered_sign_in_required(monkeypatch: pytest.MonkeyPatch) -> None:
    _stateless_network(monkeypatch, refuse={"stale"})

    with pytest.raises(McpExecutionError) as raised:
        async with mcp_stateless.open_session(_server(_Renewer(answer=None))):
            pass

    assert (raised.value.code, raised.value.status_code) == (CODE_SIGN_IN_REQUIRED, 403)


async def test_the_managed_loop_renews_a_refused_token_once(monkeypatch: pytest.MonkeyPatch) -> None:
    tried: list[str] = []
    renew = _Renewer()

    async def fake_open(self: MCPClientPool, stack: AsyncExitStack, cfg: McpServerConfig, *_: object) -> Any:
        tried.append(cfg.authorization_token or "")
        if cfg.authorization_token == "stale":
            raise _refused()
        return object(), ListToolsResult(tools=[])

    monkeypatch.setattr(MCPClientPool, "_open", fake_open)
    config = McpServerConfig(name="drive", url=URL, authorization_token="stale", renew_token=renew)

    async with MCPClientPool([config]):
        pass

    assert (tried, renew.asked) == (["stale", "renewed"], ["stale"])


async def test_the_managed_loop_offers_a_sign_in_when_the_sign_in_ended(monkeypatch: pytest.MonkeyPatch) -> None:
    """The request goes on without the server, and the pool names it so the model can offer a new sign-in."""

    async def fake_open(self: MCPClientPool, stack: AsyncExitStack, cfg: McpServerConfig, *_: object) -> Any:
        raise _refused()

    monkeypatch.setattr(MCPClientPool, "_open", fake_open)
    config = McpServerConfig(name="drive", url=URL, authorization_token="stale", renew_token=_Renewer(answer=None))

    async with MCPClientPool([config]) as pool:
        assert pool.openai_tools == []
        [(signed_out, refusal)] = pool.signed_out

    assert signed_out.name == "drive"
    assert isinstance(refusal, McpSignInRequiredError)


async def test_a_request_cannot_send_its_own_renewer() -> None:
    """Admission's field: a request body that names one gets it cleared, as the other admission fields are."""
    from gateway.services.model_access import McpAccess
    from gateway.services.tenancy.workspace_mcp_server_service import WorkspaceMcpServers
    from gateway.services.tools._mcp_admission import admit_mcp_servers

    inline = McpServerConfig(name="mine", url=URL, renew_token=_Renewer())

    class _NoServers:
        async def resolve_many(self, *args: Any) -> list[McpServerConfig]:
            return []

    async def safe(url: str, *, has_authorization_token: bool) -> None:
        return None

    with pytest.MonkeyPatch.context() as patched:
        patched.setattr("gateway.services.tools._mcp_admission.validate_mcp_url", safe)
        admitted = await admit_mcp_servers(
            [inline],
            None,
            servers=cast(WorkspaceMcpServers, _NoServers()),
            workspace_id=uuid.uuid4(),
            access=McpAccess(),
        )

    assert admitted.servers is not None and admitted.servers[0].renew_token is None


async def test_a_failed_open_whose_transport_never_closes_still_answers_with_its_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The close after a failed open is bounded as the close after a finished call is."""
    monkeypatch.setattr(mcp_stateless, "CLEANUP_TIMEOUT_S", 0.05)

    async def never_closes(*_exc: object) -> None:
        await asyncio.Event().wait()

    async def enter(stack: AsyncExitStack, server: ResolvedMcpServer, *, screen_answers: bool) -> Any:
        stack.push_async_exit(never_closes)
        raise _refused(500)

    monkeypatch.setattr(mcp_stateless, "_enter_session", enter)

    async with asyncio.timeout(2):
        with pytest.raises(BaseExceptionGroup) as raised:
            async with mcp_stateless.open_session(_server(None)):
                pass

    assert not mcp_stateless.refused_token(raised.value)
