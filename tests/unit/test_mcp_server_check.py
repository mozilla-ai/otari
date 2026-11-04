"""The stored-server connection check, against real servers on loopback.

The point of the check is naming why a server failed, and the SDK reports most
failures from inside a task group, wrapped in ways that changed between its
releases. So the causes are produced by real sockets and a real MCP server here
rather than by hand-built exceptions: a fake would pin the shape this code
expects, not the shape the SDK actually raises.
"""

from __future__ import annotations

import asyncio
import socket
import threading
import time
import uuid
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx2
import pytest
import uvicorn
from mcp.server.mcpserver import MCPServer
from mcp.shared.exceptions import MCPError

from gateway.models.mcp import ResolvedMcpServer
from gateway.services import mcp_stateless
from gateway.services.mcp_stateless import (
    CheckFailure,
    ConcurrencyGate,
    McpCapacityUnavailable,
    check_stored_server,
    classify_check_failure,
    refuse_unusable_answer,
)


def _server(url: str, *, allowed_tools: list[str] | None = None) -> ResolvedMcpServer:
    return ResolvedMcpServer(id=uuid.uuid4(), name="github", url=url, allowed_tools=allowed_tools)


class _Refusing(BaseHTTPRequestHandler):
    """Answers every request as its path says: a status code, an HTML page, or nothing for a while."""

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass

    def do_POST(self) -> None:
        self.rfile.read(int(self.headers.get("content-length", 0)))
        mode = self.path.strip("/")
        if mode == "hang":
            time.sleep(2)
            return
        if mode == "html":
            status, content_type, body = 200, "text/html", b"<html>Sign in</html>"
        else:
            status, content_type, body = int(mode), "application/json", b'{"detail": "no"}'
        self.send_response(status)
        if 300 <= status < 400:
            self.send_header("location", "https://elsewhere.example.com/mcp")
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture(scope="module")
def refusing() -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Refusing)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()


@pytest.fixture(scope="module", params=[False, True], ids=["sse", "json"])
def mcp_url(request: pytest.FixtureRequest) -> Iterator[str]:
    """A real streamable HTTP MCP server, answering in both of the transport's response modes."""
    app = MCPServer("check")

    @app.tool()
    def list_issues() -> str:
        return "[]"

    @app.tool()
    def close_issue(number: int) -> str:
        return str(number)

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(
            app.streamable_http_app(json_response=request.param), host="127.0.0.1", port=port, log_level="error"
        )
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    try:
        yield f"http://127.0.0.1:{port}/mcp"
    finally:
        server.should_exit = True
        thread.join(timeout=5)


@pytest.mark.asyncio
async def test_a_working_server_reports_its_tools(mcp_url: str) -> None:
    check = await check_stored_server(_server(mcp_url))

    assert check.failure is None
    assert sorted(check.tool_names) == ["close_issue", "list_issues"]


@pytest.mark.asyncio
async def test_a_check_lists_the_whole_catalog_whatever_the_allow_list(mcp_url: str) -> None:
    """The reader is choosing what to allow, so the catalog is not narrowed for them."""
    check = await check_stored_server(_server(mcp_url, allowed_tools=["list_issues", "missing_tool"]))

    assert check.failure is None
    assert sorted(check.tool_names) == ["close_issue", "list_issues"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("path", "failure", "http_status"),
    [
        ("401", CheckFailure.UNAUTHORIZED, 401),
        ("403", CheckFailure.FORBIDDEN, 403),
        # The SDK alone reports this one as a terminated session.
        ("404", CheckFailure.NOT_FOUND, 404),
        ("307", CheckFailure.REDIRECTED, 307),
        ("406", CheckFailure.HTTP_ERROR, 406),
        ("502", CheckFailure.SERVER_ERROR, 502),
        # And this one as nothing at all, until the connect deadline ran out.
        ("html", CheckFailure.NOT_MCP, None),
    ],
)
async def test_a_refusal_is_named_by_what_the_server_answered(
    refusing: str, path: str, failure: CheckFailure, http_status: int | None
) -> None:
    started = time.monotonic()

    check = await check_stored_server(_server(f"http://{refusing}/{path}"))

    assert (check.failure, check.http_status) == (failure, http_status)
    assert check.tool_names == []
    assert time.monotonic() - started < mcp_stateless.CONNECT_TIMEOUT_S, "named at once, not at the deadline"


@pytest.mark.asyncio
async def test_a_closed_port_is_unreachable() -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    check = await check_stored_server(_server(f"http://127.0.0.1:{port}/mcp"))

    assert check.failure == CheckFailure.UNREACHABLE


@pytest.mark.asyncio
async def test_a_failed_handshake_is_a_tls_failure(refusing: str) -> None:
    check = await check_stored_server(_server(f"https://{refusing}/mcp"))

    assert check.failure == CheckFailure.TLS_FAILED


@pytest.mark.asyncio
async def test_a_server_that_never_answers_times_out(refusing: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mcp_stateless, "CONNECT_TIMEOUT_S", 0.3)

    check = await check_stored_server(_server(f"http://{refusing}/hang"))

    assert check.failure == CheckFailure.TIMEOUT


@pytest.mark.asyncio
async def test_a_check_with_no_free_slot_raises_rather_than_reporting_the_server(
    refusing: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A busy gateway says nothing about the server, so it must not be recorded as the server's health."""
    gate = ConcurrencyGate(limit=1, admission_timeout_s=0.05)
    monkeypatch.setattr(mcp_stateless, "DISCOVERY_GATE", gate)

    async with gate.slot():
        with pytest.raises(McpCapacityUnavailable):
            await check_stored_server(_server(f"http://{refusing}/401"))


@pytest.mark.asyncio
async def test_a_notification_acknowledged_without_a_body_is_not_refused() -> None:
    """Only a request is owed an MCP body; a 200 for a notification may carry none."""
    notification = httpx2.Request(
        "POST", "https://mcp.example.com/mcp", json={"jsonrpc": "2.0", "method": "notifications/initialized"}
    )
    request = httpx2.Request("POST", "https://mcp.example.com/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "x"})

    await refuse_unusable_answer(httpx2.Response(200, request=notification))
    with pytest.raises(Exception) as raised:
        await refuse_unusable_answer(httpx2.Response(200, request=request))
    assert classify_check_failure(raised.value) == (CheckFailure.NOT_MCP, None)


@pytest.mark.asyncio
async def test_the_event_stream_get_may_be_refused() -> None:
    """The SDK opens a GET stream after initializing and treats a 405 there as normal."""
    await refuse_unusable_answer(httpx2.Response(405, request=httpx2.Request("GET", "https://mcp.example.com/mcp")))


def test_a_protocol_error_inside_a_group_is_not_mcp() -> None:
    group = ExceptionGroup("tg", [ExceptionGroup("inner", [MCPError(-32600, "x")])])

    assert classify_check_failure(group) == (CheckFailure.NOT_MCP, None)


def test_an_unrecognized_failure_is_named_generically() -> None:
    assert classify_check_failure(RuntimeError("boom")) == (CheckFailure.CONNECTION_FAILED, None)
    assert classify_check_failure(asyncio.CancelledError()) == (CheckFailure.CONNECTION_FAILED, None)
