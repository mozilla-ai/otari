"""What real request traffic records on a stored MCP server, and how often."""

from __future__ import annotations

import asyncio
import uuid
from contextlib import AsyncExitStack
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx2
import pytest
from mcp.types import ListToolsResult, ToolAnnotations
from mcp.types import Tool as MCPTool

from gateway.exceptions.tools_exceptions import McpSignInRequiredError
from gateway.models.mcp import AttachedTimeBudget, McpServerConfig
from gateway.models.tools import WorkspaceMcpServer
from gateway.services.mcp_client import MCPClientPool
from gateway.services.mcp_stateless import CheckFailure, ServerCheck
from gateway.services.model_access import mcp_read_tools
from gateway.services.tools import McpServerContact, McpServerContacts, WorkspaceMcpServerHealth
from gateway.services.tools._mcp_server_health import (
    _MAX_PENDING_WRITES,
    ATTACHED_COOLDOWN_S,
    CONTACT_WRITE_INTERVAL_S,
    MAX_ALLOWED_TOOLS,
    health_after_contact,
    server_health,
)

SERVER = uuid.uuid4()
URL = "https://93.184.216.34/mcp"
LISTING = ServerCheck(tool_names=["list_issues", "create_issue"], read_only_hinted=["list_issues"])


def _stored(status: str = "healthy", *, age_s: float = 0, **fields: Any) -> dict[str, Any]:
    checked_at = (datetime.now(UTC) - timedelta(seconds=age_s)).isoformat()
    return {"status": status, "checked_at": checked_at, **fields}


def _after(stored: dict[str, Any] | None, contact: McpServerContact, **kwargs: Any) -> dict[str, Any] | None:
    return health_after_contact(stored, allowed_tools=None, per_user=False, contact=contact, **kwargs)


def test_a_listing_is_stored_as_a_check_would_store_it() -> None:
    health = _after(None, McpServerContact(SERVER, URL, listing=LISTING))

    assert health is not None
    assert health["status"] == "healthy"
    assert health["tools"] == ["list_issues", "create_issue"]
    assert health["read_only_hinted_tools"] == ["list_issues"]
    assert health["tool_count"] == 2


def test_a_contact_that_listed_nothing_keeps_the_tools_on_record() -> None:
    old = server_health(None, LISTING).model_dump(mode="json")
    old["checked_at"] = (datetime.now(UTC) - timedelta(days=8)).isoformat()

    health = _after(old, McpServerContact(SERVER, URL))

    assert health is not None
    assert health["tools"] == old["tools"]
    assert health["checked_at"] > old["checked_at"]


def test_a_first_contact_without_a_listing_says_healthy_and_claims_no_tool_count() -> None:
    health = _after(_stored("failing", error="timeout"), McpServerContact(SERVER, URL))

    assert health is not None
    assert health["status"] == "healthy"
    assert health["tool_count"] is None
    assert health["tools"] == []


def test_the_same_answer_is_not_written_again_within_the_interval() -> None:
    fresh = server_health(None, LISTING).model_dump(mode="json")

    assert _after(fresh, McpServerContact(SERVER, URL)) is None
    assert _after(fresh, McpServerContact(SERVER, URL, listing=LISTING)) is None
    failing = _stored("failing", error="unreachable")
    assert _after(failing, McpServerContact(SERVER, URL, failure=CheckFailure.UNREACHABLE)) is None


def test_a_change_is_written_at_once() -> None:
    fresh = server_health(None, LISTING).model_dump(mode="json")
    more = ServerCheck(tool_names=["list_issues", "create_issue", "delete_issue"])

    failed = _after(fresh, McpServerContact(SERVER, URL, failure=CheckFailure.UNREACHABLE))
    grown = _after(fresh, McpServerContact(SERVER, URL, listing=more))
    recovered = _after(_stored("failing", error="unreachable"), McpServerContact(SERVER, URL, listing=LISTING))

    assert failed is not None and failed["status"] == "failing" and failed["error"] == "unreachable"
    assert grown is not None and grown["tool_count"] == 3
    assert recovered is not None and recovered["status"] == "healthy"


def test_a_failed_contact_keeps_the_read_hints_on_record() -> None:
    fresh = server_health(None, LISTING).model_dump(mode="json")

    failed = _after(fresh, McpServerContact(SERVER, URL, failure=CheckFailure.UNREACHABLE))
    assert failed is not None and failed["status"] == "failing"
    assert failed["tools"] == [] and failed["tool_count"] is None
    assert failed["read_only_hinted_tools"] == ["list_issues"]

    answered = _after(failed, McpServerContact(SERVER, URL))
    assert answered is not None and answered["status"] == "healthy"
    assert answered["read_only_hinted_tools"] == ["list_issues"]


def test_an_old_answer_is_refreshed() -> None:
    stale = server_health(None, LISTING).model_dump(mode="json")
    stale["checked_at"] = (datetime.now(UTC) - timedelta(seconds=CONTACT_WRITE_INTERVAL_S + 1)).isoformat()

    assert _after(stale, McpServerContact(SERVER, URL, listing=LISTING)) is not None


@pytest.mark.parametrize("failure", [CheckFailure.UNAUTHORIZED, CheckFailure.FORBIDDEN])
def test_one_user_s_rejected_token_is_not_a_server_failure(failure: CheckFailure) -> None:
    contact = McpServerContact(SERVER, URL, failure=failure, http_status=401)

    personal = health_after_contact(None, allowed_tools=None, per_user=True, contact=contact)
    shared = health_after_contact(None, allowed_tools=None, per_user=False, contact=contact)

    assert personal is None
    assert shared is not None and shared["status"] == "failing"


def test_a_listing_is_read_against_the_allow_list() -> None:
    health = health_after_contact(
        None,
        allowed_tools=["list_issues", "gone"],
        per_user=False,
        contact=McpServerContact(SERVER, URL, listing=LISTING),
    )

    assert health is not None
    assert health["tool_count"] == 1
    assert health["missing_allowed_tools"] == ["gone"]


def test_an_allow_list_edit_recounts_a_complete_listing() -> None:
    health = server_health(None, LISTING).for_allow_list(["list_issues", "gone"])

    assert (health.tool_count, health.missing_allowed_tools) == (1, ["gone"])
    assert health.tools == ["list_issues", "create_issue"]
    assert health.read_only_hinted_tools == ["list_issues"]


def test_an_allow_list_edit_does_not_call_a_tool_past_the_stored_preview_missing() -> None:
    names = [f"tool_{i}" for i in range(MAX_ALLOWED_TOOLS + 1)]
    stored = server_health(None, ServerCheck(tool_names=names, read_only_hinted=["tool_0", names[-1]]))
    assert len(stored.tools) == MAX_ALLOWED_TOOLS

    health = stored.for_allow_list([names[-1]])

    assert (health.tool_count, health.missing_allowed_tools) == (None, [])
    assert health.read_only_hinted_tools == ["tool_0", names[-1]]
    assert mcp_read_tools(None, health.read_only_hinted_tools) == ["tool_0", names[-1]]


class _Recorder(McpServerContacts):
    """Keeps what would be written, so the throttle is read without a database."""

    def __init__(self, *, per_user: bool = False, gate: asyncio.Event | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.written: list[McpServerContact] = []
        self.per_user = per_user
        self.gate = gate

    async def _store(self, contact: McpServerContact) -> Any:
        if self.gate is not None:
            await self.gate.wait()
        if self.per_user and contact.failure in (CheckFailure.UNAUTHORIZED, CheckFailure.FORBIDDEN):
            return "discarded"
        self.written.append(contact)
        return "decided"


@pytest.mark.asyncio
async def test_one_server_is_handed_to_a_write_once_per_interval_unless_its_status_changes() -> None:
    now = [0.0]
    recorder = _Recorder(interval_s=900, clock=lambda: now[0])

    recorder.reached(SERVER, URL, LISTING)
    recorder.reached(str(SERVER), URL)
    recorder.reached(SERVER, URL, LISTING)
    await recorder.drain()
    assert len(recorder.written) == 1

    recorder.failed(SERVER, URL, httpx2.ConnectError("refused"))
    recorder.failed(SERVER, URL, httpx2.ConnectError("refused"))
    await recorder.drain()
    assert [contact.failure for contact in recorder.written] == [None, CheckFailure.UNREACHABLE]

    now[0] = 901.0
    recorder.failed(SERVER, URL, httpx2.ConnectError("refused"))
    await recorder.drain()
    assert len(recorder.written) == 3


def _refused(status: int) -> httpx2.HTTPStatusError:
    request = httpx2.Request("POST", URL)
    return httpx2.HTTPStatusError("refused", request=request, response=httpx2.Response(status, request=request))


@pytest.mark.asyncio
async def test_a_listing_after_a_call_that_listed_nothing_is_written() -> None:
    """A tool call records success with no catalog; the listing after it still stores the tools."""
    recorder = _Recorder(interval_s=900, clock=lambda: 0.0)

    recorder.reached(SERVER, URL)
    recorder.reached(SERVER, URL, LISTING)
    recorder.reached(SERVER, URL, LISTING)
    await recorder.drain()

    assert [contact.listing for contact in recorder.written] == [None, LISTING]


@pytest.mark.asyncio
async def test_one_user_s_refusal_on_a_per_user_server_does_not_hold_back_a_real_failure() -> None:
    recorder = _Recorder(per_user=True, interval_s=900, clock=lambda: 0.0)

    recorder.reached(SERVER, URL)
    await recorder.drain()
    recorder.failed(SERVER, URL, _refused(401))
    await recorder.drain()
    recorder.failed(SERVER, URL, _refused(401))
    recorder.failed(SERVER, URL, httpx2.ConnectError("refused"))
    await recorder.drain()

    assert [contact.failure for contact in recorder.written] == [None, CheckFailure.UNREACHABLE]


@pytest.mark.asyncio
async def test_a_shared_account_s_refusal_keeps_the_same_status_throttle() -> None:
    recorder = _Recorder(interval_s=900, clock=lambda: 0.0)

    recorder.failed(SERVER, URL, _refused(401))
    await recorder.drain()
    recorder.failed(SERVER, URL, httpx2.ConnectError("refused"))
    await recorder.drain()

    assert [contact.failure for contact in recorder.written] == [CheckFailure.UNAUTHORIZED]


@pytest.mark.asyncio
async def test_a_failure_arriving_while_one_user_s_refusal_is_being_written_is_recorded() -> None:
    """The refusal's write is held open while the timeout arrives; only the write decides what counts."""
    gate = asyncio.Event()
    recorder = _Recorder(per_user=True, gate=gate, interval_s=900, clock=lambda: 0.0)

    recorder.failed(SERVER, URL, _refused(401))
    await asyncio.sleep(0)
    recorder.failed(SERVER, URL, TimeoutError())
    gate.set()
    await recorder.drain()
    recorder.failed(SERVER, URL, _refused(401))
    await recorder.drain()

    assert [contact.failure for contact in recorder.written] == [CheckFailure.TIMEOUT]


@pytest.mark.asyncio
async def test_an_identical_contact_in_flight_is_not_handed_to_a_second_write() -> None:
    gate = asyncio.Event()
    recorder = _Recorder(gate=gate, interval_s=900, clock=lambda: 0.0)

    recorder.failed(SERVER, URL, TimeoutError())
    recorder.failed(SERVER, URL, TimeoutError())
    gate.set()
    await recorder.drain()

    assert [contact.failure for contact in recorder.written] == [CheckFailure.TIMEOUT]


@pytest.mark.asyncio
async def test_a_new_url_is_not_held_back_by_the_old_one_s_throttle() -> None:
    recorder = _Recorder(interval_s=900, clock=lambda: 0.0)
    moved = "https://93.184.216.35/mcp"

    recorder.reached(SERVER, URL, LISTING)
    await recorder.drain()
    recorder.reached(SERVER, moved, LISTING)
    await recorder.drain()

    assert [contact.url for contact in recorder.written] == [URL, moved]


@pytest.mark.asyncio
async def test_a_contact_the_write_found_stale_does_not_count() -> None:
    class Stale(_Recorder):
        async def _store(self, contact: McpServerContact) -> Any:
            if not self.written and contact.listing is None:
                self.written.append(contact)
                return "stale"
            return await super()._store(contact)

    recorder = Stale(interval_s=900, clock=lambda: 0.0)
    recorder.reached(SERVER, URL)
    await recorder.drain()
    recorder.reached(SERVER, URL)
    await recorder.drain()

    assert len(recorder.written) == 2


@pytest.mark.asyncio
async def test_writes_in_flight_are_capped_and_a_contact_past_the_cap_is_skipped() -> None:
    gate = asyncio.Event()
    recorder = _Recorder(gate=gate)

    for _ in range(_MAX_PENDING_WRITES + 5):
        recorder.reached(uuid.uuid4(), URL)
    assert len(recorder._tasks) == _MAX_PENDING_WRITES
    gate.set()
    await recorder.drain()
    recorder.reached(uuid.uuid4(), URL)
    await recorder.drain()

    assert len(recorder.written) == _MAX_PENDING_WRITES + 1


@pytest.mark.asyncio
async def test_a_missing_sign_in_and_an_inline_server_record_nothing() -> None:
    recorder = _Recorder()

    recorder.failed(SERVER, URL, McpSignInRequiredError(SERVER, name="github", user_id="user-1"))
    recorder.reached(None, URL, LISTING)
    recorder.reached("not-a-uuid", URL)
    await recorder.drain()

    assert recorder.written == []


@pytest.mark.asyncio
async def test_a_write_that_fails_never_reaches_the_request() -> None:
    class Broken(McpServerContacts):
        async def _store(self, contact: McpServerContact) -> Any:
            raise RuntimeError("no database")

    recorder = Broken()
    recorder.reached(SERVER, URL)
    await asyncio.sleep(0)
    await recorder.drain()


def test_unset_read_labels_are_the_hints_of_the_first_listing() -> None:
    hinted = ["list_issues", "*", " @read", "a,b", "list_issues"]

    assert mcp_read_tools(None, hinted) == ["list_issues"]
    assert mcp_read_tools(None, None) == []
    assert mcp_read_tools([], hinted) == []
    assert mcp_read_tools(["get_issue"], hinted) == ["get_issue"]


def test_only_the_first_listing_s_hints_are_kept() -> None:
    server = WorkspaceMcpServer(workspace_id=uuid.uuid4(), name="s", url=URL)
    server.health = WorkspaceMcpServerHealth(status="healthy", checked_at="2026-01-01T00:00:00+00:00").model_dump(
        mode="json"
    )
    WorkspaceMcpServerHealth.keep_first_read_hints(server)
    assert server.read_tools_hinted is None, "an answer that listed nothing is not a listing"

    server.health = server_health(None, LISTING).model_dump(mode="json")
    WorkspaceMcpServerHealth.keep_first_read_hints(server)
    assert server.read_tools_hinted == ["list_issues"]

    both = ["list_issues", "create_issue"]
    relabeled = ServerCheck(tool_names=both, read_only_hinted=both)
    server.health = server_health(None, relabeled).model_dump(mode="json")
    WorkspaceMcpServerHealth.keep_first_read_hints(server)
    assert server.read_tools_hinted == ["list_issues"]


class _Heard:
    """A contact observer that keeps what it was told."""

    def __init__(self) -> None:
        self.reached_with: list[tuple[uuid.UUID | str | None, ServerCheck | None]] = []
        self.failures: list[uuid.UUID | str | None] = []

    def reached(self, server_id: uuid.UUID | str | None, url: str, listing: ServerCheck | None = None) -> None:
        self.reached_with.append((server_id, listing))

    def failed(self, server_id: uuid.UUID | str | None, url: str, failure: BaseException) -> None:
        self.failures.append(server_id)


def _tool(name: str, *, read_only: bool = False) -> MCPTool:
    annotations = ToolAnnotations(read_only_hint=True) if read_only else None
    return MCPTool(name=name, input_schema={"type": "object"}, annotations=annotations)


@pytest.mark.asyncio
async def test_the_tool_loop_reports_each_stored_server_it_connects_to(monkeypatch: pytest.MonkeyPatch) -> None:
    listings = {
        "github": ListToolsResult(tools=[_tool("list_issues", read_only=True), _tool("create_issue")]),
        "paged": ListToolsResult(tools=[_tool("first")], next_cursor="more"),
        "inline": ListToolsResult(tools=[_tool("other")]),
    }

    async def fake_open(self: MCPClientPool, stack: AsyncExitStack, cfg: McpServerConfig, *_: object) -> Any:
        if cfg.name == "down":
            raise httpx2.ConnectError("refused")
        return object(), listings[cfg.name]

    monkeypatch.setattr(MCPClientPool, "_open", fake_open)
    heard = _Heard()
    configs = [
        McpServerConfig(name="github", url=URL, stored_server_id="github-id"),
        McpServerConfig(name="paged", url=URL, stored_server_id="paged-id"),
        McpServerConfig(name="inline", url=URL),
        McpServerConfig(name="down", url=URL, stored_server_id="down-id", attached=True),
    ]

    async with MCPClientPool(configs, contacts=heard):
        pass

    [(first, whole), (second, partial)] = heard.reached_with
    assert (first, second) == ("github-id", "paged-id")
    assert whole is not None and whole.tool_names == ["list_issues", "create_issue"]
    assert whole.read_only_hinted == ["list_issues"]
    # One page of several is not the server's catalog, so it is not recorded as one.
    assert partial is None
    assert heard.failures == ["down-id"]


@pytest.mark.asyncio
async def test_the_tool_loop_does_not_blame_a_server_for_one_user_s_ended_sign_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_open(self: MCPClientPool, stack: AsyncExitStack, cfg: McpServerConfig, *_: object) -> Any:
        if cfg.name == "slow":
            raise TimeoutError
        raise McpSignInRequiredError(SERVER, name="drive", user_id="user-1")

    monkeypatch.setattr(MCPClientPool, "_open", fake_open)
    recorder = _Recorder()
    configs = [
        McpServerConfig(name="drive", url=URL, stored_server_id=str(SERVER)),
        # An attached server that ran out of what the others left of the budget says nothing about itself.
        McpServerConfig(
            name="slow",
            url=URL,
            stored_server_id=str(uuid.uuid4()),
            attached=True,
            attached_budget=AttachedTimeBudget(1.0),
        ),
    ]

    async with MCPClientPool(configs, contacts=recorder):
        pass
    await recorder.drain()

    assert recorder.written == []


@pytest.mark.asyncio
async def test_an_attached_server_that_times_out_on_its_own_deadline_is_recorded_as_failing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_open(self: MCPClientPool, stack: AsyncExitStack, cfg: McpServerConfig, *_: object) -> Any:
        raise TimeoutError

    monkeypatch.setattr(MCPClientPool, "_open", fake_open)
    recorder = _Recorder()
    attached = McpServerConfig(name="slow", url=URL, stored_server_id=str(SERVER), attached=True)

    async with MCPClientPool([attached], contacts=recorder):
        pass
    await recorder.drain()

    [contact] = recorder.written
    assert (contact.server_id, contact.failure) == (SERVER, CheckFailure.TIMEOUT)
    assert WorkspaceMcpServerHealth.cooling_down(_after(None, contact)), "later requests leave it out"


def _failing(error: CheckFailure, seconds_ago: float) -> dict[str, Any]:
    checked_at = (datetime.now(UTC) - timedelta(seconds=seconds_ago)).isoformat()
    failed = server_health(None, ServerCheck(tool_names=[], failure=error))
    return failed.model_copy(update={"checked_at": checked_at}).model_dump(mode="json")


def test_an_unreachable_server_cools_down_for_a_while_and_a_refusal_does_not() -> None:
    assert WorkspaceMcpServerHealth.cooling_down(_failing(CheckFailure.TIMEOUT, 5))
    assert WorkspaceMcpServerHealth.cooling_down(_failing(CheckFailure.UNREACHABLE, 5))
    assert not WorkspaceMcpServerHealth.cooling_down(_failing(CheckFailure.TIMEOUT, ATTACHED_COOLDOWN_S + 1))
    assert not WorkspaceMcpServerHealth.cooling_down(_failing(CheckFailure.UNAUTHORIZED, 5))
    assert not WorkspaceMcpServerHealth.cooling_down(server_health(None, LISTING).model_dump(mode="json"))
    assert not WorkspaceMcpServerHealth.cooling_down(None)


def test_a_server_that_keeps_failing_is_written_again_once_its_cooldown_is_over() -> None:
    timed_out = McpServerContact(SERVER, URL, failure=CheckFailure.TIMEOUT)

    assert _after(_failing(CheckFailure.TIMEOUT, 5), timed_out) is None
    again = _after(_failing(CheckFailure.TIMEOUT, ATTACHED_COOLDOWN_S + 1), timed_out)
    assert again is not None and WorkspaceMcpServerHealth.cooling_down(again)
