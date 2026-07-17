"""Unit tests for compute-node resolution (composite_compute) and its integration
with dispatch. A local fake sandbox runs the wrapped code so no live otari-exec is
needed; it executes the exact code the real sandbox would."""

from __future__ import annotations

import contextlib
import io
from typing import Any

import pytest

from gateway.services.composite_compute import resolve_bindings, visible_tool_results
from gateway.services.composite_dispatch import Serve, decide

_LIST = (
    "# Email Search Results\nFound 2 emails.\n\n"
    "- [msg_id:19f65405572bff12] Subject: a\n"
    "- [msg_id:19f653757c83bd78] Subject: b\n"
)
_IDS = ["19f65405572bff12", "19f653757c83bd78"]


class _FakeSandbox:
    """Runs the wrapped code locally and returns stdout, like otari-exec would."""

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            exec(arguments["code"], {})  # noqa: S102 - mirrors the sandbox contract
        return buf.getvalue()


class _BrokenSandbox:
    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        raise RuntimeError("sandbox unreachable")


def _messages() -> list[dict[str, Any]]:
    return [
        {"role": "user", "content": "organize inbox"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "gmail_list_emails", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": _LIST}]},
    ]


def _plan() -> dict[str, Any]:
    return {
        "version": 1,
        "nodes": [
            {"type": "emit_tool_use", "tool": "gmail_list_emails", "args": {"query": {"const": "is:unread"}}},
            {"type": "compute", "bind": "ids",
             "code": "import re\noutput = re.findall(r'\\[msg_id:([0-9a-f]+)\\]', results[-1])"},
            {"type": "map", "over": {"var": "ids"},
             "body": [{"type": "emit_tool_use", "tool": "gmail_read_email", "args": {"message_id": {"item": ""}}}]},
        ],
    }


def test_visible_tool_results_orders_result_strings() -> None:
    assert visible_tool_results(_messages()) == [_LIST]


@pytest.mark.asyncio
async def test_resolve_bindings_runs_compute_code() -> None:
    bindings = await resolve_bindings(_plan(), _messages(), _FakeSandbox())
    assert bindings == {"ids": _IDS}


@pytest.mark.asyncio
async def test_resolve_bindings_omits_failed_node_never_raises() -> None:
    # Sandbox unreachable -> binding omitted, no exception.
    assert await resolve_bindings(_plan(), _messages(), _BrokenSandbox()) == {}


@pytest.mark.asyncio
async def test_no_compute_nodes_is_zero_io() -> None:
    plan = {"version": 1, "nodes": [{"type": "emit_tool_use", "tool": "x", "args": {}}]}
    # A runner that would explode if called; proves resolve_bindings does no I/O here.
    assert await resolve_bindings(plan, _messages(), _BrokenSandbox()) == {}


@pytest.mark.asyncio
async def test_decide_with_resolved_bindings_serves_the_loop_turn() -> None:
    composite = {
        "automation_key": "automation:e7",
        "status": "approved",
        "plan": _plan(),
        "verifier_spec": {},
        "recognize_envelope": {},
    }
    bindings = await resolve_bindings(_plan(), _messages(), _FakeSandbox())
    d = decide([composite], _messages(), session_label="automation:e7", bindings=bindings)
    assert isinstance(d, Serve)
    assert d.action.tool_name == "gmail_read_email"
    assert d.action.tool_input == {"message_id": _IDS[0]}


def test_decide_without_bindings_punts_not_serves() -> None:
    # Same plan, but no compute resolved -> the map's `over` var fails -> NoDispatch.
    composite = {
        "automation_key": "automation:e7", "status": "approved",
        "plan": _plan(), "verifier_spec": {}, "recognize_envelope": {},
    }
    d = decide([composite], _messages(), session_label="automation:e7")
    assert not isinstance(d, Serve)
