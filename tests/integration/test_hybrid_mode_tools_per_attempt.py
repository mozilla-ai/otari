"""Who runs a provider's code execution, web search or web fetch on each attempt of a hybrid route.

The platform plans the attempts and Otari decides each one's tools, as a standalone
gateway does (``test_code_execution_per_attempt.py``, ``test_web_tools_per_attempt.py``):
an attempt whose model runs the keyword natively is sent it, and one that cannot runs
Otari's tool. What hybrid adds is the contract with the control plane: a route with one
attempt is dispatched as it always was, each policy is resolved once per request, a
usage report goes to the attempt that ran, and no attempt is skipped, so a request one
of whose attempts would be refused is decided as a whole.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Generator
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import Any, cast
from unittest.mock import patch

import httpx
import pytest
from any_llm.types.messages import (
    MessageDelta,
    MessageDeltaEvent,
    MessageDeltaUsage,
    MessageResponse,
    MessageStartEvent,
    MessageStopEvent,
    MessageStreamEvent,
    MessageUsage,
    TextBlock,
)
from fastapi.testclient import TestClient

from conftest import InstallControlPlane
from gateway.api.deps import reset_config
from gateway.core.config import API_ROOT, GatewayConfig
from gateway.core.database import reset_db
from gateway.services.sandbox_backend import CODE_EXECUTION_TOOL_NAME, SandboxNotReachableError
from gateway.services.tools import WEB_FETCH_HEADER, WEB_FETCH_TOOL_NAME, WEB_SEARCH_HEADER, WEB_SEARCH_TOOL_NAME

from .conftest import app_for

_CLAUDE = "claude-haiku-4-5"
_HEAD = "3f1b6a1e-0000-4000-8000-0000000000a1"
_FALLBACK = "3f1b6a1e-0000-4000-8000-0000000000a2"
_DECLARATIONS = {
    "code": {"type": "code_execution_20250825", "name": "code_execution"},
    "search": {"type": "web_search_20250305", "name": "web_search"},
    "fetch": {"type": "web_fetch_20250910", "name": "web_fetch"},
}
_HEADERS = {"code": "Otari-Code-Execution", "search": WEB_SEARCH_HEADER, "fetch": WEB_FETCH_HEADER}
_NAMES = {"code": CODE_EXECUTION_TOOL_NAME, "search": WEB_SEARCH_TOOL_NAME, "fetch": WEB_FETCH_TOOL_NAME}
_POLICY_PATHS = {
    "code": "/gateway/code-execution/resolve",
    "search": "/gateway/web-search/resolve",
    "fetch": "/gateway/web-search/resolve",
}
_TOOLS = ["code", "search", "fetch"]


@pytest.fixture
def platform_client(monkeypatch: pytest.MonkeyPatch) -> Generator[TestClient]:
    monkeypatch.setenv("OTARI_AI_TOKEN", "gw_test_token")
    monkeypatch.setenv("OTARI_SANDBOX_URL", "http://sandbox:8080")
    monkeypatch.setenv("OTARI_WEB_SEARCH_URL", "http://searxng:8080")
    monkeypatch.delenv("OTARI_WEB_SEARCH_INTERCEPT", raising=False)
    app = app_for(
        GatewayConfig(
            mode="hybrid",
            provider_user_identifier_enabled=False,
            web_fetch_enabled=True,
            platform={"base_url": "http://platform.test/api/v1"},
        )
    )
    with TestClient(app) as client:
        yield client
    reset_config()
    reset_db()


def _attempt(position: int, attempt_id: str, provider: str) -> dict[str, Any]:
    return {
        "attempt_id": attempt_id,
        "position": position,
        "provider": provider,
        "model": _CLAUDE,
        "api_key": f"sk-{provider}",
        "api_base": None,
        "managed": False,
    }


def _reply(text: str) -> MessageResponse:
    return MessageResponse(
        id="msg_platform",
        type="message",
        role="assistant",
        model=_CLAUDE,
        content=[TextBlock(type="text", text=text, citations=None)],
        stop_reason=cast(Any, "end_turn"),
        stop_sequence=None,
        usage=MessageUsage(input_tokens=5, output_tokens=2),
    )


async def _events(text: str) -> AsyncIterator[MessageStreamEvent]:
    yield MessageStartEvent(type="message_start", message=cast(Any, _reply(text)))
    yield MessageDeltaEvent(
        type="message_delta",
        delta=MessageDelta(stop_reason=cast(Any, "end_turn"), stop_sequence=None),
        usage=MessageDeltaUsage(
            input_tokens=None,
            output_tokens=1,
            cache_creation_input_tokens=None,
            cache_read_input_tokens=None,
            server_tool_use=None,
        ),
    )
    yield MessageStopEvent(type="message_stop")


def _unavailable() -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return httpx.HTTPStatusError("503", request=request, response=httpx.Response(503, request=request))


@dataclass
class _Seen:
    """What one request asked of the control plane, the providers, the loops and the backends."""

    resolves: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    usage_reports: list[dict[str, Any]] = field(default_factory=list)
    provider_calls: list[dict[str, Any]] = field(default_factory=list)
    loop_calls: list[tuple[dict[str, Any], dict[str, Any]]] = field(default_factory=list)
    backends: list[str] = field(default_factory=list)

    def resolved(self, path: str) -> list[dict[str, Any]]:
        return [body for url, body in self.resolves if url.endswith(path)]

    @staticmethod
    def tool_types(kwargs: dict[str, Any]) -> set[str]:
        return {tool["type"] for tool in kwargs.get("tools") or []}


class _Backend:
    """A tool backend that is never called, since the tool loop is faked."""

    def __init__(self, seen: _Seen, kind: str) -> None:
        seen.backends.append(kind)

    async def __aenter__(self) -> _Backend:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    def purpose_hints(self) -> list[tuple[str, str]]:
        return []


def _post(
    client: TestClient,
    control_plane_transport: InstallControlPlane,
    *,
    providers: list[str],
    tool: str,
    header: str | None = "auto",
    stream: bool = False,
    head_fails: bool = False,
    policy: dict[str, Any] | None = None,
    also: tuple[str, ...] = (),
) -> tuple[Any, str, _Seen]:
    """POST a Messages request declaring ``tool`` in Anthropic's words, over a route to ``providers``.

    ``also`` declares more tools beside it, each under ``auto``.
    Returns the response, its body as read, and what the request did.
    """
    seen = _Seen()
    attempts = [
        _attempt(position, attempt_id, provider)
        for position, (attempt_id, provider) in enumerate(zip((_HEAD, _FALLBACK), providers, strict=False))
    ]

    async def fake_post_platform(
        url: str, headers: dict[str, str], body: dict[str, Any], timeout_seconds: float
    ) -> httpx.Response:
        if url.endswith("/gateway/provider-keys/resolve"):
            return httpx.Response(200, json={"request_id": "req-1", "fallback_enabled": True, "attempts": attempts})
        if url.endswith("/gateway/usage"):
            seen.usage_reports.append(body)
            return httpx.Response(202)
        seen.resolves.append((url, body))
        if url.endswith("/gateway/code-execution/resolve"):
            return httpx.Response(200, json=policy if tool == "code" and policy else {"enabled": True})
        if url.endswith("/gateway/web-search/resolve"):
            default = {"enabled": True, "authorized_tools": ["web_search", "web_fetch"]}
            return httpx.Response(200, json=policy if tool != "code" and policy else default)
        return httpx.Response(204)

    async def fake_amessages(**kwargs: Any) -> Any:
        seen.provider_calls.append(kwargs)
        if head_fails and len(seen.provider_calls) == 1:
            raise _unavailable()
        return _events("via-provider") if kwargs.get("stream") else _reply("via-provider")

    async def fake_loop(*, completion_kwargs: Any, pool: Any, max_iterations: int, **extra: Any) -> MessageResponse:
        seen.loop_calls.append((completion_kwargs, extra))
        return _reply("via-otari")

    def fake_loop_stream(
        *, completion_kwargs: Any, pool: Any, max_iterations: int, **extra: Any
    ) -> AsyncIterator[MessageStreamEvent]:
        seen.loop_calls.append((completion_kwargs, extra))
        return _events("via-otari")

    control_plane_transport(fake_post_platform)
    headers = {"Authorization": "Bearer user_test_token"}
    if header is not None:
        headers[_HEADERS[tool]] = header
    headers |= {_HEADERS[other]: "auto" for other in also}
    body = {
        "model": _CLAUDE,
        "messages": [{"role": "user", "content": "go"}],
        "max_tokens": 100,
        "tools": [dict(_DECLARATIONS[name]) for name in (tool, *also)],
        **({"stream": True} if stream else {}),
    }
    with ExitStack() as stack:
        stack.enter_context(patch("gateway.api.routes.messages.amessages", new=fake_amessages))
        stack.enter_context(patch("gateway.api.routes.messages.anthropic_tool_loop", new=fake_loop))
        stack.enter_context(patch("gateway.api.routes.messages.anthropic_tool_loop_stream", new=fake_loop_stream))
        stack.enter_context(
            patch("gateway.api.routes._pipeline.SandboxBackend", new=lambda **_kwargs: _Backend(seen, "sandbox"))
        )
        stack.enter_context(
            patch(
                "gateway.api.routes._pipeline._build_web_retrieval_backend",
                new=lambda **_kwargs: _Backend(seen, "web"),
            )
        )
        with client.stream("POST", f"{API_ROOT}/messages", json=body, headers=headers) as response:
            text = response.read().decode()
    return response, text, seen


def _report(attempt_id: str, *, status: str = "success", error_class: str | None = None, final: bool = True) -> Any:
    """Matches a usage report for ``attempt_id`` on the fields this module pins."""
    return {"correlation_id": attempt_id, "status": status, "is_final_attempt": final} | (
        {"error_class": error_class} if error_class is not None else {}
    )


def _pinned(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = ("correlation_id", "status", "is_final_attempt", "error_class")
    return sorted(
        ({key: report[key] for key in keys if key in report} for report in reports),
        key=lambda report: (report["correlation_id"], report["status"]),
    )


# --- a route with one attempt ------------------------------------------------------------------


@pytest.mark.parametrize("tool", _TOOLS)
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(("provider", "runs_here"), [("anthropic", False), ("vertexai", True)])
def test_a_single_attempt_route_is_dispatched_as_before(
    platform_client: TestClient,
    control_plane_transport: InstallControlPlane,
    tool: str,
    stream: bool,
    provider: str,
    runs_here: bool,
) -> None:
    response, text, seen = _post(
        platform_client, control_plane_transport, providers=[provider], tool=tool, stream=stream
    )

    assert response.status_code == 200, text
    assert response.headers["Otari-Attempt-ID"] == _HEAD
    if runs_here:
        assert seen.provider_calls == []
        [(loop_kwargs, loop_extra)] = seen.loop_calls
        assert seen.tool_types(loop_kwargs) == set()
        assert loop_extra.get("native_tools") == frozenset({_NAMES[tool]})
        assert seen.backends == ["sandbox" if tool == "code" else "web"]
    else:
        assert [seen.tool_types(call) for call in seen.provider_calls] == [{_DECLARATIONS[tool]["type"]}]
        assert seen.loop_calls == []
        assert seen.backends == []
    # Code execution's policy is asked whenever the sandbox could run the code, the web policy only where it does.
    assert len(seen.resolved(_POLICY_PATHS[tool])) == (1 if tool == "code" or runs_here else 0)
    assert _pinned(seen.usage_reports) == [_report(_HEAD)]


@pytest.mark.parametrize("tool", ["search", "fetch"])
def test_a_single_attempt_asks_the_web_policy_for_the_tool_it_runs(
    platform_client: TestClient, control_plane_transport: InstallControlPlane, tool: str
) -> None:
    _, _, seen = _post(platform_client, control_plane_transport, providers=["vertexai"], tool=tool)

    assert seen.resolved("/gateway/web-search/resolve") == [{"requested_tools": [_NAMES[tool]]}]


# --- a route with a fallback ---------------------------------------------------------------------


@pytest.mark.parametrize("tool", _TOOLS)
@pytest.mark.parametrize("stream", [False, True])
def test_a_native_head_that_answers_opens_no_backend(
    platform_client: TestClient, control_plane_transport: InstallControlPlane, tool: str, stream: bool
) -> None:
    response, text, seen = _post(
        platform_client, control_plane_transport, providers=["anthropic", "vertexai"], tool=tool, stream=stream
    )

    assert response.status_code == 200, text
    assert "via-provider" in text
    assert [seen.tool_types(call) for call in seen.provider_calls] == [{_DECLARATIONS[tool]["type"]}]
    assert seen.loop_calls == []
    assert seen.backends == [], "a backend was opened for a request the provider answered"
    assert response.headers["Otari-Attempt-ID"] == _HEAD
    assert _pinned(seen.usage_reports) == [_report(_HEAD)]
    # Asked once, for the tool the fallback would have run here, as when every attempt ran it here.
    assert [len(seen.resolved(_POLICY_PATHS[tool]))] == [1]


@pytest.mark.parametrize("tool", _TOOLS)
@pytest.mark.parametrize("stream", [False, True])
def test_a_head_that_fails_before_output_hands_the_fallback_otaris_tool(
    platform_client: TestClient, control_plane_transport: InstallControlPlane, tool: str, stream: bool
) -> None:
    response, text, seen = _post(
        platform_client,
        control_plane_transport,
        providers=["anthropic", "vertexai"],
        tool=tool,
        stream=stream,
        head_fails=True,
    )

    assert response.status_code == 200, text
    assert "via-otari" in text
    # The head was sent Anthropic's own tool, and only the fallback ran Otari's.
    assert [seen.tool_types(call) for call in seen.provider_calls] == [{_DECLARATIONS[tool]["type"]}]
    [(loop_kwargs, loop_extra)] = seen.loop_calls
    assert seen.tool_types(loop_kwargs) == set()
    assert loop_extra.get("native_tools") == frozenset({_NAMES[tool]})
    assert seen.backends == ["sandbox" if tool == "code" else "web"]
    assert response.headers["Otari-Attempt-ID"] == _FALLBACK
    assert _pinned(seen.usage_reports) == [
        _report(_HEAD, status="error", error_class="http_503", final=False),
        _report(_FALLBACK),
    ]
    assert len(seen.resolved(_POLICY_PATHS[tool])) == 1


@pytest.mark.parametrize("stream", [False, True])
def test_a_fallback_runs_code_and_search_in_one_loop_where_its_head_ran_both_itself(
    platform_client: TestClient, control_plane_transport: InstallControlPlane, stream: bool
) -> None:
    """The fallback's answer combines two of Otari's tools, which is admitted for that attempt alone."""
    response, text, seen = _post(
        platform_client,
        control_plane_transport,
        providers=["anthropic", "vertexai"],
        tool="code",
        also=("search",),
        stream=stream,
        head_fails=True,
    )

    assert response.status_code == 200, text
    assert "via-otari" in text
    assert [seen.tool_types(call) for call in seen.provider_calls] == [
        {_DECLARATIONS["code"]["type"], _DECLARATIONS["search"]["type"]}
    ]
    [(loop_kwargs, loop_extra)] = seen.loop_calls
    assert seen.tool_types(loop_kwargs) == set()
    assert loop_extra.get("native_tools") == frozenset({CODE_EXECUTION_TOOL_NAME, WEB_SEARCH_TOOL_NAME})
    assert sorted(seen.backends) == ["sandbox", "web"]
    assert response.headers["Otari-Attempt-ID"] == _FALLBACK
    assert _pinned(seen.usage_reports) == [
        _report(_HEAD, status="error", error_class="http_503", final=False),
        _report(_FALLBACK),
    ]
    assert len(seen.resolved(_POLICY_PATHS["code"])) == 1
    assert seen.resolved(_POLICY_PATHS["search"]) == [{"requested_tools": [WEB_SEARCH_TOOL_NAME]}]


@pytest.mark.parametrize("stream", [False, True])
def test_a_web_policy_read_for_two_tools_is_read_once(
    platform_client: TestClient, control_plane_transport: InstallControlPlane, stream: bool
) -> None:
    """Search runs here on every attempt and fetch only on the fallback, which one read covers."""
    seen = _Seen()

    async def fake_post_platform(
        url: str, headers: dict[str, str], body: dict[str, Any], timeout_seconds: float
    ) -> httpx.Response:
        if url.endswith("/gateway/provider-keys/resolve"):
            attempts = [_attempt(0, _HEAD, "anthropic"), _attempt(1, _FALLBACK, "vertexai")]
            return httpx.Response(200, json={"request_id": "req-1", "fallback_enabled": True, "attempts": attempts})
        if url.endswith("/gateway/usage"):
            seen.usage_reports.append(body)
            return httpx.Response(202)
        seen.resolves.append((url, body))
        return httpx.Response(200, json={"enabled": True, "authorized_tools": ["web_search", "web_fetch"]})

    async def fake_loop(*, completion_kwargs: Any, pool: Any, max_iterations: int, **extra: Any) -> MessageResponse:
        seen.loop_calls.append((completion_kwargs, extra))
        return _reply("via-otari")

    def fake_loop_stream(
        *, completion_kwargs: Any, pool: Any, max_iterations: int, **extra: Any
    ) -> AsyncIterator[MessageStreamEvent]:
        seen.loop_calls.append((completion_kwargs, extra))
        return _events("via-otari")

    control_plane_transport(fake_post_platform)
    with (
        patch("gateway.api.routes.messages.anthropic_tool_loop", new=fake_loop),
        patch("gateway.api.routes.messages.anthropic_tool_loop_stream", new=fake_loop_stream),
        patch("gateway.api.routes._pipeline._build_web_retrieval_backend", new=lambda **_kwargs: _Backend(seen, "web")),
    ):
        response = platform_client.post(
            f"{API_ROOT}/messages",
            json={
                "model": _CLAUDE,
                "messages": [{"role": "user", "content": "go"}],
                "max_tokens": 100,
                "tools": [_DECLARATIONS["search"], _DECLARATIONS["fetch"]],
                **({"stream": True} if stream else {}),
            },
            headers={"Authorization": "Bearer user_test_token", WEB_SEARCH_HEADER: "otari", WEB_FETCH_HEADER: "auto"},
        )

    assert response.status_code == 200, response.text
    assert seen.resolved("/gateway/web-search/resolve") == [{"requested_tools": ["web_search", "web_fetch"]}]
    [(loop_kwargs, loop_extra)] = seen.loop_calls
    # The head runs Otari's search and leaves the fetch with Anthropic.
    assert seen.tool_types(loop_kwargs) == {"web_fetch_20250910"}
    assert loop_extra.get("native_tools") == frozenset({WEB_SEARCH_TOOL_NAME})


@pytest.mark.parametrize(
    ("tool", "policy"),
    [
        ("search", {"enabled": False}),
        ("fetch", {"enabled": True, "authorized_tools": ["web_search"]}),
    ],
)
def test_a_fallback_that_would_be_refused_decides_the_request_as_a_whole(
    platform_client: TestClient,
    control_plane_transport: InstallControlPlane,
    tool: str,
    policy: dict[str, Any],
) -> None:
    """No attempt is skipped in hybrid, so a web tool is decided by every attempt, as it was before."""
    response, text, seen = _post(
        platform_client, control_plane_transport, providers=["anthropic", "vertexai"], tool=tool, policy=policy
    )

    assert response.status_code == 403, text
    assert seen.provider_calls == []
    assert seen.loop_calls == []
    assert len(seen.resolved(_POLICY_PATHS[tool])) == 1


def test_a_fallback_that_would_be_refused_leaves_the_code_to_a_head_that_runs_it(
    platform_client: TestClient, control_plane_transport: InstallControlPlane
) -> None:
    """Decided as a whole, the code follows the head, as it did before: one that runs it itself keeps it."""
    response, text, seen = _post(
        platform_client,
        control_plane_transport,
        providers=["anthropic", "vertexai"],
        tool="code",
        policy={"enabled": False},
    )

    assert response.status_code == 200, text
    assert [seen.tool_types(call) for call in seen.provider_calls] == [{_DECLARATIONS["code"]["type"]}]
    assert seen.loop_calls == []
    assert len(seen.resolved(_POLICY_PATHS["code"])) == 1


@pytest.mark.parametrize("tool", _TOOLS)
@pytest.mark.parametrize("stream", [False, True])
def test_provider_forwards_the_tool_to_every_attempt(
    platform_client: TestClient, control_plane_transport: InstallControlPlane, tool: str, stream: bool
) -> None:
    """As in standalone, ``provider`` sends the provider's own tool to every attempt and never runs Otari's."""
    response, text, seen = _post(
        platform_client,
        control_plane_transport,
        providers=["anthropic", "vertexai"],
        tool=tool,
        header="provider",
        stream=stream,
        head_fails=True,
    )

    assert response.status_code == 200, text
    assert [seen.tool_types(call) for call in seen.provider_calls] == [{_DECLARATIONS[tool]["type"]}] * 2
    assert seen.loop_calls == []
    assert _pinned(seen.usage_reports) == [
        _report(_HEAD, status="error", error_class="http_503", final=False),
        _report(_FALLBACK),
    ]


@pytest.mark.parametrize("stream", [False, True])
def test_a_fallbacks_unreachable_backend_ends_the_walk_on_that_attempt(
    platform_client: TestClient, control_plane_transport: InstallControlPlane, stream: bool
) -> None:
    """Opened only when the fallback is reached, its failure is that attempt's, reported as each walker reports one."""
    seen = _Seen()

    class _Unreachable(_Backend):
        async def __aenter__(self) -> _Backend:
            raise SandboxNotReachableError("down")

    async def fake_post_platform(
        url: str, headers: dict[str, str], body: dict[str, Any], timeout_seconds: float
    ) -> httpx.Response:
        if url.endswith("/gateway/provider-keys/resolve"):
            attempts = [_attempt(0, _HEAD, "anthropic"), _attempt(1, _FALLBACK, "vertexai")]
            return httpx.Response(200, json={"request_id": "req-1", "fallback_enabled": True, "attempts": attempts})
        if url.endswith("/gateway/usage"):
            seen.usage_reports.append(body)
            return httpx.Response(202)
        return httpx.Response(200, json={"enabled": True})

    async def fake_amessages(**kwargs: Any) -> Any:
        seen.provider_calls.append(kwargs)
        raise _unavailable()

    control_plane_transport(fake_post_platform)
    with (
        patch("gateway.api.routes.messages.amessages", new=fake_amessages),
        patch("gateway.api.routes._pipeline.SandboxBackend", new=lambda **_kwargs: _Unreachable(seen, "sandbox")),
    ):
        response = platform_client.post(
            f"{API_ROOT}/messages",
            json={
                "model": _CLAUDE,
                "messages": [{"role": "user", "content": "go"}],
                "max_tokens": 100,
                "tools": [_DECLARATIONS["code"]],
                **({"stream": True} if stream else {}),
            },
            headers={"Authorization": "Bearer user_test_token"},
        )

    assert response.status_code == 502, response.text
    assert len(seen.provider_calls) == 1
    assert seen.backends == ["sandbox"]
    # A stream reports a failure before its first chunk with a class; the walk without one names none for a backend.
    assert _pinned(seen.usage_reports) == [
        _report(_HEAD, status="error", error_class="http_503", final=False),
        _report(_FALLBACK, status="error", error_class="unknown" if stream else None),
    ]
