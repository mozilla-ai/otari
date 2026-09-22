#!/usr/bin/env python3
"""Boot Otari in hybrid mode against a fake control plane and smoke the tool paths.

The dev gateway at otari.ai deploys from this repository's ``main`` on every
merge, against a control plane pinned to an older release. Every merge is
therefore an integration test between this gateway and a peer it has never met,
run in a live environment with no gate in front of it. The hybrid tests in
``tests/integration`` fake the peer with hand-written dicts, which proves the
gateway handles *its own idea* of the platform. What nothing proves is what the
gateway **sends**: a request body the deployed peer would reject merges green.

This is that gate. It boots the packaged CLI as a subprocess with a platform
token set, so hybrid mode is selected the way a deployment selects it, and
stands up three standard-library fakes: the control plane, an OpenAI- and
Anthropic-compatible provider, and a streamable-HTTP MCP server. Every fake
records what it was asked, and the assertions are on those records as much as
on the responses, because the record is the half of the wire contract no other
test reads.

It walks:

1. Health and readiness report ``mode: hybrid`` with the platform reachable, and
   the bootstrap answers ``hybrid``; the management API is not mounted.
2. A chat completion: the resolve body names the model and provider the caller
   asked for, both tokens travel, the attempt's credential reaches the provider,
   and the usage report that follows names the winning attempt.
3. Budget and rate refusals from the control plane (402, 429 with Retry-After,
   401) reach the caller with the platform's own detail, and no provider call
   or usage report is made for a request the platform refused.
4. Managed web search: the Web Access resolve carries ``requested_tools``, the
   search query reaches the platform-hosted backend with the gateway token, and
   the result feeds the model's second turn.
5. Managed web fetch is off by default: declaring it is refused before any
   resolve is attempted.
6. MCP through the managed tool loop, inline and by workspace id: the id resolve
   carries the ids, the server sees initialize, tools/list and tools/call, and
   the tool result feeds the model's second turn.
7. Provider-native code execution is forwarded untouched when no sandbox is
   configured: Anthropic's dated tool on Messages, OpenAI's ``code_interpreter``
   on Responses, each answered in its own native result blocks.

8. Provider-native web search is forwarded the same way (``web_search_intercept``
   is off by default): Anthropic's ``web_search_20250305`` on Messages, OpenAI's
   ``web_search_preview`` on Responses, each answered in its own result blocks.
9. A streamed completion, which is how most callers actually read a model: the
   gateway injects ``stream_options.include_usage`` so a cost can be settled, the
   frames reach the caller as ``text/event-stream``, and the usage report carries
   the ``ttft_ms`` that only a streamed attempt produces.

Standard library only, and no dev dependencies, for the same reason as
``oss_edition_smoke.py``: CI runs it against ``uv sync --frozen --no-dev``, so a
dev-only import on a hybrid code path fails here.

``--live`` keeps the fake control plane and MCP server but points the resolved
attempts at the real OpenAI and Anthropic APIs, with keys read from
``OTARI_SMOKE_OPENAI_API_KEY`` and ``OTARI_SMOKE_ANTHROPIC_API_KEY`` (models from
``OTARI_SMOKE_OPENAI_MODEL`` and ``OTARI_SMOKE_ANTHROPIC_MODEL``). Managed web
search then runs a real model against the fake search backend, or against Tavily
when ``OTARI_SMOKE_TAVILY_API_KEY`` is also set. The fakes cannot record the
provider side, so those assertions are skipped and the prompts force each tool
with ``tool_choice``; what a live run adds is the class the fakes encode only a
belief about: whether the pinned SDKs still parse a real response, whether a
real model drives the managed loop, and whether the native tools still exist
under the names the gateway forwards. It runs on pushes to ``main``, which is
also the commit the dev gateway deploys.

``--image`` runs the same walk against the published container instead of a
source checkout, which is what a deployment actually runs: the image's own
filesystem, entrypoint and baked dependencies. ``otari-docker-build.yml`` proves
today only that the container answers its health probes, so an image that boots
but cannot serve a request is a break nothing here catches, and that class of
break is the one that took otari-ai's dev deployment down. The fakes then bind
every interface and the container reaches them through ``host.docker.internal``,
so the run works the same on a Linux runner and on Docker Desktop.

Usage:
    uv run --frozen --no-dev python scripts/hybrid_edition_smoke.py
    uv run --frozen --no-dev python scripts/hybrid_edition_smoke.py --live
    uv run --frozen --no-dev python scripts/hybrid_edition_smoke.py --image otari:pr-sha
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, TypeVar

# The root the packaged app serves its API at. Spelled here rather than imported
# so this gate stays standard-library only; a unit test pins it against the
# app's own constant.
API_ROOT = "/api/v1"
# The control plane's own API prefix. The gateway concatenates ``base_url`` and
# the contract paths literally, so this is the reference deployment's shape.
PLATFORM_PREFIX = "/api/v1"
# One header carries the caller's credential; hybrid forwards it as X-User-Token.
KEY_HEADER = "Otari-Key"

GATEWAY_TOKEN = f"gw_hybrid_smoke_{secrets.token_hex(8)}"
# One user token per control-plane behavior the smoke needs.
USER_TOKEN_OK = f"tk_ok_{secrets.token_hex(8)}"
USER_TOKEN_BROKE = f"tk_broke_{secrets.token_hex(8)}"
USER_TOKEN_THROTTLED = f"tk_throttled_{secrets.token_hex(8)}"
USER_TOKEN_UNKNOWN = f"tk_unknown_{secrets.token_hex(8)}"

OPENAI_KEY = f"sk-hybrid-smoke-{secrets.token_hex(8)}"
ANTHROPIC_KEY = f"sk-ant-hybrid-smoke-{secrets.token_hex(8)}"
OPENAI_MODEL = "hybrid-smoke-model"
ANTHROPIC_MODEL = "claude-hybrid-smoke"

REPLY = "hybrid-smoke-ok"
SEARCH_QUERY = "hybrid smoke search query"
SEARCH_TITLE = "Hybrid smoke result"
SEARCH_URL = "https://example.com/hybrid-smoke"
SEARCH_CONTENT = "hybrid-smoke-extracted-content"
MCP_TOOL = "smoke_lookup"
MCP_RESULT = "hybrid-smoke-mcp-result"
MCP_SERVER_ID = "8f2c1a1e-0000-4000-8000-000000000001"
CODE_STDOUT = "hybrid-smoke-code-stdout"
# Delivered before the first frame so the gateway's time-to-first-token is a
# measurable number rather than a rounding artifact.
STREAM_FIRST_FRAME_DELAY_SECONDS = 0.05

RETRY_AFTER_SECONDS = "7"
BROKE_DETAIL = "Wallet is empty"

# Read by the script before the environment is scrubbed for the gateway, so a
# key reaches the gateway only the way a deployment's would: in a resolve answer.
LIVE_OPENAI_KEY_ENV = "OTARI_SMOKE_OPENAI_API_KEY"
LIVE_ANTHROPIC_KEY_ENV = "OTARI_SMOKE_ANTHROPIC_API_KEY"
LIVE_TAVILY_KEY_ENV = "OTARI_SMOKE_TAVILY_API_KEY"
LIVE_OPENAI_MODEL_ENV = "OTARI_SMOKE_OPENAI_MODEL"
LIVE_ANTHROPIC_MODEL_ENV = "OTARI_SMOKE_ANTHROPIC_MODEL"
DEFAULT_LIVE_OPENAI_MODEL = "gpt-4.1-mini"
DEFAULT_LIVE_ANTHROPIC_MODEL = "claude-sonnet-4-5"
# Anthropic's code-execution server tool is behind this beta; the gateway forwards
# the caller's anthropic-beta header, so the caller sends it as a real SDK would.
ANTHROPIC_CODE_EXECUTION_BETA = "code-execution-2025-08-25"

# Every gateway setting is dropped from the environment, then exactly one is set
# back: the platform token, which is what selects hybrid mode. A developer's
# OTARI_MODE or OTARI_BOOTSTRAP would otherwise decide which edition boots.
SCRUBBED_ENV_PREFIXES = ("OTARI_",)
SCRUBBED_ENV_VARS = frozenset({"DATABASE_URL"})
PLATFORM_TOKEN_ENV_VAR = "OTARI_AI_TOKEN"

HEALTH_TIMEOUT_SECONDS = 90
REQUEST_TIMEOUT_SECONDS = 60
# The usage report is sent after the response, so a step waits for it.
REPORT_TIMEOUT_SECONDS = 10


class SmokeFailure(Exception):
    """A smoke step did not do what the hybrid gateway is supposed to do."""


def log(message: str) -> None:
    """Print a progress line, unbuffered so it interleaves correctly in CI logs."""
    print(message, flush=True)


# --------------------------------------------------------------------------- #
# Recording
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Recorded:
    """One request a fake received: enough to assert on what the gateway sent."""

    route: str
    headers: dict[str, str]
    query: dict[str, str]
    body: Any


class Recorder:
    """Thread-safe log of the requests one fake received."""

    def __init__(self) -> None:
        self._items: list[Recorded] = []
        self._lock = threading.Lock()

    def add(self, item: Recorded) -> None:
        with self._lock:
            self._items.append(item)

    def all(self, route: str | None = None) -> list[Recorded]:
        with self._lock:
            return [item for item in self._items if route is None or item.route == route]

    def wait_for(self, route: str, count: int, timeout: float = REPORT_TIMEOUT_SECONDS) -> list[Recorded]:
        """Return the recorded requests on ``route`` once at least ``count`` exist."""
        deadline = time.monotonic() + timeout
        while True:
            items = self.all(route)
            if len(items) >= count:
                return items
            if time.monotonic() > deadline:
                raise SmokeFailure(f"expected {count} request(s) on {route} within {timeout}s, got {len(items)}")
            time.sleep(0.1)


class _RecordingHandler(BaseHTTPRequestHandler):
    """Common request plumbing for the three fakes."""

    protocol_version = "HTTP/1.1"
    server: _FakeServer

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        """Stay quiet: the gateway's own log is the interesting one on a failure."""

    def _read_json(self) -> Any:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        if not raw:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw.decode("utf-8", errors="replace")

    def _record(self, route: str, body: Any) -> Recorded:
        parsed = urllib.parse.urlsplit(self.path)
        query = {key: values[-1] for key, values in urllib.parse.parse_qs(parsed.query).items()}
        item = Recorded(
            route=route,
            headers={key.lower(): value for key, value in self.headers.items()},
            query=query,
            body=body,
        )
        self.server.recorder.add(item)
        return item

    def _respond(self, status: int, payload: Any = None, headers: dict[str, str] | None = None) -> None:
        body = b"" if payload is None else json.dumps(payload).encode()
        self.send_response(status)
        if body:
            self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        if body:
            self.wfile.write(body)

    def _path(self) -> str:
        return urllib.parse.urlsplit(self.path).path


# Where the fakes bind, and the host the gateway reaches them on. They differ
# only for a container run: the gateway is then in its own network namespace, so
# loopback is not shared and the fakes have to be reachable from outside it.
# Bound briefly, on a throwaway port, serving nothing but this run's fixtures.
LOOPBACK = "127.0.0.1"
ALL_INTERFACES = ""
CONTAINER_HOST_ALIAS = "host.docker.internal"
# The image pins OTARI_HOST and OTARI_PORT as environment variables, and an
# env-bridged setting beats a config file, so a container listens here whatever
# the mounted config says. A deployment relies on exactly that, so this run does
# too: a free host port is published to this one.
CONTAINER_PORT = 8000


class _FakeServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, handler: type[BaseHTTPRequestHandler], bind_host: str = LOOPBACK) -> None:
        super().__init__((bind_host, 0), handler)
        self.recorder = Recorder()
        # The address the *gateway* dials, which is not always where we bound.
        self.peer_host = CONTAINER_HOST_ALIAS if bind_host == ALL_INTERFACES else LOOPBACK

    @property
    def port(self) -> int:
        return int(self.server_address[1])

    @property
    def base_url(self) -> str:
        """The URL the gateway should dial to reach this fake."""
        return f"http://{self.peer_host}:{self.port}"


ServerT = TypeVar("ServerT", bound=_FakeServer)


@contextmanager
def serve(server: ServerT, name: str) -> Iterator[ServerT]:
    thread = threading.Thread(target=server.serve_forever, name=name, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


# --------------------------------------------------------------------------- #
# Fake control plane
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class LiveProviders:
    """Real provider credentials for a ``--live`` run. Never logged."""

    openai_key: str
    anthropic_key: str
    # Optional: without it managed search runs against the fake backend.
    tavily_key: str | None
    openai_model: str
    anthropic_model: str

    @classmethod
    def from_env(cls, env: dict[str, str]) -> LiveProviders:
        missing = [name for name in (LIVE_OPENAI_KEY_ENV, LIVE_ANTHROPIC_KEY_ENV) if not env.get(name)]
        if missing:
            raise SmokeFailure(f"--live needs {', '.join(missing)} in the environment")
        return cls(
            openai_key=env[LIVE_OPENAI_KEY_ENV],
            anthropic_key=env[LIVE_ANTHROPIC_KEY_ENV],
            tavily_key=env.get(LIVE_TAVILY_KEY_ENV) or None,
            openai_model=env.get(LIVE_OPENAI_MODEL_ENV) or DEFAULT_LIVE_OPENAI_MODEL,
            anthropic_model=env.get(LIVE_ANTHROPIC_MODEL_ENV) or DEFAULT_LIVE_ANTHROPIC_MODEL,
        )


@dataclass
class ControlPlaneState:
    """What the fake control plane needs to know that is not in the request."""

    provider_base_url: str
    mcp_url: str
    # Set on a --live run: attempts then carry these keys and no api_base.
    live: LiveProviders | None = None
    resolves: int = field(default=0)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def next_ids(self) -> tuple[str, str]:
        with self._lock:
            self.resolves += 1
            n = self.resolves
        return f"req_{n:04d}", f"att_{n:04d}"


class FakeControlPlane(_FakeServer):
    """Speaks docs/hybrid-mode-protocol.md, keyed on which user token is presented."""

    def __init__(self, state: ControlPlaneState, bind_host: str = LOOPBACK) -> None:
        super().__init__(_ControlPlaneHandler, bind_host)
        self.state = state


class _ControlPlaneHandler(_RecordingHandler):
    server: FakeControlPlane

    def do_GET(self) -> None:  # noqa: N802
        path = self._path()
        if path == f"{PLATFORM_PREFIX}/utils/health-check/":
            self._respond(200, {"status": "ok"})
            return
        if path == f"{PLATFORM_PREFIX}/gateway/web-search/search":
            item = self._record("web-search/search", None)
            if item.headers.get("x-gateway-token") != GATEWAY_TOKEN:
                self._respond(401, {"detail": "bad gateway token on search"})
                return
            self._respond(
                200,
                {
                    "results": [
                        {
                            "url": SEARCH_URL,
                            "title": SEARCH_TITLE,
                            "content": "snippet",
                            # Supplied so the gateway does not try to retrieve the
                            # page: retrieval is pinned to public addresses by design.
                            "extracted_content": SEARCH_CONTENT,
                        }
                    ]
                },
            )
            return
        self._respond(404, {"detail": f"fake control plane has no GET {path}"})

    def do_POST(self) -> None:  # noqa: N802
        path = self._path()
        body = self._read_json()
        if not path.startswith(f"{PLATFORM_PREFIX}/gateway/"):
            self._respond(404, {"detail": f"fake control plane has no POST {path}"})
            return
        route = path[len(f"{PLATFORM_PREFIX}/gateway/") :]
        item = self._record(route, body)

        if item.headers.get("x-gateway-token") != GATEWAY_TOKEN:
            self._respond(401, {"detail": "unknown gateway token"})
            return
        if route == "usage":
            self._respond(204)
            return

        user_token = item.headers.get("x-user-token")
        if user_token == USER_TOKEN_BROKE:
            self._respond(402, {"detail": BROKE_DETAIL})
            return
        if user_token == USER_TOKEN_THROTTLED:
            self._respond(429, {"detail": "Too many requests"}, {"Retry-After": RETRY_AFTER_SECONDS})
            return
        if user_token != USER_TOKEN_OK:
            self._respond(401, {"detail": "unknown user token"})
            return

        if route == "provider-keys/resolve":
            self._respond(200, self._resolve(body))
        elif route == "web-search/resolve":
            self._respond(
                200,
                {
                    "enabled": True,
                    "authorized_tools": ["web_search"],
                    "provider": "searxng",
                    "max_results": 3,
                    "allowed_domains": [],
                    "blocked_domains": [],
                    "provider_options": {},
                },
            )
        elif route == "mcp-servers/resolve":
            self._respond(
                200,
                {
                    "servers": [
                        {"id": MCP_SERVER_ID, "name": "smoke", "url": self.server.state.mcp_url, "enabled": True}
                    ]
                },
            )
        else:
            self._respond(404, {"detail": f"fake control plane has no route {route}"})

    def _resolve(self, body: Any) -> dict[str, Any]:
        provider = body.get("provider") if isinstance(body, dict) else None
        model = body.get("model") if isinstance(body, dict) else None
        request_id, attempt_id = self.server.state.next_ids()
        base = self.server.state.provider_base_url
        live = self.server.state.live
        attempt: dict[str, Any]
        if live is not None:
            key = live.anthropic_key if provider == "anthropic" else live.openai_key
            attempt = {"provider": provider or "openai", "api_key": key, "api_base": None}
        elif provider == "anthropic":
            # The Anthropic SDK appends /v1/messages to base_url itself.
            attempt = {"provider": "anthropic", "api_key": ANTHROPIC_KEY, "api_base": f"{base}/anthropic"}
        else:
            attempt = {"provider": "openai", "api_key": OPENAI_KEY, "api_base": f"{base}/openai/v1"}
        attempt["model"] = model
        return {
            "request_id": request_id,
            "fallback_enabled": False,
            "user_id": "hybrid-smoke-user",
            "workspace_id": "hybrid-smoke-workspace",
            "organization_id": "hybrid-smoke-organization",
            "attempts": [{"attempt_id": attempt_id, "position": 0, "managed": False, **attempt}],
        }


# --------------------------------------------------------------------------- #
# Mock provider: OpenAI Chat Completions and Responses, Anthropic Messages
# --------------------------------------------------------------------------- #


class MockProvider(_FakeServer):
    def __init__(self, bind_host: str = LOOPBACK) -> None:
        super().__init__(_ProviderHandler, bind_host)


def _tool_names(body: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for tool in body.get("tools") or []:
        if isinstance(tool, dict):
            function = tool.get("function")
            if isinstance(function, dict) and isinstance(function.get("name"), str):
                names.add(function["name"])
            elif isinstance(tool.get("name"), str):
                names.add(tool["name"])
    return names


def _last_role(body: dict[str, Any]) -> str | None:
    messages = body.get("messages") or []
    last = messages[-1] if messages else None
    return last.get("role") if isinstance(last, dict) else None


class _ProviderHandler(_RecordingHandler):
    server: MockProvider

    def do_POST(self) -> None:  # noqa: N802
        path = self._path()
        body = self._read_json()
        if path == "/openai/v1/chat/completions":
            item = self._record("chat", body)
            if item.headers.get("authorization") != f"Bearer {OPENAI_KEY}":
                self._respond(401, {"error": {"message": "wrong OpenAI key"}})
                return
            request_body = body if isinstance(body, dict) else {}
            if request_body.get("stream"):
                self._respond_sse(self._chat_stream(request_body))
            else:
                self._respond(200, self._chat(request_body))
        elif path == "/openai/v1/responses":
            item = self._record("responses", body)
            if item.headers.get("authorization") != f"Bearer {OPENAI_KEY}":
                self._respond(401, {"error": {"message": "wrong OpenAI key"}})
                return
            self._respond(200, self._responses(body if isinstance(body, dict) else {}))
        elif path == "/anthropic/v1/messages":
            item = self._record("messages", body)
            if item.headers.get("x-api-key") != ANTHROPIC_KEY:
                self._respond(401, {"type": "error", "error": {"type": "authentication_error", "message": "wrong key"}})
                return
            self._respond(200, self._messages(body if isinstance(body, dict) else {}))
        else:
            self._respond(404, {"error": {"message": f"mock provider has no route {path}"}})

    @staticmethod
    def _tool_call_for(body: dict[str, Any]) -> dict[str, str] | None:
        """The tool this turn should call, or None to answer.

        Decided from the request rather than a counter, so the mock stays correct
        however many requests interleave, and shared by the streamed and buffered
        forms so they cannot disagree about when a tool runs.
        """
        if _last_role(body) == "tool":
            return None
        names = _tool_names(body)
        if "web_search" in names:
            return {"name": "web_search", "arguments": json.dumps({"query": SEARCH_QUERY})}
        if MCP_TOOL in names:
            return {"name": MCP_TOOL, "arguments": json.dumps({"term": "smoke"})}
        return None

    @classmethod
    def _chat(cls, body: dict[str, Any]) -> dict[str, Any]:
        """One tool call on the first turn when a tool is offered, then the reply."""
        tool_call = cls._tool_call_for(body)
        if tool_call is not None:
            message: dict[str, Any] = {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "call_hybrid_smoke", "type": "function", "function": tool_call}],
            }
            finish = "tool_calls"
        else:
            message = {"role": "assistant", "content": REPLY}
            finish = "stop"
        return {
            "id": "chatcmpl-hybrid-smoke",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": OPENAI_MODEL,
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
        }

    def _respond_sse(self, frames: list[str]) -> None:
        """Write SSE frames one at a time over a chunked response.

        Chunked rather than one buffered body on purpose: a single write with a
        Content-Length would let a gateway that accumulated the whole stream
        still pass, which is the bug this leg exists to catch.
        """
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        time.sleep(STREAM_FIRST_FRAME_DELAY_SECONDS)
        for frame in frames:
            payload = f"data: {frame}\n\n".encode()
            self.wfile.write(b"%X\r\n%s\r\n" % (len(payload), payload))
            self.wfile.flush()
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    @classmethod
    def _chat_stream(cls, body: dict[str, Any]) -> list[str]:
        """The streamed form of :meth:`_chat`, as OpenAI delivers it.

        A tool call arrives split across fragments, with the name in the first
        and the arguments in a later one, because that is what a real provider
        sends and what the gateway's slot accumulator has to survive. Nothing
        walks that branch yet: the leg that does is held back by otari#1504,
        where a streamed hybrid tool loop truncates its own stream, and lands
        with that fix.
        """
        base = {
            "id": "chatcmpl-hybrid-smoke-stream",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": OPENAI_MODEL,
        }

        def chunk(delta: dict[str, Any], finish: str | None = None) -> str:
            return json.dumps({**base, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]})

        frames = [chunk({"role": "assistant"})]
        tool_call = cls._tool_call_for(body)
        if tool_call is not None:
            frames.append(
                chunk(
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_hybrid_smoke_stream",
                                "type": "function",
                                "function": {"name": tool_call["name"], "arguments": ""},
                            }
                        ]
                    }
                )
            )
            frames.append(chunk({"tool_calls": [{"index": 0, "function": {"arguments": tool_call["arguments"]}}]}))
            frames.append(chunk({}, "tool_calls"))
        else:
            frames.extend(chunk({"content": piece}) for piece in (REPLY[: len(REPLY) // 2], REPLY[len(REPLY) // 2 :]))
            frames.append(chunk({}, "stop"))
        # The include_usage carrier: usage and no choices, which is the shape the
        # gateway requires before it will settle a streamed cost.
        frames.append(
            json.dumps(
                {**base, "choices": [], "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}}
            )
        )
        frames.append("[DONE]")
        return frames

    @staticmethod
    def _declared_types(body: dict[str, Any]) -> set[str]:
        types = (tool.get("type") for tool in body.get("tools") or [] if isinstance(tool, dict))
        return {t for t in types if isinstance(t, str)}

    @classmethod
    def _responses(cls, body: dict[str, Any]) -> dict[str, Any]:
        """A Responses answer that ran the declared built-in tool itself."""
        if "web_search_preview" in cls._declared_types(body):
            tool_item: dict[str, Any] = {"type": "web_search_call", "id": "ws_hybrid_smoke", "status": "completed"}
        else:
            tool_item = {
                "type": "code_interpreter_call",
                "id": "ci_hybrid_smoke",
                "status": "completed",
                "container_id": "cntr_hybrid_smoke",
                "code": "print('x')",
                "outputs": [{"type": "logs", "logs": CODE_STDOUT}],
            }
        return {
            "id": "resp_hybrid_smoke",
            "object": "response",
            "created_at": int(time.time()),
            "status": "completed",
            "model": body.get("model", OPENAI_MODEL),
            "output": [
                tool_item,
                {
                    "type": "message",
                    "id": "msg_hybrid_smoke",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": REPLY, "annotations": []}],
                },
            ],
            "usage": {
                "input_tokens": 11,
                "output_tokens": 7,
                "total_tokens": 18,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0},
            },
        }

    @classmethod
    def _messages(cls, body: dict[str, Any]) -> dict[str, Any]:
        """A Messages answer that ran the declared server tool itself."""
        if any(t.startswith("web_search") for t in cls._declared_types(body)):
            use: dict[str, Any] = {"name": "web_search", "input": {"query": SEARCH_QUERY}}
            result: dict[str, Any] = {
                "type": "web_search_tool_result",
                "tool_use_id": "srvtoolu_hybrid_smoke",
                "content": [
                    {
                        "type": "web_search_result",
                        "url": SEARCH_URL,
                        "title": SEARCH_TITLE,
                        "encrypted_content": "aGVsbG8=",
                        "page_age": None,
                    }
                ],
            }
        else:
            use = {"name": "code_execution", "input": {"code": "print('x')"}}
            result = {
                "type": "code_execution_tool_result",
                "tool_use_id": "srvtoolu_hybrid_smoke",
                "content": {
                    "type": "code_execution_result",
                    "stdout": CODE_STDOUT,
                    "stderr": "",
                    "return_code": 0,
                    "content": [],
                },
            }
        return {
            "id": "msg_hybrid_smoke",
            "type": "message",
            "role": "assistant",
            "model": body.get("model", ANTHROPIC_MODEL),
            "content": [
                {"type": "server_tool_use", "id": "srvtoolu_hybrid_smoke", **use},
                result,
                {"type": "text", "text": REPLY},
            ],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 11, "output_tokens": 7},
        }


# --------------------------------------------------------------------------- #
# Fake MCP server (streamable HTTP, JSON responses)
# --------------------------------------------------------------------------- #


class FakeMcpServer(_FakeServer):
    def __init__(self, bind_host: str = LOOPBACK) -> None:
        super().__init__(_McpHandler, bind_host)

    @property
    def mcp_url(self) -> str:
        return f"{self.base_url}/mcp"


class _McpHandler(_RecordingHandler):
    server: FakeMcpServer

    def do_GET(self) -> None:  # noqa: N802
        # No server-initiated stream here; the client treats 405 as "none offered".
        self._respond(405)

    def do_DELETE(self) -> None:  # noqa: N802
        self._respond(200)

    def do_POST(self) -> None:  # noqa: N802
        if self._path() != "/mcp":
            self._respond(404)
            return
        message = self._read_json()
        if not isinstance(message, dict):
            self._respond(400)
            return
        method = message.get("method")
        self._record(str(method), message.get("params"))
        if "id" not in message:
            # A notification (notifications/initialized): accepted, no body.
            self._respond(202)
            return
        result: dict[str, Any]
        if method == "initialize":
            params = message.get("params") or {}
            result = {
                "protocolVersion": params.get("protocolVersion", "2025-06-18"),
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "hybrid-smoke-mcp", "version": "0"},
            }
        elif method == "tools/list":
            result = {
                "tools": [
                    {
                        "name": MCP_TOOL,
                        "description": "Look up a smoke term.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {"term": {"type": "string"}},
                            "required": ["term"],
                        },
                    }
                ]
            }
        elif method == "tools/call":
            result = {"content": [{"type": "text", "text": MCP_RESULT}], "isError": False}
        elif method == "ping":
            result = {}
        else:
            self._respond(
                200,
                {"jsonrpc": "2.0", "id": message["id"], "error": {"code": -32601, "message": f"no {method}"}},
            )
            return
        self._respond(200, {"jsonrpc": "2.0", "id": message["id"], "result": result})


# --------------------------------------------------------------------------- #
# Gateway configuration, environment and process
# --------------------------------------------------------------------------- #


def hybrid_env(base_env: dict[str, str]) -> dict[str, str]:
    """The environment the hybrid gateway boots in: scrubbed, then the token set."""
    env = {
        name: value
        for name, value in base_env.items()
        if name not in SCRUBBED_ENV_VARS and not name.startswith(SCRUBBED_ENV_PREFIXES)
    }
    env[PLATFORM_TOKEN_ENV_VAR] = GATEWAY_TOKEN
    return env


def hybrid_config(
    *,
    port: int,
    platform_base_url: str,
    tavily_key: str | None = None,
    in_container: bool = False,
) -> dict[str, Any]:
    """The config a hybrid deployment writes: a platform block and no providers.

    No ``database_url``: a hybrid gateway runs no database. No ``sandbox_url``,
    so a provider-native code-execution declaration is forwarded untouched,
    which is the path step 7 proves. ``web_search_url`` sits under the platform
    base URL, which is what makes the gateway send its token on search queries.
    A live run adds Tavily, which the backend prefers over the URL.
    ``in_container`` leaves out ``host`` and ``port``, which the image's own
    environment owns and a config file cannot override.
    """
    config: dict[str, Any] = {
        "platform": {"base_url": platform_base_url, "resolve_timeout_ms": 5000},
        # The gateway appends /search itself; the doc's GET {base}/gateway/web-search/search.
        "web_search_url": f"{platform_base_url}/gateway/web-search",
    }
    if not in_container:
        config["host"] = LOOPBACK
        config["port"] = port
    if tavily_key:
        config["web_search_provider"] = "tavily"
        config["web_search_provider_api_key"] = tavily_key
        # Real pages are retrieved through the pinned transport; keep it short.
        config["web_search_max_results"] = 2
    return config


def write_config(path: Path, config: dict[str, Any]) -> None:
    """JSON, which the loader's ``yaml.safe_load`` accepts."""
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def _request(
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    payload: dict[str, Any] | None = None,
) -> tuple[int, Any, dict[str, str]]:
    """Send one request; return status, decoded body (None if not JSON) and headers."""
    body = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(url, data=body, method=method)
    request.add_header("Accept", "application/json")
    if body is not None:
        request.add_header("Content-Type", "application/json")
    for name, value in (headers or {}).items():
        request.add_header(name, value)
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            return response.status, _decode(response.read()), {k.lower(): v for k, v in response.headers.items()}
    except urllib.error.HTTPError as error:
        return error.code, _decode(error.read()), {k.lower(): v for k, v in error.headers.items()}


def _stream_request(
    url: str,
    *,
    headers: dict[str, str],
    payload: dict[str, Any],
) -> tuple[int, dict[str, str], list[str], list[Any]]:
    """POST and read a text/event-stream, returning its raw and decoded frames.

    Frames are read as they arrive rather than after the body completes, which is
    what makes this a check of streaming and not of a buffered response.
    """
    request = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST")
    request.add_header("Content-Type", "application/json")
    request.add_header("Accept", "text/event-stream")
    for name, value in headers.items():
        request.add_header(name, value)
    raw: list[str] = []
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            status = response.status
            response_headers = {k.lower(): v for k, v in response.headers.items()}
            for line in response:
                text = line.decode("utf-8", errors="replace").strip()
                if text.startswith("data:"):
                    raw.append(text[len("data:") :].strip())
    except urllib.error.HTTPError as error:
        return error.code, {k.lower(): v for k, v in error.headers.items()}, [], [_decode(error.read())]
    decoded = [json.loads(frame) for frame in raw if frame and frame != "[DONE]"]
    return status, response_headers, raw, decoded


def _decode(raw: bytes) -> Any:
    text = raw.decode("utf-8", errors="replace")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _free_port() -> int:
    with closing(socket.socket()) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _otari_executable() -> str:
    executable = shutil.which("otari")
    if executable is None:
        raise SmokeFailure(
            "The 'otari' CLI is not on PATH. Run this through the project environment, "
            "for example: uv run --frozen --no-dev python scripts/hybrid_edition_smoke.py"
        )
    return executable


@contextmanager
def gateway(config_path: Path, env: dict[str, str], base_url: str, log_path: Path) -> Iterator[None]:
    """Serve the hybrid gateway and wait for it to be healthy. No migration: no database."""
    log("Starting the hybrid gateway")
    with log_path.open("wb") as log_file:
        process = subprocess.Popen(
            [_otari_executable(), "serve", "--config", str(config_path)],
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
    try:
        _await_health(process, base_url)
        yield
    finally:
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=20)


@contextmanager
def gateway_container(image: str, config_path: Path, port: int, base_url: str, log_path: Path) -> Iterator[None]:
    """Run the published image in hybrid mode and wait for it to be healthy.

    ``port`` is the host port published to the container's own
    :data:`CONTAINER_PORT`. ``host.docker.internal`` is mapped explicitly rather
    than assumed: Docker
    Desktop provides it, a Linux runner does not, and ``host-gateway`` is what
    makes the same command work on both.
    """
    if shutil.which("docker") is None:
        raise SmokeFailure("--image needs docker on PATH")
    name = f"otari-hybrid-smoke-{secrets.token_hex(4)}"
    log(f"Starting the hybrid gateway from {image}")
    started = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--detach",
            "--name",
            name,
            "--add-host",
            f"{CONTAINER_HOST_ALIAS}:host-gateway",
            "--publish",
            f"{port}:{CONTAINER_PORT}",
            "--env",
            f"{PLATFORM_TOKEN_ENV_VAR}={GATEWAY_TOKEN}",
            # The MCP fake runs on the host, which from inside the container is a
            # private address, and the SSRF guard rejects it by design (it names
            # this override in the refusal). Relaxed only here, only for MCP, and
            # only because the peer is outside the container's namespace; the
            # guard itself is covered by tests/unit/test_url_safety.py.
            "--env",
            "OTARI_MCP_ALLOW_PRIVATE_HOSTS=true",
            "--volume",
            f"{config_path}:/tmp/hybrid.yml:ro",
            image,
            "otari",
            "serve",
            "--config",
            "/tmp/hybrid.yml",
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    if started.returncode != 0:
        raise SmokeFailure(f"'docker run' exited {started.returncode}: {started.stderr.strip()}")
    try:
        _await_container_health(name, base_url)
        yield
    finally:
        # Captured before the container goes, so a failure still has its log.
        logs = subprocess.run(["docker", "logs", name], capture_output=True, check=False, text=True)
        log_path.write_text(logs.stdout + logs.stderr, encoding="utf-8")
        subprocess.run(["docker", "rm", "--force", name], capture_output=True, check=False)


def _container_is_running(name: str) -> bool:
    probe = subprocess.run(
        ["docker", "inspect", "--format", "{{.State.Running}}", name],
        capture_output=True,
        check=False,
        text=True,
    )
    return probe.returncode == 0 and probe.stdout.strip() == "true"


def _await_container_health(name: str, base_url: str) -> None:
    deadline = time.monotonic() + HEALTH_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if not _container_is_running(name):
            raise SmokeFailure("The gateway container exited before becoming healthy")
        try:
            status, _, _ = _request("GET", f"{base_url}{API_ROOT}/health")
        except OSError:
            status = 0
        if status == 200:
            return
        time.sleep(0.5)
    raise SmokeFailure(f"The gateway container did not answer {API_ROOT}/health within {HEALTH_TIMEOUT_SECONDS}s")


def _await_health(process: subprocess.Popen[bytes], base_url: str) -> None:
    deadline = time.monotonic() + HEALTH_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise SmokeFailure(f"The hybrid gateway exited with code {process.returncode} before becoming healthy")
        try:
            status, _, _ = _request("GET", f"{base_url}{API_ROOT}/health")
        except OSError:
            status = 0
        if status == 200:
            return
        time.sleep(0.5)
    raise SmokeFailure(f"The hybrid gateway did not answer {API_ROOT}/health within {HEALTH_TIMEOUT_SECONDS}s")


def _tail(log_path: Path, lines: int = 80) -> str:
    if not log_path.exists():
        return "=== no gateway log ==="
    tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]
    return "=== gateway log (tail) ===\n" + "\n".join(tail)


# --------------------------------------------------------------------------- #
# Smoke steps
# --------------------------------------------------------------------------- #


@dataclass
class Fakes:
    control_plane: FakeControlPlane
    provider: MockProvider
    mcp: FakeMcpServer
    # Real providers: the mock provider records nothing, and prompts force tools.
    live: LiveProviders | None = None
    # X-Correlation-ID of every 200 the caller received: the attempts that were
    # dispatched, and therefore the ones a usage report is owed for.
    dispatched: list[str] = field(default_factory=list)

    @property
    def openai_model(self) -> str:
        return self.live.openai_model if self.live else OPENAI_MODEL

    @property
    def anthropic_model(self) -> str:
        return self.live.anthropic_model if self.live else ANTHROPIC_MODEL

    def note_dispatched(self, headers: dict[str, str], what: str) -> None:
        attempt = headers.get("x-correlation-id")
        _check(bool(attempt), f"{what}: no X-Correlation-ID on a served response: {headers!r}")
        self.dispatched.append(str(attempt))


def _expect(status: int, expected: int, what: str, body: Any) -> None:
    if status != expected:
        raise SmokeFailure(f"{what}: expected HTTP {expected}, got {status}: {body!r}")


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeFailure(message)


def _content_of(body: Any) -> str:
    if isinstance(body, dict):
        choices = body.get("choices") or []
        if choices and isinstance(choices[0], dict):
            return (choices[0].get("message") or {}).get("content") or ""
    return ""


def check_health_and_edition(base_url: str) -> None:
    """The probes say hybrid with the platform reachable, and no management API is mounted."""
    status, body, _ = _request("GET", f"{base_url}{API_ROOT}/health")
    _expect(status, 200, f"GET {API_ROOT}/health", body)
    _check(isinstance(body, dict) and body.get("mode") == "hybrid", f"health does not report hybrid: {body!r}")
    _check(body.get("platform_reachable") == "yes", f"health does not see the control plane: {body!r}")

    status, body, _ = _request("GET", f"{base_url}{API_ROOT}/health/readiness")
    _expect(status, 200, f"GET {API_ROOT}/health/readiness", body)
    _check(isinstance(body, dict) and body.get("platform") == "connected", f"readiness: {body!r}")

    status, body, _ = _request("GET", f"{base_url}{API_ROOT}/bootstrap")
    _expect(status, 200, f"GET {API_ROOT}/bootstrap", body)
    _check(isinstance(body, dict) and body.get("deployment_type") == "hybrid", f"bootstrap: {body!r}")

    # Management routers are standalone-only; a hybrid gateway must not answer them.
    status, body, _ = _request("GET", f"{base_url}{API_ROOT}/keys", headers={KEY_HEADER: USER_TOKEN_OK})
    _check(status == 404, f"GET {API_ROOT}/keys answered {status} on a hybrid gateway: {body!r}")
    log("Health, readiness and bootstrap report hybrid; the management API is not mounted")


def run_completion(base_url: str, fakes: Fakes) -> None:
    """One completion: resolve body, both tokens, the attempt's key, and the usage report."""
    before = len(fakes.control_plane.recorder.all("usage"))
    status, body, headers = _request(
        "POST",
        f"{base_url}{API_ROOT}/chat/completions",
        headers={KEY_HEADER: USER_TOKEN_OK},
        payload={
            "model": f"openai:{fakes.openai_model}",
            "messages": [{"role": "user", "content": f"Reply with exactly: {REPLY}"}],
        },
    )
    _expect(status, 200, f"POST {API_ROOT}/chat/completions", body)
    _check(bool(_content_of(body).strip()), f"the completion came back empty: {body!r}")
    if not fakes.live:
        _check(REPLY in _content_of(body), f"the completion did not come from the mock provider: {body!r}")

    resolves = fakes.control_plane.recorder.all("provider-keys/resolve")
    _check(len(resolves) == 1, f"expected one resolve, got {len(resolves)}")
    resolve = resolves[0]
    _check(
        resolve.body == {"model": fakes.openai_model, "provider": "openai"},
        f"resolve body is not what the contract documents: {resolve.body!r}",
    )
    _check(resolve.headers.get("x-gateway-token") == GATEWAY_TOKEN, "resolve did not carry X-Gateway-Token")
    _check(resolve.headers.get("x-user-token") == USER_TOKEN_OK, "resolve did not forward the caller's token")

    if not fakes.live:
        chats = fakes.provider.recorder.all("chat")
        _check(len(chats) == 1, f"expected one provider call, got {len(chats)}")
        _check(chats[0].body.get("model") == OPENAI_MODEL, f"provider saw model {chats[0].body.get('model')!r}")

    _check(headers.get("x-otari-request-id") == "req_0001", f"X-Otari-Request-ID missing or wrong: {headers!r}")
    _check(headers.get("x-correlation-id") == "att_0001", f"X-Correlation-ID missing or wrong: {headers!r}")
    fakes.note_dispatched(headers, "the completion")

    reports = fakes.control_plane.recorder.wait_for("usage", before + 1)
    report = reports[-1].body
    _check(isinstance(report, dict), f"usage report is not an object: {report!r}")
    _check(report.get("correlation_id") == "att_0001", f"usage report names the wrong attempt: {report!r}")
    _check(report.get("status") == "success", f"usage report status: {report!r}")
    _check(report.get("is_final_attempt") is True, f"usage report is_final_attempt: {report!r}")
    usage = report.get("usage") or {}
    if fakes.live:
        _check(int(usage.get("total_tokens") or 0) > 0, f"usage report carries no tokens: {report!r}")
    else:
        _check(usage.get("total_tokens") == 18, f"usage report tokens do not match the provider's: {report!r}")
    _check(reports[-1].headers.get("x-gateway-token") == GATEWAY_TOKEN, "usage report did not carry the token")
    log("A completion resolved, dispatched with the attempt's credential, and was reported")


def check_platform_refusals(base_url: str, fakes: Fakes) -> None:
    """A refusal from the control plane reaches the caller with the platform's detail.

    None of these may call a provider or send a usage report: there was no
    attempt to report.
    """
    provider_calls = len(fakes.provider.recorder.all())
    reports = len(fakes.control_plane.recorder.all("usage"))
    payload = {"model": f"openai:{fakes.openai_model}", "messages": [{"role": "user", "content": "hello"}]}

    status, body, _ = _request(
        "POST", f"{base_url}{API_ROOT}/chat/completions", headers={KEY_HEADER: USER_TOKEN_BROKE}, payload=payload
    )
    _expect(status, 402, "a caller whose wallet is empty", body)
    _check(BROKE_DETAIL in json.dumps(body), f"the platform's 402 detail was not forwarded: {body!r}")

    status, body, headers = _request(
        "POST", f"{base_url}{API_ROOT}/chat/completions", headers={KEY_HEADER: USER_TOKEN_THROTTLED}, payload=payload
    )
    _expect(status, 429, "a caller the platform is throttling", body)
    _check(headers.get("retry-after") == RETRY_AFTER_SECONDS, f"Retry-After not preserved: {headers!r}")

    status, body, _ = _request(
        "POST", f"{base_url}{API_ROOT}/chat/completions", headers={KEY_HEADER: USER_TOKEN_UNKNOWN}, payload=payload
    )
    _expect(status, 401, "a caller the platform does not know", body)

    _check(len(fakes.provider.recorder.all()) == provider_calls, "a refused request still reached the provider")
    time.sleep(0.5)
    _check(len(fakes.control_plane.recorder.all("usage")) == reports, "a refused request was reported as usage")
    log("Budget, rate-limit and authentication refusals are forwarded, and nothing is dispatched or reported")


def run_web_search(base_url: str, fakes: Fakes) -> None:
    """Managed web search: resolve body, tokened search query, and the result reaching the model."""
    chats_before = len(fakes.provider.recorder.all("chat"))
    status, body, headers = _request(
        "POST",
        f"{base_url}{API_ROOT}/chat/completions",
        headers={KEY_HEADER: USER_TOKEN_OK},
        payload={
            "model": f"openai:{fakes.openai_model}",
            "messages": [{"role": "user", "content": f"Use the web_search tool to search for: {SEARCH_QUERY}"}],
            "tools": [{"type": "otari_web_search"}],
            # Forced so a real model calls the tool on turn one; the mock does anyway.
            "tool_choice": {"type": "function", "function": {"name": "web_search"}},
        },
    )
    _expect(status, 200, "a completion declaring otari_web_search", body)
    fakes.note_dispatched(headers, "the search-assisted completion")
    _check(bool(_content_of(body).strip()), f"the search-assisted completion came back empty: {body!r}")

    resolves = fakes.control_plane.recorder.all("web-search/resolve")
    _check(len(resolves) == 1, f"expected one Web Access resolve, got {len(resolves)}")
    _check(
        resolves[0].body == {"requested_tools": ["web_search"]},
        f"Web Access resolve body is not what the contract documents: {resolves[0].body!r}",
    )
    _check(resolves[0].headers.get("x-user-token") == USER_TOKEN_OK, "Web Access resolve did not forward the token")
    if fakes.live and fakes.live.tavily_key:
        # The search ran against Tavily and the pages were retrieved for real,
        # neither of which a fake observes; the loop completing is the proof.
        log("Managed web search resolved with requested_tools and completed against Tavily")
        return

    searches = fakes.control_plane.recorder.all("web-search/search")
    _check(len(searches) >= 1, "the search backend was never queried")
    _check(searches[0].headers.get("x-gateway-token") == GATEWAY_TOKEN, "the search query carried no gateway token")
    _check(searches[0].query.get("format") == "json", f"search format: {searches[0].query!r}")
    if fakes.live:
        log("Managed web search resolved with requested_tools; a real model queried the platform's backend")
        return
    _check(len(searches) == 1, f"expected one search query, got {len(searches)}")
    _check(searches[0].query.get("q") == SEARCH_QUERY, f"search query: {searches[0].query!r}")

    chats = fakes.provider.recorder.all("chat")[chats_before:]
    _check(len(chats) == 2, f"expected two model turns around the search, got {len(chats)}")
    _check(REPLY in _content_of(body), f"the search-assisted completion did not finish: {body!r}")
    tool_message = json.dumps(chats[1].body.get("messages"))
    _check(SEARCH_TITLE in tool_message and SEARCH_URL in tool_message, "the result did not reach the model")
    log("Managed web search resolved with requested_tools, queried the platform with the token, and fed the model")


def check_web_fetch_is_off_by_default(base_url: str, fakes: Fakes) -> None:
    """Managed fetch is opt-in; declaring it is refused before any resolve is made."""
    resolves = len(fakes.control_plane.recorder.all("web-search/resolve"))
    status, body, _ = _request(
        "POST",
        f"{base_url}{API_ROOT}/chat/completions",
        headers={KEY_HEADER: USER_TOKEN_OK},
        payload={
            "model": f"openai:{fakes.openai_model}",
            "messages": [{"role": "user", "content": "fetch something"}],
            "tools": [{"type": "otari_web_fetch"}],
        },
    )
    _check(status in {400, 403}, f"otari_web_fetch on a default deployment answered {status}: {body!r}")
    _check(len(fakes.control_plane.recorder.all("web-search/resolve")) == resolves, "fetch reached the resolve")
    log("Managed web fetch is off by default and refused before any resolve")


def run_mcp(base_url: str, fakes: Fakes) -> None:
    """MCP through the managed loop, inline and by workspace id."""
    for label, extra in (
        ("inline", {"mcp_servers": [{"name": "smoke", "url": fakes.mcp.mcp_url}]}),
        ("by id", {"mcp_server_ids": [MCP_SERVER_ID]}),
    ):
        chats_before = len(fakes.provider.recorder.all("chat"))
        calls_before = len(fakes.mcp.recorder.all("tools/call"))
        status, body, headers = _request(
            "POST",
            f"{base_url}{API_ROOT}/chat/completions",
            headers={KEY_HEADER: USER_TOKEN_OK},
            payload={
                "model": f"openai:{fakes.openai_model}",
                "messages": [{"role": "user", "content": f"Use the {MCP_TOOL} tool with term 'smoke'."}],
                "tool_choice": {"type": "function", "function": {"name": MCP_TOOL}},
                **extra,
            },
        )
        _expect(status, 200, f"a completion with MCP {label}", body)
        fakes.note_dispatched(headers, f"the MCP-assisted completion ({label})")
        _check(bool(_content_of(body).strip()), f"the MCP-assisted completion ({label}) came back empty: {body!r}")

        calls = fakes.mcp.recorder.all("tools/call")[calls_before:]
        _check(len(calls) >= 1, f"expected a tools/call ({label}), got none")
        _check(calls[0].body.get("name") == MCP_TOOL, f"tools/call named {calls[0].body.get('name')!r}")
        if fakes.live:
            continue
        _check(len(calls) == 1, f"expected one tools/call ({label}), got {len(calls)}")
        _check(calls[0].body.get("arguments") == {"term": "smoke"}, f"tools/call arguments: {calls[0].body!r}")
        _check(REPLY in _content_of(body), f"the MCP-assisted completion ({label}) did not finish: {body!r}")

        chats = fakes.provider.recorder.all("chat")[chats_before:]
        _check(len(chats) == 2, f"expected two model turns around the MCP call ({label}), got {len(chats)}")
        second_turn = json.dumps(chats[1].body.get("messages"))
        _check(MCP_RESULT in second_turn, f"the MCP result did not reach the model ({label})")

    _check(bool(fakes.mcp.recorder.all("initialize")), "the MCP server never saw initialize")
    _check(bool(fakes.mcp.recorder.all("tools/list")), "the MCP server never saw tools/list")
    resolves = fakes.control_plane.recorder.all("mcp-servers/resolve")
    _check(len(resolves) == 1, f"expected one MCP id resolve, got {len(resolves)}")
    _check(resolves[0].body == {"mcp_server_ids": [MCP_SERVER_ID]}, f"MCP resolve body: {resolves[0].body!r}")
    log("MCP ran through the managed loop, inline and by workspace id, with the id resolve as documented")


def run_native_code_execution(base_url: str, fakes: Fakes) -> None:
    """A provider-native code-execution declaration is forwarded and answered natively."""
    # Anthropic's dated server tool on Messages.
    prompt = "Use the code execution tool to compute 2**10 in Python and report the printed result."
    status, body, headers = _request(
        "POST",
        f"{base_url}{API_ROOT}/messages",
        headers={KEY_HEADER: USER_TOKEN_OK, "anthropic-beta": ANTHROPIC_CODE_EXECUTION_BETA},
        payload={
            "model": f"anthropic:{fakes.anthropic_model}",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": prompt}],
            "tools": [{"type": "code_execution_20250825", "name": "code_execution"}],
        },
    )
    _expect(status, 200, f"POST {API_ROOT}/messages with code_execution_20250825", body)
    fakes.note_dispatched(headers, "the Messages call")
    blocks = body.get("content") if isinstance(body, dict) else None
    types = [block.get("type") for block in blocks or [] if isinstance(block, dict)]
    _check("server_tool_use" in types, f"no server_tool_use block came back: {body!r}")
    _check("code_execution_tool_result" in types, f"no code_execution_tool_result block came back: {body!r}")
    resolves = fakes.control_plane.recorder.all("provider-keys/resolve")
    _check(
        resolves[-1].body == {"model": fakes.anthropic_model, "provider": "anthropic"},
        f"resolve for the Messages call: {resolves[-1].body!r}",
    )
    if not fakes.live:
        _check(CODE_STDOUT in json.dumps(body), "the provider's stdout did not reach the caller")
        messages = fakes.provider.recorder.all("messages")
        _check(len(messages) == 1, f"expected one Messages call, got {len(messages)}")
        forwarded = {tool.get("type") for tool in messages[0].body.get("tools") or [] if isinstance(tool, dict)}
        _check("code_execution_20250825" in forwarded, f"the declaration did not reach Anthropic: {forwarded!r}")

    # OpenAI's code_interpreter on Responses, forced so a real model runs it.
    status, body, headers = _request(
        "POST",
        f"{base_url}{API_ROOT}/responses",
        headers={KEY_HEADER: USER_TOKEN_OK},
        payload={
            "model": f"openai:{fakes.openai_model}",
            "input": prompt,
            "tools": [{"type": "code_interpreter", "container": {"type": "auto"}}],
            "tool_choice": {"type": "code_interpreter"},
        },
    )
    _expect(status, 200, f"POST {API_ROOT}/responses with code_interpreter", body)
    fakes.note_dispatched(headers, "the Responses call")
    output = body.get("output") if isinstance(body, dict) else None
    types = [item.get("type") for item in output or [] if isinstance(item, dict)]
    _check("code_interpreter_call" in types, f"no code_interpreter_call item came back: {body!r}")
    if not fakes.live:
        _check(CODE_STDOUT in json.dumps(body), "the interpreter's logs did not reach the caller")
        responses = fakes.provider.recorder.all("responses")
        _check(len(responses) == 1, f"expected one Responses call, got {len(responses)}")
        forwarded = {tool.get("type") for tool in responses[0].body.get("tools") or [] if isinstance(tool, dict)}
        _check("code_interpreter" in forwarded, f"the declaration did not reach OpenAI: {forwarded!r}")
    where = "against the real APIs" if fakes.live else "and answered natively"
    log(f"Provider-native code execution was forwarded on Messages and Responses {where}")


def run_native_web_search(base_url: str, fakes: Fakes) -> None:
    """A provider-native web-search declaration is forwarded and answered natively."""
    prompt = f"Use web search to look up: {SEARCH_QUERY}. Then summarize in one sentence."
    status, body, headers = _request(
        "POST",
        f"{base_url}{API_ROOT}/messages",
        headers={KEY_HEADER: USER_TOKEN_OK},
        payload={
            "model": f"anthropic:{fakes.anthropic_model}",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": prompt}],
            "tools": [{"type": "web_search_20250305", "name": "web_search", "max_uses": 1}],
        },
    )
    _expect(status, 200, f"POST {API_ROOT}/messages with web_search_20250305", body)
    fakes.note_dispatched(headers, "the native-search Messages call")
    blocks = body.get("content") if isinstance(body, dict) else None
    types = [block.get("type") for block in blocks or [] if isinstance(block, dict)]
    _check("server_tool_use" in types, f"no server_tool_use block came back: {body!r}")
    _check("web_search_tool_result" in types, f"no web_search_tool_result block came back: {body!r}")
    if not fakes.live:
        messages = fakes.provider.recorder.all("messages")
        forwarded = {tool.get("type") for tool in messages[-1].body.get("tools") or [] if isinstance(tool, dict)}
        _check("web_search_20250305" in forwarded, f"the declaration did not reach Anthropic: {forwarded!r}")
        _check(SEARCH_URL in json.dumps(body), "the provider's search result did not reach the caller")

    status, body, headers = _request(
        "POST",
        f"{base_url}{API_ROOT}/responses",
        headers={KEY_HEADER: USER_TOKEN_OK},
        payload={
            "model": f"openai:{fakes.openai_model}",
            "input": prompt,
            "tools": [{"type": "web_search_preview"}],
            "tool_choice": {"type": "web_search_preview"},
        },
    )
    _expect(status, 200, f"POST {API_ROOT}/responses with web_search_preview", body)
    fakes.note_dispatched(headers, "the native-search Responses call")
    output = body.get("output") if isinstance(body, dict) else None
    types = [item.get("type") for item in output or [] if isinstance(item, dict)]
    _check("web_search_call" in types, f"no web_search_call item came back: {body!r}")
    if not fakes.live:
        responses = fakes.provider.recorder.all("responses")
        forwarded = {tool.get("type") for tool in responses[-1].body.get("tools") or [] if isinstance(tool, dict)}
        _check("web_search_preview" in forwarded, f"the declaration did not reach OpenAI: {forwarded!r}")
    where = "against the real APIs" if fakes.live else "and answered natively"
    log(f"Provider-native web search was forwarded on Messages and Responses {where}")


def _streamed_content(frames: list[Any]) -> str:
    """Assemble the assistant text from chat-completion chunks."""
    out = []
    for frame in frames:
        for choice in frame.get("choices") or []:
            piece = (choice.get("delta") or {}).get("content")
            if isinstance(piece, str):
                out.append(piece)
    return "".join(out)


def run_streaming_completion(base_url: str, fakes: Fakes) -> None:
    """A streamed completion: real SSE out, include_usage in, ttft_ms reported.

    ``ttft_ms`` is the assertion worth having here. It is computed from the first
    chunk and sent only for a streamed attempt, so its presence is what separates
    this path from the buffered one in the platform's own record.
    """
    before = len(fakes.control_plane.recorder.all("usage"))
    chats_before = len(fakes.provider.recorder.all("chat"))
    status, headers, raw, frames = _stream_request(
        f"{base_url}{API_ROOT}/chat/completions",
        headers={KEY_HEADER: USER_TOKEN_OK},
        payload={
            "model": f"openai:{fakes.openai_model}",
            "messages": [{"role": "user", "content": f"Reply with exactly: {REPLY}"}],
            "stream": True,
        },
    )
    _expect(status, 200, f"POST {API_ROOT}/chat/completions with stream=true", frames)
    _check("text/event-stream" in headers.get("content-type", ""), f"not an SSE response: {headers!r}")
    _check(bool(raw) and raw[-1] == "[DONE]", f"the stream did not terminate with [DONE]: {raw[-3:]!r}")
    fakes.note_dispatched(headers, "the streamed completion")
    _check(bool(_streamed_content(frames).strip()), f"the streamed completion carried no text: {raw!r}")
    if not fakes.live:
        _check(REPLY == _streamed_content(frames), f"reassembled text is not the reply: {_streamed_content(frames)!r}")
        sent = fakes.provider.recorder.all("chat")[chats_before:]
        _check(len(sent) == 1, f"expected one provider call, got {len(sent)}")
        _check(sent[0].body.get("stream") is True, "the gateway did not ask the provider to stream")
        # Injected by the gateway, not by the caller: without it a streamed
        # attempt reports no tokens and settles at no cost.
        options = sent[0].body.get("stream_options") or {}
        _check(options.get("include_usage") is True, f"include_usage was not injected: {options!r}")

    report = fakes.control_plane.recorder.wait_for("usage", before + 1)[-1].body
    _check(report.get("status") == "success", f"the streamed attempt was not reported successful: {report!r}")
    ttft = report.get("ttft_ms")
    _check(isinstance(ttft, int) and ttft > 0, f"the streamed usage report carries no ttft_ms: {report!r}")
    log(f"A streamed completion delivered SSE, injected include_usage, and reported ttft_ms={ttft}")


def check_every_attempt_was_reported(fakes: Fakes) -> None:
    """Exactly one usage report per dispatched attempt, and none for anything else.

    Keyed on the ``X-Correlation-ID`` each served response carried, which is
    the attempt the report must name. A request the platform refused, or one
    refused at tool admission after its credentials resolved, dispatched
    nothing and owes no report.
    """
    expected = fakes.dispatched
    _check(len(set(expected)) == len(expected), f"two served responses shared an attempt id: {expected!r}")

    def reported() -> list[str]:
        return [
            str(item.body.get("correlation_id"))
            for item in fakes.control_plane.recorder.all("usage")
            if isinstance(item.body, dict)
        ]

    # A streamed attempt reports only once its stream is fully closed, which is
    # after the caller has its last frame, so this waits on the set rather than
    # on a count and says which attempt is missing when it gives up.
    deadline = time.monotonic() + REPORT_TIMEOUT_SECONDS
    while sorted(reported()) != sorted(expected) and time.monotonic() < deadline:
        time.sleep(0.2)
    time.sleep(0.5)  # a report for something that was never dispatched would arrive late
    missing = [attempt for attempt in expected if attempt not in reported()]
    unexpected = [attempt for attempt in reported() if attempt not in expected]
    _check(
        not missing and not unexpected,
        f"usage reports do not match what was dispatched: missing {missing}, unexpected {unexpected}",
    )
    log(f"Every one of the {len(expected)} dispatched attempts was reported back, and nothing else was")


STEPS: tuple[Callable[[str, Fakes], None], ...] = (
    lambda base_url, fakes: check_health_and_edition(base_url),
    run_completion,
    check_platform_refusals,
    run_web_search,
    check_web_fetch_is_off_by_default,
    run_mcp,
    run_native_code_execution,
    run_native_web_search,
    run_streaming_completion,
    lambda base_url, fakes: check_every_attempt_was_reported(fakes),
)


def smoke(base_url: str, fakes: Fakes) -> None:
    for step in STEPS:
        step(base_url, fakes)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--image",
        default=None,
        help="Run this container image instead of the source checkout's CLI, as a deployment does.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Resolve attempts to the real OpenAI and Anthropic APIs and search through Tavily, "
        "with keys from OTARI_SMOKE_*_API_KEY.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    live = LiveProviders.from_env(dict(os.environ)) if args.live else None
    # A container reaches the fakes from its own namespace, so they cannot bind loopback.
    bind_host = ALL_INTERFACES if args.image else LOOPBACK
    with tempfile.TemporaryDirectory(prefix="otari-hybrid-smoke-") as workdir_name:
        workdir = Path(workdir_name)
        config_path = workdir / "hybrid.yml"
        log_path = workdir / "gateway.log"
        port = _free_port()
        base_url = f"http://127.0.0.1:{port}"
        try:
            with (
                serve(MockProvider(bind_host), "mock-provider") as provider,
                serve(FakeMcpServer(bind_host), "fake-mcp") as mcp,
            ):
                state = ControlPlaneState(provider_base_url=provider.base_url, mcp_url=mcp.mcp_url, live=live)
                with serve(FakeControlPlane(state, bind_host), "fake-control-plane") as control_plane:
                    fakes = Fakes(control_plane=control_plane, provider=provider, mcp=mcp, live=live)
                    platform_base_url = f"{control_plane.base_url}{PLATFORM_PREFIX}"
                    config = hybrid_config(
                        port=port,
                        platform_base_url=platform_base_url,
                        tavily_key=live.tavily_key if live else None,
                        in_container=bool(args.image),
                    )
                    write_config(config_path, config)
                    # World-readable: the container runs as its own user and has
                    # to read the mount. The file holds this run's fixtures.
                    config_path.chmod(0o644)
                    if args.image:
                        run_gateway = gateway_container(args.image, config_path, port, base_url, log_path)
                    else:
                        run_gateway = gateway(config_path, hybrid_env(dict(os.environ)), base_url, log_path)
                    with run_gateway:
                        smoke(base_url, fakes)
        except SmokeFailure as failure:
            log(f"\nHybrid smoke FAILED: {failure}\n\n{_tail(log_path)}")
            return 1
        except Exception as error:
            log(f"\nHybrid smoke FAILED: {type(error).__name__}: {error}\n\n{_tail(log_path)}")
            return 1
    where = []
    if args.image:
        where.append("in the container")
    if live:
        where.append("against live providers")
    log("\nHybrid smoke passed" + (f" {' '.join(where)}." if where else "."))
    return 0


if __name__ == "__main__":
    sys.exit(main())
