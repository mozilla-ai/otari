"""The provider the path-parity tests point a provider instance at, reached over real HTTP.

It runs in one of two modes:

- Scripted, with no ``upstream``: it answers chat completions itself. With the
  gateway's ``web_search`` function offered and no tool result in the
  conversation yet, it calls that function once; otherwise it answers with text.
  The round after ``n`` tool results reports ``ROUND_USAGE[n]``, so a total that
  leaves out a round, or counts one twice, cannot come out right by accident.
- Recording, with an ``upstream`` root: it forwards each call there unchanged
  and passes the answer back, streamed or not.

Either way it records the tokens each call reported in ``rounds``, which is what
the gateway's totals are checked against.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

REPLY = "The parity provider answered."
SEARCH_FUNCTION = "web_search"
# (prompt_tokens, completion_tokens) per round.
ROUND_USAGE: tuple[tuple[int, int], ...] = ((1000, 10), (2000, 20))
# Headers that describe this hop rather than the call, so they are not forwarded.
_HOP_HEADERS = frozenset({"host", "content-length", "connection", "accept-encoding", "transfer-encoding"})
# The events after which a Responses or Messages stream sends nothing more; a chat stream ends at [DONE].
_TERMINAL_EVENTS = frozenset({"response.completed", "response.incomplete", "response.failed", "message_stop"})


class ParityProvider(ThreadingHTTPServer):
    """The HTTP server, the bodies it was sent, and the tokens each call reported."""

    daemon_threads = True

    def __init__(self, upstream: str | None = None) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.upstream = upstream.rstrip("/") if upstream else None
        self.requests: list[dict[str, Any]] = []
        self.rounds: list[tuple[int, int]] = []
        self._lock = threading.Lock()

    @property
    def root(self) -> str:
        host, port = self.server_address[:2]
        return f"http://{host!s}:{port}"

    def record(self, body: dict[str, Any] | None = None, usage: tuple[int, int] | None = None) -> None:
        with self._lock:
            if body is not None:
                self.requests.append(body)
            if usage is not None:
                self.rounds.append(usage)

    def reset(self) -> None:
        with self._lock:
            self.requests.clear()
            self.rounds.clear()

    def usage(self) -> tuple[int, int]:
        """The prompt and completion tokens every call reported, summed."""
        with self._lock:
            return sum(p for p, _ in self.rounds), sum(c for _, c in self.rounds)


def _reported(payload: Any, so_far: tuple[int, int] = (0, 0)) -> tuple[int, int]:
    """The tokens reported so far, updated by one body or stream event of any of the three API shapes.

    A count an event leaves out, or reports as 0, keeps its earlier value, since an
    Anthropic stream reports input tokens on ``message_start`` and output tokens on
    the cumulative ``message_delta``.
    """
    if not isinstance(payload, dict):
        return so_far
    holder = payload.get("response") or payload.get("message") or payload
    usage = holder.get("usage") if isinstance(holder, dict) else None
    if not isinstance(usage, dict):
        return so_far
    prompt = usage.get("prompt_tokens") or usage.get("input_tokens") or so_far[0]
    completion = usage.get("completion_tokens") or usage.get("output_tokens") or so_far[1]
    return int(prompt), int(completion)


def _round(body: dict[str, Any]) -> int:
    return sum(1 for m in body.get("messages") or [] if m.get("role") == "tool")


def _scripted(body: dict[str, Any]) -> tuple[dict[str, Any], Iterator[dict[str, Any]]]:
    """The completion for ``body``, and the chunks that stream it."""
    offered = {t["function"]["name"] for t in body.get("tools") or [] if t.get("type") == "function"}
    delta: dict[str, Any]
    if SEARCH_FUNCTION in offered and _round(body) == 0:
        function = {"name": SEARCH_FUNCTION, "arguments": json.dumps({"query": "otari"})}
        call: dict[str, Any] = {"id": "call_parity_0", "type": "function", "function": function}
        message: dict[str, Any] = {"role": "assistant", "content": None, "tool_calls": [call]}
        finish, delta = "tool_calls", {"tool_calls": [{"index": 0, **call}]}
    else:
        message = {"role": "assistant", "content": REPLY}
        finish, delta = "stop", {"content": REPLY}
    prompt, completion = ROUND_USAGE[_round(body)]
    usage = {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion}
    base = {"id": f"chatcmpl-parity-{_round(body)}", "created": 1_700_000_000, "model": body.get("model")}
    whole = {
        **base,
        "object": "chat.completion",
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": usage,
    }
    chunk = {**base, "object": "chat.completion.chunk"}
    chunks = [
        {**chunk, "choices": [{"index": 0, "delta": {"role": "assistant", **delta}, "finish_reason": None}]},
        {**chunk, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]},
    ]
    if (body.get("stream_options") or {}).get("include_usage"):
        chunks.append({**chunk, "choices": [], "usage": usage})
    return whole, iter(chunks)


class _Handler(BaseHTTPRequestHandler):
    server: ParityProvider

    def do_POST(self) -> None:  # noqa: N802
        raw = self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}"
        body = json.loads(raw)
        self.server.record(body)
        if self.server.upstream is not None:
            self._forward(raw)
            return
        if not self.path.endswith("/chat/completions"):
            self._send(404, "application/json", json.dumps({"error": {"message": f"no route {self.path}"}}).encode())
            return
        whole, chunks = _scripted(body)
        self.server.record(usage=_reported(whole))
        if not body.get("stream"):
            self._send(200, "application/json", json.dumps(whole).encode())
            return
        self._send(200, "text/event-stream")
        for chunk in chunks:
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")

    def _forward(self, raw: bytes) -> None:
        request = urllib.request.Request(f"{self.server.upstream}{self.path}", data=raw, method="POST")
        for name, value in self.headers.items():
            if name.lower() not in _HOP_HEADERS:
                request.add_header(name, value)
        try:
            upstream = urllib.request.urlopen(request, timeout=120)  # noqa: S310 - the test's own provider root
        except urllib.error.HTTPError as error:
            self._send(error.code, error.headers.get("Content-Type", "application/json"), error.read())
            return
        with upstream:
            content_type = upstream.headers.get("Content-Type", "application/json")
            if "text/event-stream" not in content_type:
                payload = upstream.read()
                self.server.record(usage=_reported(json.loads(payload or b"{}")))
                self._send(upstream.status, content_type, payload)
                return
            self._send(upstream.status, content_type)
            usage, recorded = (0, 0), False
            for line in upstream:
                data = line.decode(errors="replace").strip().removeprefix("data:").strip()
                event = json.loads(data) if data.startswith("{") else None
                usage = _reported(event, usage)
                # A client stops reading at the event that ends the stream, so the round
                # is recorded before that event reaches it rather than at EOF.
                if not recorded and (data == "[DONE]" or (event or {}).get("type") in _TERMINAL_EVENTS):
                    self.server.record(usage=usage)
                    recorded = True
                self.wfile.write(line)
                self.wfile.flush()
            if not recorded:
                self.server.record(usage=usage)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        """Keep the test output quiet."""

    def _send(self, status: int, content_type: str, data: bytes | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        if data is not None:
            self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        if data is not None:
            self.wfile.write(data)


@contextmanager
def running(upstream: str | None = None) -> Iterator[ParityProvider]:
    """Serve the provider on a background thread for the duration of the block."""
    server = ParityProvider(upstream)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
