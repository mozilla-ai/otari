"""The provider the path-parity tests point a provider instance at, reached over real HTTP.

A request to any of the gateway's inference endpoints runs the gateway's own
dispatch, the any-llm provider and its conversions, and arrives here. It runs in
one of two modes:

- Scripted, with no ``upstream``: it answers chat completions itself. With the
  gateway's ``web_search`` function offered and no tool result in the
  conversation yet, it calls that function once; otherwise it answers with text.
  Each round reports ``ROUND_USAGE[n]`` for the round that follows ``n`` tool
  results, so a total that leaves out a round, or counts one twice, cannot come
  out right by accident.
- Recording, with an ``upstream`` root: it forwards each call to that provider
  unchanged and passes the answer back, streamed or not.

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
# Request headers a provider authenticates or versions a call with, forwarded upstream.
_FORWARDED_HEADERS = ("Authorization", "x-api-key", "anthropic-version", "anthropic-beta", "Content-Type", "Accept")


class ParityProvider(ThreadingHTTPServer):
    """The HTTP server, the bodies it was sent, and the tokens each call reported."""

    daemon_threads = True

    def __init__(self, upstream: str | None = None) -> None:
        """Listen on a free port on localhost, forwarding to ``upstream`` when one is given."""
        super().__init__(("127.0.0.1", 0), _Handler)
        self.upstream = upstream.rstrip("/") if upstream else None
        self.requests: list[dict[str, Any]] = []
        self.rounds: list[tuple[int, int]] = []
        self._lock = threading.Lock()

    @property
    def root(self) -> str:
        """The address the server listens on, with no path."""
        host, port = self.server_address[:2]
        return f"http://{host!s}:{port}"

    def record(self, body: dict[str, Any]) -> None:
        """Keep one request body for a test to inspect."""
        with self._lock:
            self.requests.append(body)

    def record_round(self, usage: tuple[int, int]) -> None:
        """Keep the prompt and completion tokens one call reported."""
        with self._lock:
            self.rounds.append(usage)

    def reset(self) -> None:
        """Forget what was seen so far."""
        with self._lock:
            self.requests.clear()
            self.rounds.clear()

    def usage(self) -> tuple[int, int]:
        """The prompt and completion tokens every call reported, summed."""
        with self._lock:
            return sum(p for p, _ in self.rounds), sum(c for _, c in self.rounds)


class _UsageReader:
    """The tokens one call reports, read from a JSON body or from each event of a stream.

    It reads the OpenAI chat shape (``prompt_tokens``), the Responses and Messages
    shapes (``input_tokens``), and the places a stream carries them: a final usage
    chunk, ``response.completed``, or Anthropic's ``message_start`` and cumulative
    ``message_delta``.
    """

    def __init__(self) -> None:
        self.prompt = 0
        self.completion = 0

    def feed(self, payload: Any) -> None:
        if not isinstance(payload, dict):
            return
        for holder in (payload, payload.get("response"), payload.get("message")):
            usage = holder.get("usage") if isinstance(holder, dict) else None
            if isinstance(usage, dict):
                self.prompt = usage.get("prompt_tokens") or usage.get("input_tokens") or self.prompt
                self.completion = usage.get("completion_tokens") or usage.get("output_tokens") or self.completion


# The events after which a Responses or Messages stream sends nothing more.
_TERMINAL_EVENTS = frozenset({"response.completed", "response.incomplete", "response.failed", "message_stop"})


def _ends_stream(event: Any) -> bool:
    return isinstance(event, dict) and event.get("type") in _TERMINAL_EVENTS


def _round(body: dict[str, Any]) -> int:
    return sum(1 for m in body.get("messages") or [] if m.get("role") == "tool")


def _answer(body: dict[str, Any]) -> dict[str, Any]:
    offered = {t["function"]["name"] for t in body.get("tools") or [] if t.get("type") == "function"}
    if SEARCH_FUNCTION in offered and _round(body) == 0:
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_parity_0",
                    "type": "function",
                    "function": {"name": SEARCH_FUNCTION, "arguments": json.dumps({"query": "otari"})},
                }
            ],
        }
    return {"role": "assistant", "content": REPLY}


def _usage(body: dict[str, Any]) -> dict[str, int]:
    prompt, completion = ROUND_USAGE[_round(body)]
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion}


def _finish_reason(message: dict[str, Any]) -> str:
    return "tool_calls" if message.get("tool_calls") else "stop"


def _completion(body: dict[str, Any], message: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": f"chatcmpl-parity-{_round(body)}",
        "object": "chat.completion",
        "created": 1_700_000_000,
        "model": body.get("model", "parity-model"),
        "choices": [{"index": 0, "message": message, "finish_reason": _finish_reason(message)}],
        "usage": _usage(body),
    }


def _chunks(body: dict[str, Any], message: dict[str, Any]) -> Iterator[dict[str, Any]]:
    base = {
        "id": f"chatcmpl-parity-{_round(body)}",
        "object": "chat.completion.chunk",
        "created": 1_700_000_000,
        "model": body.get("model", "parity-model"),
    }
    yield {**base, "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]}
    if message.get("tool_calls"):
        for index, call in enumerate(message["tool_calls"]):
            delta = {"tool_calls": [{"index": index, **call}]}
            yield {**base, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}
    else:
        yield {**base, "choices": [{"index": 0, "delta": {"content": message["content"]}, "finish_reason": None}]}
    yield {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": _finish_reason(message)}]}
    if (body.get("stream_options") or {}).get("include_usage"):
        yield {**base, "choices": [], "usage": _usage(body)}


class _Handler(BaseHTTPRequestHandler):
    server: ParityProvider

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) or b"{}"
        body = json.loads(raw)
        self.server.record(body)
        if self.server.upstream is not None:
            self._forward(raw)
        else:
            self._script(body)

    def _script(self, body: dict[str, Any]) -> None:
        if not self.path.endswith("/chat/completions"):
            self._json(404, {"error": {"message": f"the parity provider has no route {self.path}"}})
            return
        message = _answer(body)
        prompt, completion = ROUND_USAGE[_round(body)]
        self.server.record_round((prompt, completion))
        if not body.get("stream"):
            self._json(200, _completion(body, message))
            return
        self._open_stream(200, "text/event-stream")
        for chunk in _chunks(body, message):
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def _forward(self, raw: bytes) -> None:
        request = urllib.request.Request(f"{self.server.upstream}{self.path}", data=raw, method="POST")
        for name in _FORWARDED_HEADERS:
            if value := self.headers.get(name):
                request.add_header(name, value)
        reader = _UsageReader()
        try:
            upstream = urllib.request.urlopen(request, timeout=120)  # noqa: S310 - the upstream is the test's own provider root
        except urllib.error.HTTPError as error:
            payload = error.read()
            self._raw(error.code, error.headers.get("Content-Type", "application/json"), payload)
            return
        with upstream:
            content_type = upstream.headers.get("Content-Type", "application/json")
            if "text/event-stream" not in content_type:
                payload = upstream.read()
                reader.feed(json.loads(payload or b"{}"))
                self.server.record_round((reader.prompt, reader.completion))
                self._raw(upstream.status, content_type, payload)
                return
            self._open_stream(upstream.status, content_type)
            recorded = False
            for line in upstream:
                text = line.decode(errors="replace").strip()
                data = text.removeprefix("data:").strip() if text.startswith("data:") else ""
                event = json.loads(data) if data and data != "[DONE]" else None
                reader.feed(event)
                # A client stops reading at the event that ends the stream, so the
                # round is recorded before that event reaches it, not at EOF.
                if not recorded and (data == "[DONE]" or _ends_stream(event)):
                    self.server.record_round((reader.prompt, reader.completion))
                    recorded = True
                self.wfile.write(line)
                self.wfile.flush()
        if not recorded:
            self.server.record_round((reader.prompt, reader.completion))

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        """Keep the test output quiet."""

    def _open_stream(self, status: int, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.end_headers()

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        self._raw(status, "application/json", json.dumps(payload).encode())

    def _raw(self, status: int, content_type: str, data: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
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
