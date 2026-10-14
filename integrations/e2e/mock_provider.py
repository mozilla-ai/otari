"""A scripted OpenAI-compatible provider for the framework end-to-end tests.

It answers ``POST /v1/chat/completions`` the way a real model would answer the
agents the docs pages build, deterministically:

- With tools offered and no tool result in the conversation yet, it calls the
  first offered tool that ``TOOL_ARGUMENTS`` has arguments for.
- With a JSON schema ``response_format``, it answers with ``STRUCTURED_OUTPUT``.
- Otherwise it answers with text, echoing the last tool result when there is one,
  so a test can tell that the tool's output reached the model.

Streamed requests get the same answer as server-sent events. Every request body is
recorded so a test can assert what the framework actually sent through Otari.
"""

from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

REPLY = "Hello from the mock provider behind Otari."
TOOL_ARGUMENTS: dict[str, dict[str, Any]] = {
    "get_weather": {"city": "Paris"},
    # Pydantic AI delivers a structured output through a tool of this name.
    "final_result": {"city": "Paris", "temperature_c": 21},
}
STRUCTURED_OUTPUT: dict[str, Any] = {"city": "Paris", "temperature_c": 21}
USAGE = {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}


class MockProvider(ThreadingHTTPServer):
    """The HTTP server, plus every request body it was sent."""

    daemon_threads = True

    def __init__(self) -> None:
        """Listen on a free port on localhost."""
        super().__init__(("127.0.0.1", 0), _Handler)
        self.requests: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    @property
    def base_url(self) -> str:
        """The API root a provider instance is configured with."""
        host, port = self.server_address[:2]
        return f"http://{host!s}:{port}/v1"

    def record(self, body: dict[str, Any]) -> None:
        """Keep one request body for a test to inspect."""
        with self._lock:
            self.requests.append(body)


def answer(body: dict[str, Any]) -> dict[str, Any]:
    """Return the assistant message for one request, as a chat completion message."""
    messages = body.get("messages") or []
    tool_results = [m for m in messages if m.get("role") == "tool"]
    offered = [t["function"]["name"] for t in body.get("tools") or [] if t.get("type") == "function"]

    # A structured-output tool ends the run, so it is called even after a tool result.
    pending = [name for name in offered if name in TOOL_ARGUMENTS and (not tool_results or name == "final_result")]
    if pending:
        name = pending[0]
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": f"call_{len(tool_results)}",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(TOOL_ARGUMENTS[name])},
                }
            ],
        }

    response_format = body.get("response_format") or {}
    if response_format.get("type") == "json_schema":
        return {"role": "assistant", "content": json.dumps(STRUCTURED_OUTPUT)}

    if tool_results:
        return {"role": "assistant", "content": f"The tool said: {tool_results[-1].get('content')}"}
    return {"role": "assistant", "content": REPLY}


def _completion(body: dict[str, Any], message: dict[str, Any]) -> dict[str, Any]:
    """Wrap a message in a chat completion, shaped as the OpenAI SDK expects."""
    return {
        "id": "chatcmpl-framework-e2e",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.get("model"),
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if message.get("tool_calls") else "stop",
            }
        ],
        "usage": USAGE,
    }


def _chunks(body: dict[str, Any], message: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Yield the same message as streamed chat completion chunks."""
    base = {
        "id": "chatcmpl-framework-e2e",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": body.get("model"),
    }
    yield {**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]}
    if message.get("tool_calls"):
        calls = [{"index": i, **call} for i, call in enumerate(message["tool_calls"])]
        yield {**base, "choices": [{"index": 0, "delta": {"tool_calls": calls}, "finish_reason": None}]}
        finish = "tool_calls"
    else:
        # Word by word, so a test can tell streaming from one buffered chunk.
        for word in message["content"].split(" "):
            delta = {"content": word + " "}
            yield {**base, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}
        finish = "stop"
    yield {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]}
    if (body.get("stream_options") or {}).get("include_usage"):
        yield {**base, "choices": [], "usage": USAGE}


class _Handler(BaseHTTPRequestHandler):
    """Answer ``POST .../chat/completions`` with what ``answer`` decides."""

    protocol_version = "HTTP/1.1"
    server: MockProvider

    def do_POST(self) -> None:  # noqa: N802
        """Record the request and answer it, streamed when it asked to be."""
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        if not self.path.endswith("/chat/completions"):
            self._json(404, {"error": {"message": f"mock provider has no route {self.path}"}})
            return
        self.server.record(body)
        message = answer(body)
        if not body.get("stream"):
            self._json(200, _completion(body, message))
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        for chunk in _chunks(body, message):
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()
        self.close_connection = True

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        """Stay quiet: the gateway's log is the useful one on a failure."""

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        """Send a JSON response with its length, so the connection stays reusable."""
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@contextmanager
def running() -> Iterator[MockProvider]:
    """Run the mock provider on a background thread for the duration of the block."""
    server = MockProvider()
    thread = threading.Thread(target=server.serve_forever, name="mock-provider", daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
