"""A fake OpenAI-compatible provider for the load test.

Each upstream model is a path prefix (``/m1/v1``, ``/m2/v1``, ``/m3/v1``), so one
process stands in for Model 1, Model 2 and the last-resort fallback. Every slot
answers after a set latency, streams, and can be told at runtime to throttle
(a quota in requests per minute, like a real provider's), to return 429s, or to
fail a stream after its first chunk.

Control plane (not reachable through the gateway):

    GET  /_stats                     what each slot accepted and refused
    POST /_reset                     zero the counters
    POST /_control {"m1": {...}}     change a slot's settings ("*" for all)

Slot settings: latency_ms, ttft_ms, chunks, chunk_interval_ms,
completion_tokens, quota_rpm (null for none), error_429_rate, stream_fail_rate.

Standard library only, so it runs in a bare python image.
"""

from __future__ import annotations

import json
import os
import random
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

SLOTS = ("m1", "m2", "m3")


def _default_settings(slot: str) -> dict[str, Any]:
    quota = os.environ.get(f"FAKE_{slot.upper()}_QUOTA_RPM", "")
    return {
        "latency_ms": int(os.environ.get("FAKE_LATENCY_MS", "300")),
        "ttft_ms": int(os.environ.get("FAKE_TTFT_MS", "200")),
        "chunks": int(os.environ.get("FAKE_CHUNKS", "20")),
        "chunk_interval_ms": int(os.environ.get("FAKE_CHUNK_INTERVAL_MS", "20")),
        "completion_tokens": int(os.environ.get("FAKE_COMPLETION_TOKENS", "200")),
        "quota_rpm": int(quota) if quota else None,
        "error_429_rate": 0.0,
        "stream_fail_rate": 0.0,
    }


class SlotState:
    """Settings and counters for one upstream model."""

    def __init__(self, slot: str) -> None:
        self.slot = slot
        self.settings = _default_settings(slot)
        self.reset()

    def reset(self) -> None:
        self.accepted_times: deque[float] = deque()
        self.accepted_total = 0
        self.max_accepted_per_60s = 0
        self.throttled_quota = 0
        self.throttled_injected = 0
        self.streams_failed = 0


class FakeServer(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 1024

    def __init__(self, address: tuple[str, int]) -> None:
        super().__init__(address, Handler)
        self.lock = threading.Lock()
        self.slots = {slot: SlotState(slot) for slot in SLOTS}

    def admit(self, state: SlotState) -> str | None:
        """Decide whether a call is served; return the 429 reason when it is not."""
        now = time.time()
        with self.lock:
            window = state.accepted_times
            while window and window[0] <= now - 60:
                window.popleft()
            if random.random() < state.settings["error_429_rate"]:
                state.throttled_injected += 1
                return "injected"
            quota = state.settings["quota_rpm"]
            if quota is not None and len(window) >= quota:
                state.throttled_quota += 1
                return "quota"
            window.append(now)
            state.accepted_total += 1
            state.max_accepted_per_60s = max(state.max_accepted_per_60s, len(window))
            return None

    def stats(self) -> dict[str, Any]:
        with self.lock:
            return {
                slot: {
                    "accepted_total": state.accepted_total,
                    "max_accepted_per_60s": state.max_accepted_per_60s,
                    "throttled_quota": state.throttled_quota,
                    "throttled_injected": state.throttled_injected,
                    "streams_failed": state.streams_failed,
                }
                for slot, state in self.slots.items()
            }


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    # Headers and body go out as separate writes; with Nagle on, the body waits
    # for a delayed ACK and every response gains 40ms that is not the gateway's.
    disable_nagle_algorithm = True
    server: FakeServer

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        """Quiet: at 1,000 RPM an access log is noise."""

    # GET and POST are http.server's spellings.
    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/_stats":
            self._json(200, self.server.stats())
        elif self.path == "/healthz":
            self._json(200, {"ok": True})
        elif self.path.endswith("/models"):
            slot = self.path.strip("/").split("/")[0]
            self._json(200, {"object": "list", "data": [{"id": f"{slot}-model", "object": "model"}]})
        else:
            self._json(404, {"error": {"message": f"no route {self.path}"}})

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        if self.path == "/_reset":
            with self.server.lock:
                for state in self.server.slots.values():
                    state.reset()
            self._json(200, {"ok": True})
            return
        if self.path == "/_control":
            with self.server.lock:
                for slot, settings in json.loads(body or b"{}").items():
                    targets = self.server.slots.values() if slot == "*" else [self.server.slots[slot]]
                    for state in targets:
                        state.settings.update(settings)
            self._json(200, self.server.stats())
            return

        slot = self.path.strip("/").split("/")[0]
        state = self.server.slots.get(slot)
        if state is None or not self.path.endswith("/chat/completions"):
            self._json(404, {"error": {"message": f"no route {self.path}"}})
            return

        request = json.loads(body or b"{}")
        refused = self.server.admit(state)
        if refused is not None:
            self._json(
                429,
                {"error": {"message": f"{slot} rate limited ({refused})", "type": "rate_limit_exceeded"}},
                extra_headers={"Retry-After": "1"},
            )
            return
        prompt_tokens = max(1, len(json.dumps(request.get("messages", []))) // 4)
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": state.settings["completion_tokens"],
            "total_tokens": prompt_tokens + state.settings["completion_tokens"],
        }
        base = {
            "id": f"chatcmpl-{slot}-{random.getrandbits(48):x}",
            "created": int(time.time()),
            "model": request.get("model", f"{slot}-model"),
        }
        if request.get("stream"):
            self._stream(state, base, usage)
            return
        time.sleep(state.settings["latency_ms"] / 1000)
        message = {"role": "assistant", "content": f"served-by:{slot}"}
        choice = {"index": 0, "message": message, "finish_reason": "stop"}
        self._json(200, {**base, "object": "chat.completion", "choices": [choice], "usage": usage})

    def _stream(self, state: SlotState, base: dict[str, Any], usage: dict[str, int]) -> None:
        settings = state.settings
        fail = random.random() < settings["stream_fail_rate"]
        time.sleep(settings["ttft_ms"] / 1000)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        base = {**base, "object": "chat.completion.chunk"}
        for index in range(max(1, settings["chunks"])):
            delta = {"role": "assistant", "content": f"served-by:{state.slot}"} if index == 0 else {"content": " tok"}
            self._chunk({**base, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]})
            if fail:
                # Break the chunked encoding mid-body: the gateway's client sees
                # an incomplete read, which is what a dropped upstream looks like.
                with self.server.lock:
                    state.streams_failed += 1
                self.close_connection = True
                self.wfile.flush()
                self.connection.shutdown(2)
                return
            time.sleep(settings["chunk_interval_ms"] / 1000)
        self._chunk({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
        self._chunk({**base, "choices": [], "usage": usage})
        self._raw_chunk(b"data: [DONE]\n\n")
        self._raw_chunk(b"")

    def _chunk(self, payload: dict[str, Any]) -> None:
        self._raw_chunk(f"data: {json.dumps(payload)}\n\n".encode())

    def _raw_chunk(self, data: bytes) -> None:
        self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
        self.wfile.flush()

    def _json(self, status: int, payload: Any, extra_headers: dict[str, str] | None = None) -> None:
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        for name, value in (extra_headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(data)


def main() -> None:
    port = int(os.environ.get("FAKE_PORT", "9000"))
    server = FakeServer(("0.0.0.0", port))
    print(f"fake provider on :{port}, slots {', '.join(SLOTS)}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
