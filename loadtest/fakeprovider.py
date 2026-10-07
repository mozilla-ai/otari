"""A fake OpenAI-compatible provider: answers after a fixed latency, streamed or not.

Each upstream model is a path prefix (``/m1/v1``, ``/m2/v1``, ``/m3/v1``), so one
process stands in for Model 1, Model 2 and the last-resort fallback. The reply
names the slot that served it (``served-by:m2``), so the client can tell a
spilled request from a direct one. Its timing comes from FAKE_LATENCY_MS,
FAKE_TTFT_MS, FAKE_CHUNKS and FAKE_CHUNK_INTERVAL_MS.

Standard library only, so it runs in a bare python image.
"""

from __future__ import annotations

import json
import os
import random
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

LATENCY_MS = int(os.environ.get("FAKE_LATENCY_MS", "50"))
TTFT_MS = int(os.environ.get("FAKE_TTFT_MS", "50"))
CHUNKS = int(os.environ.get("FAKE_CHUNKS", "5"))
CHUNK_INTERVAL_MS = int(os.environ.get("FAKE_CHUNK_INTERVAL_MS", "10"))
COMPLETION_TOKENS = 200
SLOTS = ("m1", "m2", "m3")


class Server(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 1024


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    # Headers and body go out as separate writes; with Nagle on, the body waits
    # for a delayed ACK and every response gains 40ms that is not the gateway's.
    disable_nagle_algorithm = True

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        """Quiet: at 1,000 requests a minute an access log is noise."""

    # GET and POST are http.server's spellings.
    def do_GET(self) -> None:  # noqa: N802
        slot = self.path.strip("/").split("/")[0]
        if self.path == "/healthz":
            self._json(200, {"ok": True})
        elif slot in SLOTS and self.path.endswith("/models"):
            self._json(200, {"object": "list", "data": [{"id": f"{slot}-model", "object": "model"}]})
        else:
            self._json(404, {"error": {"message": f"no route {self.path}"}})

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        request = json.loads(self.rfile.read(length) or b"{}") if length else {}
        slot = self.path.strip("/").split("/")[0]
        if slot not in SLOTS or not self.path.endswith("/chat/completions"):
            self._json(404, {"error": {"message": f"no route {self.path}"}})
            return
        prompt_tokens = max(1, len(json.dumps(request.get("messages", []))) // 4)
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": COMPLETION_TOKENS,
            "total_tokens": prompt_tokens + COMPLETION_TOKENS,
        }
        base = {
            "id": f"chatcmpl-{slot}-{random.getrandbits(48):x}",
            "created": int(time.time()),
            "model": request.get("model", f"{slot}-model"),
        }
        if request.get("stream"):
            self._stream(slot, base, usage)
            return
        time.sleep(LATENCY_MS / 1000)
        message = {"role": "assistant", "content": f"served-by:{slot}"}
        choice = {"index": 0, "message": message, "finish_reason": "stop"}
        self._json(200, {**base, "object": "chat.completion", "choices": [choice], "usage": usage})

    def _stream(self, slot: str, base: dict[str, Any], usage: dict[str, int]) -> None:
        time.sleep(TTFT_MS / 1000)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        base = {**base, "object": "chat.completion.chunk"}
        for index in range(CHUNKS):
            delta = {"role": "assistant", "content": f"served-by:{slot}"} if index == 0 else {"content": " tok"}
            self._chunk({**base, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]})
            time.sleep(CHUNK_INTERVAL_MS / 1000)
        self._chunk({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
        self._chunk({**base, "choices": [], "usage": usage})
        self._raw_chunk(b"data: [DONE]\n\n")
        self._raw_chunk(b"")

    def _chunk(self, payload: dict[str, Any]) -> None:
        self._raw_chunk(f"data: {json.dumps(payload)}\n\n".encode())

    def _raw_chunk(self, data: bytes) -> None:
        self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
        self.wfile.flush()

    def _json(self, status: int, payload: Any) -> None:
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


if __name__ == "__main__":
    print("fake provider on :9000", flush=True)
    Server(("0.0.0.0", 9000), Handler).serve_forever()
