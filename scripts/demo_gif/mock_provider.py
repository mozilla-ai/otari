"""A stand-in OpenAI-compatible upstream for the README dashboard GIF.

Serves ``GET /v1/models`` and a streamed ``POST /v1/chat/completions`` on
127.0.0.1:8099 (the ``api_base`` otari.yml gives it), so the Playground stop in
tour.mjs has a reply to stream. The reply is canned per model and paced like a
real one, and the final chunk carries usage so the turn readout shows tokens and
cost.

Standard library only. Run by scripts/demo_gif/record.sh.
"""

import json
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

PORT = 8099

MODELS = ["gpt-6.1-sol", "gpt-6-luna", "gpt-6-astra", "deepseek-v4-pro", "deepseek-v4-flash"]

# Two distinct answers to the tour's prompt, so the side-by-side comparison
# reads as two models rather than one echoed twice.
REPLIES = {
    "gpt": (
        "Use **exponential backoff with full jitter**, capped, and only on errors that can succeed later.\n\n"
        "```python\n"
        "delay = random.uniform(0, min(cap, base * 2 ** attempt))\n"
        "```\n\n"
        "- Retry `429`, `503` and timeouts; never `400` or `401`.\n"
        "- Honor `Retry-After` when the provider sends it.\n"
        "- Give up after ~5 attempts and surface the last error."
    ),
    "deepseek": (
        "Short answer: **jittered backoff plus a retry budget**.\n\n"
        "1. Back off `base * 2^n`, randomized so clients don't retry in lockstep.\n"
        "2. Cap retries at ~10% of traffic, so an outage doesn't triple your load.\n"
        "3. Make writes idempotent first; otherwise a retry can double-charge.\n\n"
        "With a gateway in front, a fallback model usually beats a fourth retry."
    ),
}


def reply_for(model: str) -> str:
    return REPLIES["deepseek" if model.startswith("deepseek") else "gpt"]


def chunks(text: str) -> list[str]:
    """Split into word-sized pieces, keeping whitespace, like a token stream."""
    return re.findall(r"\S+\s*|\s+", text)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_args: object) -> None:
        pass

    def _json(self, status: int, body: dict[str, Any]) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/").endswith("/models"):
            self._json(
                200,
                {
                    "object": "list",
                    "data": [{"id": m, "object": "model", "created": 0, "owned_by": "demo"} for m in MODELS],
                },
            )
        else:
            self._json(404, {"error": {"message": "not found"}})

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        if not self.path.rstrip("/").endswith("/chat/completions"):
            self._json(404, {"error": {"message": "not found"}})
            return
        model = str(body.get("model", "gpt"))
        text = reply_for(model)
        prompt_tokens = 38 + sum(len(str(m.get("content", ""))) // 4 for m in body.get("messages", []))
        completion_tokens = len(text) // 4
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        base = {"id": "chatcmpl-demo", "created": int(time.time()), "model": model}

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        def send(payload: dict[str, Any] | str) -> None:
            data = (
                payload
                if isinstance(payload, str)
                else json.dumps({**base, "object": "chat.completion.chunk", **payload})
            )
            self.wfile.write(f"data: {data}\n\n".encode())
            self.wfile.flush()

        # The two comparison models stream at different speeds, which is the point
        # of comparing them.
        pause = 0.022 if model.startswith("deepseek") else 0.034
        time.sleep(0.35 if model.startswith("deepseek") else 0.5)
        send({"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]})
        for piece in chunks(text):
            send({"choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}]})
            time.sleep(pause)
        send({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
        send({"choices": [], "usage": usage})
        send("[DONE]")
        self.close_connection = True


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
