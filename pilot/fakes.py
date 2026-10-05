"""Stand-ins for the upstreams behind the pilot, standard library only.

One server, one path prefix per upstream, each OpenAI-compatible so Otari
reaches it as a ``provider_type: openai`` instance:

- ``/vertexai`` and ``/mistral``: chat completions, streamed or not, with usage.
  A model named ``*-429`` answers 429, the provider throttling Otari.
- ``/exa``: Exa's answer endpoint, which puts ``citations`` on the message.
- ``/liner``: a shim from OpenAI chat completions to Liner's quick-answer agent
  API (``answer`` plus ``references``), returning the references as
  ``citations``. With ``LINER_API_KEY`` set it calls Liner itself; otherwise it
  answers the way Liner does.
- ``/searx/search``: a SearXNG-style search backend for Otari's ``exa-search``
  tool (Otari only sends an Exa key over https, which a local fake cannot serve).

    python pilot/fakes.py --port 9100
"""

import argparse
import json
import os
import time
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

LINER_URL = "https://platform.liner.com/api/v1/agents/quick-answer"
CITATIONS = [
    {"url": "https://www.mozilla.org/firefox/", "title": "Firefox"},
    {"url": "https://en.wikipedia.org/wiki/Firefox", "title": "Firefox - Wikipedia"},
]
COUNTS: dict[str, int] = {}


def _usage(messages: list[dict], completion_tokens: int) -> dict:
    prompt_tokens = sum(len(str(m.get("content", "")).split()) for m in messages) + 5
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def _completion(model: str, content: str, messages: list[dict], **message_extras: object) -> dict:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content, **message_extras},
            }
        ],
        "usage": _usage(messages, len(content.split())),
    }


def _liner(messages: list[dict]) -> tuple[str, list[dict]]:
    """Liner's quick-answer agent, called as Liner documents it, or answered the way it does."""
    key = os.environ.get("LINER_API_KEY")
    if not key:
        return "Liner says Firefox is a web browser ((1)).", CITATIONS
    body = json.dumps(
        {"messages": [{"role": m["role"], "content": str(m["content"])} for m in messages], "stream": False}
    ).encode()
    request = urllib.request.Request(
        LINER_URL, data=body, headers={"x-api-key": key, "Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310  # fixed https URL
        data = json.load(response)
    references = [{"url": r.get("url"), "title": r.get("title")} for r in data.get("references", [])]
    return data.get("answer", ""), references


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: object) -> None:
        pass

    def _json(self, status: int, payload: object) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parts = urlsplit(self.path)
        if parts.path == "/counts":
            self._json(200, COUNTS)
            return
        if parts.path == "/searx/search":
            query = parse_qs(parts.query).get("q", [""])[0]
            COUNTS["search"] = COUNTS.get("search", 0) + 1
            self._json(
                200,
                {
                    "results": [
                        {
                            "url": c["url"],
                            "title": c["title"],
                            "content": f"About {query}",
                            "publishedDate": "2026-10-01T00:00:00",
                        }
                        for c in CITATIONS
                    ]
                },
            )
            return
        if parts.path.endswith("/models"):
            self._json(200, {"object": "list", "data": []})
            return
        self._json(404, {"error": {"message": "not found"}})

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        request = json.loads(self.rfile.read(length) or b"{}")
        upstream = urlsplit(self.path).path.strip("/").split("/")[0]
        if not self.path.endswith("/chat/completions"):
            self._json(404, {"error": {"message": "not found"}})
            return
        model = request.get("model", "")
        messages = request.get("messages", [])
        COUNTS[f"{upstream}:{model}"] = COUNTS.get(f"{upstream}:{model}", 0) + 1
        COUNTS["metadata_forwarded"] = COUNTS.get("metadata_forwarded", 0) + ("metadata" in request)
        if model.endswith("-429"):
            body = json.dumps({"error": {"message": "Resource exhausted", "type": "rate_limit_error"}}).encode()
            self.send_response(429)
            self.send_header("Content-Type", "application/json")
            self.send_header("Retry-After", "7")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        extras: dict[str, object] = {}
        if upstream == "exa":
            content, extras["citations"] = "Exa says Firefox is a web browser.", CITATIONS
        elif upstream == "liner":
            content, extras["citations"] = _liner(messages)
        else:
            content = f"Hello from {upstream} {model}. This is a pilot answer."
        if request.get("stream"):
            self._stream(model, content, messages, extras)
        else:
            self._json(200, _completion(model, content, messages, **extras))

    def _stream(self, model: str, content: str, messages: list[dict], extras: dict) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        base = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": model,
        }
        words = content.split(" ")
        for index, word in enumerate(words):
            delta: dict[str, object] = {"content": word + (" " if index < len(words) - 1 else "")}
            if index == 0:
                delta.update(role="assistant", **extras)
            chunk = {**base, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.flush()
            time.sleep(0.01)
        done = {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
        self.wfile.write(f"data: {json.dumps(done)}\n\n".encode())
        usage = {**base, "choices": [], "usage": _usage(messages, len(words))}
        self.wfile.write(f"data: {json.dumps(usage)}\n\ndata: [DONE]\n\n".encode())
        self.wfile.flush()
        self.close_connection = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9100)
    port = parser.parse_args().port
    print(f"fakes listening on 127.0.0.1:{port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
