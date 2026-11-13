"""A visual demo of agent memory: one page, the same question with memory and without, side by side.

    python3 server.py            then open http://127.0.0.1:8020
    python3 server.py --seed     first put the demo's memory in place (erases the workspace's memory)

The page never sees a key: this server calls the local gateway with Alice's and Bob's keys from keys.env and
the master key from .env, and hands the page only answers, counts and entry texts.
"""

import json
import pathlib
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "bench"))
from bench import KEYS, WS, admin, http, last_read, psql, switches, tool  # noqa: E402

MODEL = "ollama:qwen2.5:3b"
PORT = 8020
# What the demo's story needs beyond what prepare.sh seeds.
SEEDS = [
    ("decision", "workspace", "Meeting with Alejandro on Nov 10: we agreed shared agent memory goes in the gateway as a core feature, with per-workspace keys and a consolidator that decides add, update or conflict."),
    ("fact", "workspace", "Alejandro likes painting with watercolors at the weekend."),
    ("fact", "workspace", "Alejandro leads the platform team and reviews infrastructure changes."),
]


def wait_consolidated(timeout: float = 120) -> None:
    started = time.monotonic()
    while int(psql("select count(*) from memory_proposals where state = 'pending'")[0]) > 0:
        if time.monotonic() - started > timeout:
            return
        time.sleep(1.5)


def seed() -> None:
    subprocess.run([str(HERE.parent / "terminal" / "prepare.sh"), "--quick"], check=True)
    for kind, scope, text in SEEDS:
        tool("ALICE", "memory_propose", {"kind": kind, "scope": scope, "text": text})
    wait_consolidated()
    switches(auto_recall=True)
    print("demo memory ready; recall is on")


def entries() -> list[dict]:
    """What the team's agents know, best first: a search with no floor returns up to ten."""
    found = tool("ALICE", "memory_search", {"query": "what the team knows about deploys, meetings and people", "limit": 10})
    return [
        {k: entry[k] for k in ("id", "text", "kind", "scope", "version", "confidence")}
        for entry in found.get("results", [])
    ]


def state() -> dict:
    memory = admin("GET", f"/workspaces/{WS}/memory")
    return {
        "auto_recall": memory["auto_recall"],
        "recall_rerank": memory["recall_rerank"],
        "rerank_available": memory["rerank_available"],
        "entries": entries(),
    }


def one_answer(who: str, messages: list[dict], declined: bool) -> dict:
    headers = {"Otari-Memory": "off"} if declined else {}
    before = None if declined else last_read()
    started = time.monotonic()
    status, got, body = http(
        "POST", "/chat/completions", KEYS[f"{who}_KEY"], {"model": MODEL, "max_tokens": 160, "messages": messages}, headers
    )
    elapsed = round((time.monotonic() - started) * 1000)
    if status != 200:
        return {"error": f"{status}: {body}"}
    lower = {key.lower(): value for key, value in got.items()}
    recalled = int(lower.get("otari-memory-recalled", "0"))
    handed: list[dict] = []
    if recalled:
        after = last_read()
        if after is not None and after != before:
            for entry_id in after[1]:
                opened = tool("ALICE", "memory_get", {"id": entry_id}).get("entry")
                if opened:
                    handed.append({k: opened[k] for k in ("id", "text", "kind", "scope")})
    usage = body.get("usage") or {}
    return {
        "answer": body["choices"][0]["message"]["content"].strip(),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "ms": elapsed,
        "recalled": handed,
        "memory": not declined and "otari-memory-recalled" in lower,
    }


def ask(body: dict) -> dict:
    who = "BOB" if body.get("who") == "BOB" else "ALICE"
    turns = [turn for turn in [*body.get("earlier", []), body["question"]] if turn.strip()]
    messages = [{"role": "user", "content": turn} for turn in turns]
    # One at a time: a local model answers one request at once, and timing both together would mislead.
    with ThreadPoolExecutor(max_workers=1) as pool:
        without = pool.submit(one_answer, who, messages, True).result()
        with_memory = pool.submit(one_answer, who, messages, False).result()
    return {"without": without, "with": with_memory}


def teach(body: dict) -> dict:
    proposed = tool("ALICE", "memory_propose", {"kind": body.get("kind", "procedure"), "scope": "workspace", "text": body["text"]})
    if "error" in proposed:
        return {"refused": proposed["error"]}
    wait_consolidated(90)
    return {"state": state()}


class Handler(BaseHTTPRequestHandler):
    def _send(self, status: int, payload: object, content_type: str = "application/json") -> None:
        data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802 - http.server's name
        if self.path in ("/", "/index.html"):
            self._send(200, (HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/api/state":
            self._send(200, state())
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        try:
            if self.path == "/api/ask":
                self._send(200, ask(body))
            elif self.path == "/api/settings":
                switches(**{k: bool(v) for k, v in body.items() if k in ("auto_recall", "recall_rerank")})
                self._send(200, state())
            elif self.path == "/api/teach":
                self._send(200, teach(body))
            else:
                self._send(404, {"error": "not found"})
        except SystemExit as exc:  # bench's helpers stop on a failed call
            self._send(502, {"error": str(exc)})

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass


if __name__ == "__main__":
    if "--seed" in sys.argv:
        seed()
    print(f"Agent memory demo on http://127.0.0.1:{PORT}")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
