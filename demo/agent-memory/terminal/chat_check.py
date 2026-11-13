"""One streamed chat turn through the local gateway, as Open WebUI sends it: what came back, and what memory did.

    chat_check.py WHO "message" ["earlier message" ...]   (earlier messages go first)
"""

import json
import pathlib
import sys
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "bench"))
from bench import KEYS  # noqa: E402

MODEL = "openai/gpt-oss-120b"
who, message, earlier = sys.argv[1], sys.argv[2], sys.argv[3:]
messages = [{"role": "user", "content": turn} for turn in earlier] + [{"role": "user", "content": message}]
request = urllib.request.Request(
    "http://127.0.0.1:8010/api/v1/chat/completions",
    json.dumps({"model": MODEL, "stream": True, "messages": messages}).encode(),
    {"Authorization": f"Bearer {KEYS[f'{who}_KEY']}", "Content-Type": "application/json"},
)
with urllib.request.urlopen(request, timeout=180) as response:
    headers = {key.lower(): value for key, value in response.headers.items()}
    text = []
    for raw in response:
        line = raw.decode().strip()
        if not line.startswith("data:") or line == "data: [DONE]":
            continue
        for choice in json.loads(line[5:]).get("choices", []):
            text.append((choice.get("delta") or {}).get("content") or "")
print("".join(text).strip())
print({key: value for key, value in headers.items() if key.startswith("otari-memory")})
