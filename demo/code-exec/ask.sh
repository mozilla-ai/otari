#!/usr/bin/env bash
# Send a single chat completion to the OSS gateway with a code-execution-eligible
# question. Streams the SSE response and highlights tool calls + results as
# they land. Use `--no-stream` for backends that don't support streaming
# (e.g. the llamafile path).
#
# The tool-array `type` decides who runs the code:
#   * otari_code_execution  → the gateway's own sandbox container runs it
#   * code_interpreter / code_execution_<date>  → passed through to the
#     provider, which runs it in its native sandbox (the gateway just proxies)
#
# Env vars:
#   OTARI_URL  — default http://localhost:${OTARI_PORT:-8000}
#   OTARI_KEY  — default 'demo-master-key'  (your master key or API key)
#   OTARI_USER — default 'demo'
#
# Usage:
#   ./ask.sh "What is 23 factorial?"                              # gateway sandbox
#   ./ask.sh --model openai:gpt-4o-mini --tool-type code_interpreter "..."  # OpenAI native
#   ./ask.sh --model anthropic:... --tool-type code_execution_20250825 "..." # Anthropic native

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OTARI_ROOT="$(cd "$HERE/../.." && pwd)"
if [[ -f "$HERE/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$HERE/.env"
  set +a
fi

# Resolve compose service -> container id (project-name-agnostic so the demo
# survives renaming the repo directory). Empty fallback when the service is
# not running; callers tolerate it via `|| echo 0` / 2>&1.
OTARI_CONTAINER=$(cd "$OTARI_ROOT" && docker compose ps -q otari 2>/dev/null | head -1 || true)
SANDBOX_CONTAINER=$(cd "$OTARI_ROOT" && docker compose ps -q sandbox 2>/dev/null | head -1 || true)

OTARI_PORT=${OTARI_PORT:-8000}
OTARI_URL=${OTARI_URL:-http://localhost:${OTARI_PORT}}
OTARI_KEY=${OTARI_KEY:-demo-master-key}
OTARI_USER=${OTARI_USER:-demo}

MODEL="anthropic:claude-sonnet-4-6"
TOOL_TYPE="otari_code_execution"
stream=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)      MODEL="$2"; shift 2 ;;
    --tool-type)  TOOL_TYPE="$2"; shift 2 ;;
    --no-stream)  stream=0; shift ;;
    -h|--help)
      grep -E "^# " "$0" | sed 's/^# //'
      exit 0
      ;;
    --) shift; break ;;
    -*) echo "unknown flag: $1" >&2; exit 2 ;;
    *)  break ;;
  esac
done

query="${1:?usage: ./ask.sh [--model MODEL] [--tool-type TYPE] [--no-stream] \"your question\"}"

# Ensure the demo user exists. The gateway's chat-completions path requires
# the request's `user` to be a registered user (budget enforcement reads its
# spend record); first-boot the demo user doesn't exist yet. Idempotent —
# 409 on subsequent calls is fine.
curl -sf -X POST "$OTARI_URL/v1/users" \
  -H "Authorization: Bearer $OTARI_KEY" \
  -H "Content-Type: application/json" \
  -d "{\"user_id\": \"$OTARI_USER\"}" >/dev/null 2>&1 || true

body=$(python3 -c '
import json, sys
print(json.dumps({
    "model": sys.argv[1],
    "messages": [{"role": "user", "content": sys.argv[2]}],
    "tools": [{"type": sys.argv[5]}],
    "stream": sys.argv[3] == "1",
    "user": sys.argv[4],
}))
' "$MODEL" "$query" "$stream" "$OTARI_USER" "$TOOL_TYPE")

if [[ "$TOOL_TYPE" == "otari_code_execution" ]]; then
  runs_in="gateway sandbox (OTARI_SANDBOX_URL set in compose)"
else
  runs_in="provider's native sandbox (gateway passes '$TOOL_TYPE' through)"
fi

echo "──────────────────────────────────────────────────────────"
echo "Q: $query"
echo "    model:     $MODEL"
echo "    tool-type: $TOOL_TYPE"
echo "    runs in:   $runs_in"
echo "──────────────────────────────────────────────────────────"
echo

_gw_before=$(docker logs $OTARI_CONTAINER 2>&1 | wc -l | tr -d ' ' || echo 0)
_sbx_before=$(docker logs $SANDBOX_CONTAINER 2>&1 | wc -l | tr -d ' ' || echo 0)

if [[ "$stream" == "1" ]]; then
  curl -sN -X POST "$OTARI_URL/v1/chat/completions" \
    -H "Authorization: Bearer $OTARI_KEY" \
    -H "Content-Type: application/json" \
    -d "$body" -D /tmp/.ask-headers | python3 -u -c '
import json, os, sys

BOLD="\033[1m"; DIM="\033[2m"; YEL="\033[33m"; CYN="\033[36m"; GRN="\033[32m"; RED="\033[31m"; RST="\033[0m"

def _print_error_if_not_sse() -> bool:
    try:
        with open("/tmp/.ask-headers") as f:
            status_line = f.readline().strip()
    except FileNotFoundError:
        return False
    if status_line and " 2" not in status_line:
        return True
    return False

is_error = _print_error_if_not_sse()
raw_payload = []
slots: dict[int, dict[str, str]] = {}

def emit_complete_tool_calls() -> None:
    for slot in slots.values():
        name = slot.get("name", "?")
        args = slot.get("args", "")
        if name == "code_execution":
            try:
                code = json.loads(args).get("code", args)
            except Exception:
                code = args
            print(f"\n{YEL}{BOLD}▶ code_execution{RST}")
            for line in code.splitlines():
                print(f"  {CYN}{line}{RST}")
        else:
            print(f"\n{DIM}▶ {name}({args}){RST}")
    slots.clear()

for line in sys.stdin:
    raw_payload.append(line)
    line = line.strip()
    if not line.startswith("data:"):
        continue
    payload = line[5:].strip()
    if payload == "[DONE]":
        break
    try:
        chunk = json.loads(payload)
    except Exception:
        continue
    if not chunk.get("choices"):
        continue
    choice = chunk["choices"][0]
    delta = choice.get("delta") or {}
    for tc in delta.get("tool_calls") or []:
        idx = tc.get("index", 0)
        slot = slots.setdefault(idx, {"name": "", "args": ""})
        fn = tc.get("function") or {}
        if fn.get("name"):
            slot["name"] += fn["name"]
        if fn.get("arguments"):
            slot["args"] += fn["arguments"]
    if delta.get("content"):
        print(delta["content"], end="", flush=True)
    if choice.get("finish_reason") == "tool_calls":
        emit_complete_tool_calls()

if is_error:
    body = "".join(raw_payload).strip() or "(empty body)"
    print(f"{RED}{BOLD}!! gateway returned an error:{RST}\n{body}")
    sys.exit(2)
print()
'
else
  curl -sS -X POST "$OTARI_URL/v1/chat/completions" \
    -H "Authorization: Bearer $OTARI_KEY" \
    -H "Content-Type: application/json" \
    -d "$body" | python3 -c '
import json, sys

BOLD="\033[1m"; DIM="\033[2m"; YEL="\033[33m"; CYN="\033[36m"; RED="\033[31m"; RST="\033[0m"

r = json.load(sys.stdin)
if "choices" not in r:
    print(f"{RED}{BOLD}!! gateway returned an error:{RST}\n{json.dumps(r, indent=2)}", file=sys.stderr)
    sys.exit(2)
msg = r["choices"][0]["message"]
# Surface tool calls the assistant made along the way (non-streaming path
# only has the final assistant message, so this reflects the *last* round).
for tc in msg.get("tool_calls") or []:
    fn = tc.get("function") or {}
    args = fn.get("arguments") or ""
    if fn.get("name") == "code_execution":
        try:
            code = json.loads(args).get("code", args)
        except Exception:
            code = args
        print(f"{YEL}{BOLD}▶ code_execution{RST}")
        for line in code.splitlines():
            print(f"  {CYN}{line}{RST}")
        print()
print(msg.get("content") or "(no content)")
'
fi

DIM=$'\e[2m'; BOLD=$'\e[1m'; RST=$'\e[0m'
echo
echo "${BOLD}${DIM}── gateway saw ──${RST}"
docker logs $OTARI_CONTAINER 2>&1 \
  | tail -n +$((_gw_before + 1)) \
  | grep -iE "POST .*chat|ERROR" \
  | sed "s/^/${DIM}  /; s/$/${RST}/" \
  || true
echo
echo "${BOLD}${DIM}── sandbox saw ──${RST}"
docker logs $SANDBOX_CONTAINER 2>&1 \
  | tail -n +$((_sbx_before + 1)) \
  | grep -iE "POST|DELETE" \
  | sed "s/^/${DIM}  /; s/$/${RST}/" \
  || true
