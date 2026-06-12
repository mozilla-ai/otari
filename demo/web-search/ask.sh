#!/usr/bin/env bash
# Send a single chat completion to the OSS gateway with web search enabled.
# Streams the SSE response and highlights tool calls + results as they land.
# Use `--no-stream` for backends that don't support streaming
# (e.g. the llamafile path).
#
# The tool-array `type` decides who runs the search:
#   * otari_web_search   → the gateway's own web_search backend runs it
#   * web_search / web_search_<date>  → NOT intercepted; forwarded to the
#     provider as-is. Whether the provider runs a native search depends on the
#     provider AND endpoint — this script always calls /v1/chat/completions,
#     where e.g. Anthropic's server-side web_search does NOT fire (it expects
#     /v1/messages), so the model may just answer from memory. Use
#     otari_web_search for the reliable gateway-run path.
#
# Env vars:
#   OTARI_URL  — default http://localhost:${OTARI_PORT:-8000}
#   OTARI_KEY  — default 'demo-master-key'  (your master key or API key)
#   OTARI_USER — default 'demo'
#
# Usage:
#   ./ask.sh "What's the latest stable release of Python?"   # otari_web_search (gateway runs it)
#   ./ask.sh --tool-type web_search "..."                    # passthrough (see note above)

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
SEARXNG_CONTAINER=$(cd "$OTARI_ROOT" && docker compose ps -q searxng 2>/dev/null | head -1 || true)
# Brave adapter is up only when the gateway uses the `web-search-brave` profile
# (./start.sh --brave). Empty otherwise — its log section is skipped.
BRAVE_CONTAINER=$(cd "$OTARI_ROOT" && docker compose ps -q brave-adapter 2>/dev/null | head -1 || true)

OTARI_PORT=${OTARI_PORT:-8000}
OTARI_URL=${OTARI_URL:-http://localhost:${OTARI_PORT}}
OTARI_KEY=${OTARI_KEY:-demo-master-key}
OTARI_USER=${OTARI_USER:-demo}

MODEL="anthropic:claude-sonnet-4-6"
TOOL_TYPE="otari_web_search"
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
# Per-tool purpose_hint override. The gateway-default hint is intentionally
# soft ("Prefer web_search for..."); that suits frontier models but small
# open-weight models (Qwen3-4B etc.) often hedge ("I do not have real-time
# access") instead of calling the tool. A directive demo-side override makes
# the loop reliable across model sizes without coercing every caller.
print(json.dumps({
    "model": sys.argv[1],
    "messages": [{"role": "user", "content": sys.argv[2]}],
    "tools": [{
        "type": sys.argv[5],
        "purpose_hint": (
            "If the user asks about current state, recent events, news, prices, "
            "or anything time-sensitive, you MUST call web_search FIRST. "
            "Do not answer from training memory for such questions."
        ),
    }],
    "stream": sys.argv[3] == "1",
    "user": sys.argv[4],
}))
' "$MODEL" "$query" "$stream" "$OTARI_USER" "$TOOL_TYPE")

if [[ "$TOOL_TYPE" == "otari_web_search" ]]; then
  runs_in="gateway web_search backend (OTARI_WEB_SEARCH_URL set in compose)"
else
  runs_in="provider's native search (gateway passes '$TOOL_TYPE' through)"
fi

echo "──────────────────────────────────────────────────────────"
echo "Q: $query"
echo "    model:     $MODEL"
echo "    tool-type: $TOOL_TYPE"
echo "    runs in:   $runs_in"
echo "──────────────────────────────────────────────────────────"
echo

_gw_before=$(docker logs $OTARI_CONTAINER 2>&1 | wc -l | tr -d ' ' || echo 0)
_sx_before=$(docker logs $SEARXNG_CONTAINER 2>&1 | wc -l | tr -d ' ' || echo 0)
_bx_before=$(docker logs $BRAVE_CONTAINER 2>&1 | wc -l | tr -d ' ' || echo 0)

if [[ "$stream" == "1" ]]; then
  curl -sN -X POST "$OTARI_URL/v1/chat/completions" \
    -H "Authorization: Bearer $OTARI_KEY" \
    -H "Content-Type: application/json" \
    -d "$body" -D /tmp/.ask-headers | python3 -u -c '
import json, sys

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
        if name == "web_search":
            try:
                q = json.loads(args).get("query", args)
            except Exception:
                q = args
            print(f"\n{YEL}{BOLD}▶ web_search{RST}")
            print(f"  {CYN}query: {q}{RST}")
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

BOLD="\033[1m"; YEL="\033[33m"; CYN="\033[36m"; RED="\033[31m"; RST="\033[0m"

r = json.load(sys.stdin)
if "choices" not in r:
    print(f"{RED}{BOLD}!! gateway returned an error:{RST}\n{json.dumps(r, indent=2)}", file=sys.stderr)
    sys.exit(2)
msg = r["choices"][0]["message"]
for tc in msg.get("tool_calls") or []:
    fn = tc.get("function") or {}
    args = fn.get("arguments") or ""
    if fn.get("name") == "web_search":
        try:
            q = json.loads(args).get("query", args)
        except Exception:
            q = args
        print(f"{YEL}{BOLD}▶ web_search{RST}")
        print(f"  {CYN}query: {q}{RST}")
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
if [[ -n "$BRAVE_CONTAINER" ]]; then
  echo
  echo "${BOLD}${DIM}── brave-adapter saw ──${RST}"
  docker logs $BRAVE_CONTAINER 2>&1 \
    | tail -n +$((_bx_before + 1)) \
    | grep -iE "GET /search|error" \
    | sed "s/^/${DIM}  /; s/$/${RST}/" \
    || true
else
  echo
  echo "${BOLD}${DIM}── searxng saw ──${RST}"
  docker logs $SEARXNG_CONTAINER 2>&1 \
    | tail -n +$((_sx_before + 1)) \
    | grep -iE "search|engine|ERROR|WARNING" \
    | sed "s/^/${DIM}  /; s/$/${RST}/" \
    || true
fi
