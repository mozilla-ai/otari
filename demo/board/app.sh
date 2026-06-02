#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────
#  Research Assistant — the demo's CONSTANT.
#
#  A deliberately tiny "app": one question in, one answer out, over the
#  OpenAI-compatible /v1/chat/completions wire format. The whole point of the
#  board demo is that THIS FILE NEVER CHANGES across the acts. We only change:
#    • --base-url   where it points (a raw model, the OSS gateway, Otari.ai)
#    • --tools      whether we ask the gateway to equip server-side tools
#  …and the app gains capabilities and governance without a code change.
#
#  Env vars (override per act):
#    BASE_URL   API base, e.g. http://localhost:8088          (no trailing /v1)
#    API_KEY    bearer token (gateway master key, or platform token)
#    USER_ID    end-user id for budget attribution            (default 'demo')
#
#  Usage:
#    ./app.sh "question"                          # plain chat, no tools
#    ./app.sh --tools otari_web_search,otari_code_execution "question"
#    ./app.sh --model llamafile:local-llamafile --no-stream "question"
# ─────────────────────────────────────────────────────────────────────────
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GATEWAY_ROOT="$(cd "$HERE/../.." && pwd)"
if [[ -f "$HERE/.env" ]]; then
  set -a; # shellcheck disable=SC1091
  source "$HERE/.env"; set +a
fi

GATEWAY_PORT=${GATEWAY_PORT:-8088}
BASE_URL=${BASE_URL:-http://localhost:${GATEWAY_PORT}}
API_KEY=${API_KEY:-demo-master-key}
USER_ID=${USER_ID:-demo}

MODEL="llamafile:local-llamafile"
TOOLS=""          # comma-separated: otari_web_search,otari_code_execution
stream=1
register_user=1   # set 0 when talking to a non-gateway endpoint
show_request=0    # set 1 to print the outbound request body before sending

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-url)    BASE_URL="$2"; shift 2 ;;
    --model)       MODEL="$2"; shift 2 ;;
    --tools)       TOOLS="$2"; shift 2 ;;
    --no-stream)   stream=0; shift ;;
    --no-register) register_user=0; shift ;;
    --show-request) show_request=1; shift ;;
    -h|--help)     grep -E "^# " "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    --) shift; break ;;
    -*) echo "unknown flag: $1" >&2; exit 2 ;;
    *)  break ;;
  esac
done

query="${1:?usage: ./app.sh [--base-url URL] [--model M] [--tools t1,t2] [--no-stream] \"question\"}"

BOLD=$'\e[1m'; DIM=$'\e[2m'; YEL=$'\e[33m'; CYN=$'\e[36m'; GRN=$'\e[32m'; RED=$'\e[31m'; MAG=$'\e[35m'; RST=$'\e[0m'

# Gateway chat requires the request's `user` to be a registered user (budget
# enforcement reads its spend record). Idempotent; 409 on repeat is fine. Skip
# with --no-register for the Act-1 raw-model leg (no /v1/users there).
if [[ "$register_user" == "1" ]]; then
  curl -sf -X POST "$BASE_URL/v1/users" \
    -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" \
    -d "{\"user_id\": \"$USER_ID\"}" >/dev/null 2>&1 || true
fi

body=$(python3 -c '
import json, sys
model, query, stream, user, tools_csv = sys.argv[1:6]
req = {
    "model": model,
    "messages": [{"role": "user", "content": query}],
    "stream": stream == "1",
    "user": user,
}
tools = [t for t in tools_csv.split(",") if t]
if tools:
    # Per-tool purpose_hint nudges smaller open-weights models to actually call
    # the tool instead of hedging ("I don’t have real-time access"). Frontier
    # models do not need it, but it is harmless for them.
    hints = {
        "otari_web_search": (
            "If the user asks about current state, recent events, news, prices, "
            "or anything time-sensitive, you MUST call otari_web_search FIRST. "
            "Do not answer from training memory for such questions."
        ),
        "otari_code_execution": (
            "For any arithmetic beyond trivial mental math, you MUST call "
            "otari_code_execution and use its result. Never compute it yourself."
        ),
    }
    req["tools"] = [
        {"type": t, **({"purpose_hint": hints[t]} if t in hints else {})}
        for t in tools
    ]
    # Give the server-side tool loop room to run multiple rounds (cap is 25).
    req["max_tool_iterations"] = 20
print(json.dumps(req))
' "$MODEL" "$query" "$stream" "$USER_ID" "$TOOLS")

echo "──────────────────────────────────────────────────────────"
echo "${BOLD}Research Assistant${RST}"
echo "  Q:        $query"
echo "  base_url: $BASE_URL"
echo "  model:    $MODEL"
echo "  tools:    ${TOOLS:-${DIM}(none — plain chat)${RST}}"
echo "──────────────────────────────────────────────────────────"
echo

if [[ "$show_request" == "1" ]]; then
  echo "${BOLD}${MAG}1) REQUEST → POST $BASE_URL/v1/chat/completions${RST}"
  echo "$body" | python3 -m json.tool | sed "s/^/  ${DIM}/; s/$/${RST}/"
  echo
  echo "${BOLD}${MAG}2) RESPONSE (tool call + final answer) ↓${RST}"
fi

render_stream() {
  python3 -u -c '
import json, sys
BOLD="\033[1m"; DIM="\033[2m"; YEL="\033[33m"; CYN="\033[36m"; RED="\033[31m"; RST="\033[0m"
try:
    with open("/tmp/.app-headers") as f:
        status = f.readline().strip()
    is_error = bool(status) and " 2" not in status
except FileNotFoundError:
    is_error = False
raw=[]; slots={}
def flush():
    for s in slots.values():
        name=s.get("name","?"); args=s.get("args","")
        if name in ("web_search","code_execution","otari_web_search","otari_code_execution"):
            try: val=json.loads(args)
            except Exception: val={}
            arg = val.get("query") or val.get("code") or args
            print(f"\n{YEL}{BOLD}▶ {name}{RST}")
            for line in str(arg).splitlines():
                print(f"  {CYN}{line}{RST}")
        else:
            print(f"\n{DIM}▶ {name}({args}){RST}")
    slots.clear()
for line in sys.stdin:
    raw.append(line); line=line.strip()
    if not line.startswith("data:"): continue
    p=line[5:].strip()
    if p=="[DONE]": break
    try: chunk=json.loads(p)
    except Exception: continue
    if not chunk.get("choices"): continue
    ch=chunk["choices"][0]; delta=ch.get("delta") or {}
    for tc in delta.get("tool_calls") or []:
        s=slots.setdefault(tc.get("index",0),{"name":"","args":""})
        fn=tc.get("function") or {}
        if fn.get("name"): s["name"]+=fn["name"]
        if fn.get("arguments"): s["args"]+=fn["arguments"]
    if delta.get("content"): print(delta["content"], end="", flush=True)
    if ch.get("finish_reason")=="tool_calls": flush()
if is_error:
    print(f"{RED}{BOLD}!! endpoint returned an error:{RST}\n" + "".join(raw).strip())
    sys.exit(2)
print()
'
}

render_json() {
  python3 -c '
import json, sys
BOLD="\033[1m"; YEL="\033[33m"; CYN="\033[36m"; RED="\033[31m"; RST="\033[0m"
r=json.load(sys.stdin)
if "choices" not in r:
    print(f"{RED}{BOLD}!! endpoint returned an error:{RST}\n{json.dumps(r, indent=2)}", file=sys.stderr)
    sys.exit(2)
msg=r["choices"][0]["message"]
for tc in msg.get("tool_calls") or []:
    fn=tc.get("function") or {}; args=fn.get("arguments") or ""
    name=fn.get("name")
    if name in ("web_search","code_execution","otari_web_search","otari_code_execution"):
        try: val=json.loads(args)
        except Exception: val={}
        arg=val.get("query") or val.get("code") or args
        print(f"{YEL}{BOLD}▶ {name}{RST}")
        for line in str(arg).splitlines(): print(f"  {CYN}{line}{RST}")
        print()
print(msg.get("content") or "(no content)")
'
}

if [[ "$stream" == "1" ]]; then
  curl -sN -X POST "$BASE_URL/v1/chat/completions" \
    -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" \
    -d "$body" -D /tmp/.app-headers | render_stream
else
  curl -sS -X POST "$BASE_URL/v1/chat/completions" \
    -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" \
    -d "$body" | render_json
fi
