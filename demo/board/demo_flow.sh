#!/usr/bin/env bash
# ═════════════════════════════════════════════════════════════════════════
#  Otari board demo — "The Research Assistant that grows up"
#
#  One tiny app (app.sh), unchanged throughout. We change only what it points
#  at and whether the gateway equips tools, and watch it gain capabilities,
#  privacy, and governance. Narrative arc: open-source Otari → Otari.ai.
#
#  Acts 1-3 run live here against the local OSS gateway + a local open-weights
#  model (llamafile). Act 4 (Otari.ai) is a talk-track + UI checklist — see
#  RUNBOOK.md — because the platform, wallet, and managed providers live in the
#  hosted product, not this repo.
#
#  Prereqs:
#    • ./start.sh running in another terminal (gateway + sandbox + searxng)
#    • a llamafile reachable on :8080, OR $LLAMAFILE_BIN set so we auto-start one
#
#  Usage:
#    ./demo_flow.sh                 # full paced walkthrough
#    ./demo_flow.sh --fast          # no "press Enter" pauses
# ═════════════════════════════════════════════════════════════════════════
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
GATEWAY_ROOT="$(cd "$HERE/../.." && pwd)"
APP="$HERE/app.sh"

if [[ -f "$HERE/.env" ]]; then
  set -a; # shellcheck disable=SC1091
  source "$HERE/.env"; set +a
fi

BOLD=$'\e[1m'; DIM=$'\e[2m'; YEL=$'\e[33m'; CYN=$'\e[36m'; GRN=$'\e[32m'; RED=$'\e[31m'; MAG=$'\e[35m'; RST=$'\e[0m'

PAUSE=1
[[ "${1:-}" == "--fast" ]] && PAUSE=0
# pause sets SKIP=1 when the presenter asks to skip the NEXT step (types s/skip),
# otherwise SKIP=0. In --fast mode it never pauses and never skips.
SKIP=0
pause() {
  SKIP=0
  [[ "$PAUSE" == "1" ]] || { echo; return; }
  echo
  local ans
  read -r -p "${DIM}  press Enter to continue · type 's' + Enter to skip the next step…${RST} " ans
  case "$ans" in
    s|S|skip|SKIP) SKIP=1; echo "${DIM}  (skipping)${RST}" ;;
  esac
  echo
}

GATEWAY_PORT=${GATEWAY_PORT:-8088}
BASE_URL="http://localhost:${GATEWAY_PORT}"
LLAMAFILE_API_BASE=${LLAMAFILE_API_BASE:-http://host.docker.internal:8080/v1}
PROBE=${LLAMAFILE_API_BASE/host.docker.internal/127.0.0.1}

# Act 4 (platform mode) — read from .env, checked just-in-time before Act 4.
OTARI_MODEL=${OTARI_MODEL:-}
OTARI_USER_TOKEN=${OTARI_USER_TOKEN:-}

# The gateway dispatches ONE server-side tool backend per request (combining
# otari_web_search + otari_code_execution in a single call is a planned
# refinement, not yet allowed). So we use two purpose-built questions — one per
# tool — and show the same app, same model, gaining each capability in turn.
#
#   Q_SEARCH    — needs CURRENT info the model can't know from training
#                 (who won the most recent EuroLeague basketball title).
#   Q_CODE      — date arithmetic the model shouldn't fake (Python's 5-year
#                 support window → EOL date + days remaining). Self-contained
#                 and offline-safe: stdlib datetime only.
#
# Two independent capabilities, one per tool: live knowledge, then exact compute.
Q_BASELINE="Who won the most recent EuroLeague basketball title, and how many days ago was the final?"
Q_SEARCH="Who won the most recent EuroLeague basketball championship in Athens, Greece, and on what date was the final played?"
Q_CODE="The latest stable Python is 3.14, first released on 2025-10-07. Python feature releases get 5 years of support from their initial release. Using today's date, compute 3.14's end-of-life date and exactly how many days from today until then."

present() {
  local title="$1"; shift
  echo
  echo "${BOLD}${YEL}═══════════════════════════════════════════════════════════════════════${RST}"
  echo "${BOLD}${YEL}  $title${RST}"
  echo "${BOLD}${YEL}═══════════════════════════════════════════════════════════════════════${RST}"
  for line in "$@"; do echo "${DIM}  $line${RST}"; done
  echo
}

# Resolve compose service → container id (project-name-agnostic).
GW_CONTAINER=$(cd "$GATEWAY_ROOT" && docker compose ps -q gateway 2>/dev/null | head -1 || true)
SBX_CONTAINER=$(cd "$GATEWAY_ROOT" && docker compose ps -q sandbox 2>/dev/null | head -1 || true)
SX_CONTAINER=$(cd "$GATEWAY_ROOT" && docker compose ps -q searxng 2>/dev/null | head -1 || true)

# run_with_proof <backend> <label> <ask.sh args...>
#   Snapshots the backend container's log line count, runs the app call (which
#   prints REQUEST + RESPONSE), then prints only the NEW backend log lines —
#   proof the tool was actually invoked and what it answered. <backend> is one
#   of: searxng | sandbox.
run_with_proof() {
  local backend="$1"; shift
  local container grep_re
  case "$backend" in
    searxng) container="$SX_CONTAINER";  grep_re="search|engine|GET" ;;
    sandbox) container="$SBX_CONTAINER"; grep_re="POST|DELETE|sessions|exec" ;;
    *) container=""; grep_re="." ;;
  esac
  local before=0
  [[ -n "$container" ]] && before=$(docker logs "$container" 2>&1 | wc -l | tr -d ' ' || echo 0)

  "$APP" "$@" || true

  echo
  echo "${BOLD}${MAG}3) TOOL ran server-side — ${backend} container saw:${RST}"
  if [[ -n "$container" ]]; then
    docker logs "$container" 2>&1 \
      | tail -n +$((before + 1)) \
      | grep -iE "$grep_re" \
      | tail -8 \
      | sed "s/^/  ${DIM}/; s/$/${RST}/" \
      || echo "  ${DIM}(no new ${backend} log lines — tool may not have fired)${RST}"
  else
    echo "  ${DIM}(${backend} container not found; is the right compose profile up?)${RST}"
  fi
}

OPENAI_MODEL=${OPENAI_MODEL:-gpt-5.5}

# openai_frontier <tool> <question>
#   Calls OpenAI's Responses API DIRECTLY (not through the gateway) with one of
#   its NATIVE hosted tools, to show what a frontier provider gives you out of
#   the box. <tool> is "web_search" or "shell". Renders which tools fired plus
#   the final text. No-op with a clear message if OPENAI_API_KEY isn't set.
openai_frontier() {
  local tool="$1" question="$2"
  if [[ -z "${OPENAI_API_KEY:-}" || "$OPENAI_API_KEY" == *REPLACE_ME* ]]; then
    echo "  ${YEL}(skipped — set OPENAI_API_KEY in .env to show the frontier contrast)${RST}"
    return 0
  fi

  # Build the Responses API body. The shell tool needs a container env and the
  # message-array input shape; web_search takes the simple string input.
  local body
  body=$(python3 -c '
import json, sys
model, tool, q = sys.argv[1:4]
if tool == "shell":
    req = {
        "model": model,
        "tools": [{"type": "shell", "environment": {"type": "container_auto"}}],
        "input": [{
            "type": "message", "role": "user",
            "content": [{"type": "input_text", "text": q}],
        }],
        "tool_choice": "auto",
    }
else:
    req = {"model": model, "tools": [{"type": tool}], "input": q}
print(json.dumps(req))
' "$OPENAI_MODEL" "$tool" "$question")

  echo "${BOLD}${MAG}1) REQUEST → POST https://api.openai.com/v1/responses${RST}"
  echo "$body" | python3 -m json.tool | sed "s/^/  ${DIM}/; s/$/${RST}/"
  echo
  echo "${BOLD}${MAG}2) RESPONSE (native ${tool} tool + answer) ↓${RST}"
  curl -sS -L "https://api.openai.com/v1/responses" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $OPENAI_API_KEY" \
    -d "$body" | python3 -c '
import json, sys
BOLD="\033[1m"; DIM="\033[2m"; YEL="\033[33m"; CYN="\033[36m"; GRN="\033[32m"; RED="\033[31m"; RST="\033[0m"
try: r=json.load(sys.stdin)
except Exception as e:
    print(f"{RED}{BOLD}!! could not parse OpenAI response: {e}{RST}"); sys.exit(2)
err=r.get("error")
if err:
    msg=err.get("message", err) if isinstance(err, dict) else err
    print(f"{RED}{BOLD}!! OpenAI error:{RST} {msg}"); sys.exit(2)
# Walk the Responses API output array: tool-call items + the final message.
text_parts=[]
for item in r.get("output", []) or []:
    itype=item.get("type","")
    if itype in ("web_search_call","shell_call") or itype.endswith("_call"):
        label=itype.replace("_call","")
        action=item.get("action") or {}
        detail=action.get("query") or action.get("command") or action.get("type") or ""
        if isinstance(detail, list): detail=" ".join(map(str, detail))
        print(f"{YEL}{BOLD}▶ {label}{RST}")
        if detail: print(f"  {CYN}{detail}{RST}")
    elif itype=="message":
        for c in item.get("content", []) or []:
            if c.get("type")=="output_text" and c.get("text"):
                text_parts.append(c["text"])
# Fallback to the convenience field some SDKs populate.
if not text_parts and r.get("output_text"):
    text_parts.append(r["output_text"])
print()
print("\n".join(text_parts) or "(no text output)")
'
}

# ───── llamafile auto-start (mirrors the single-feature demos) ─────────────
_LLAMAFILE_PID=""
_cleanup() {
  if [[ -n "$_LLAMAFILE_PID" ]] && kill -0 "$_LLAMAFILE_PID" 2>/dev/null; then
    echo "${DIM}stopping auto-started llamafile (pid $_LLAMAFILE_PID)…${RST}"
    kill "$_LLAMAFILE_PID" 2>/dev/null || true
  fi
}
trap _cleanup EXIT

ensure_llamafile() {
  if curl -sf "${PROBE}/models" >/dev/null 2>&1; then return 0; fi
  if [[ -z "${LLAMAFILE_BIN:-}" || ! -x "${LLAMAFILE_BIN:-/nonexistent}" ]]; then
    cat <<EOF
${RED}No llamafile reachable at ${PROBE} and \$LLAMAFILE_BIN is not set.${RST}
${DIM}Download one once and point \$LLAMAFILE_BIN at it:

    curl -L -o /tmp/qwen3-4b.llamafile \\
        https://huggingface.co/mozilla-ai/Qwen3-4B-llamafile/resolve/main/Qwen_Qwen3-4B-Q5_K_M.llamafile
    chmod +x /tmp/qwen3-4b.llamafile
    export LLAMAFILE_BIN=/tmp/qwen3-4b.llamafile

Then re-run.${RST}
EOF
    return 1
  fi
  echo "${DIM}auto-starting llamafile from ${LLAMAFILE_BIN##*/} (≈30s warmup)…${RST}"
  "$LLAMAFILE_BIN" --server --port 8080 --host 0.0.0.0 -ngl 0 \
    --alias local-llamafile > /tmp/llamafile-board-demo.log 2>&1 &
  _LLAMAFILE_PID=$!
  for _ in $(seq 1 90); do
    curl -sf "${PROBE}/models" >/dev/null 2>&1 && { echo "${DIM}  ready (pid $_LLAMAFILE_PID)${RST}"; return 0; }
    kill -0 "$_LLAMAFILE_PID" 2>/dev/null || { echo "${YEL}  llamafile exited early — see /tmp/llamafile-board-demo.log${RST}"; _LLAMAFILE_PID=""; return 1; }
    sleep 1
  done
  echo "${YEL}  llamafile didn't become ready in 90s${RST}"; return 1
}

# ───── preflight ───────────────────────────────────────────────────────────
if ! curl -sf "${BASE_URL}/health" >/dev/null 2>&1; then
  echo "${RED}Gateway not reachable at ${BASE_URL}. Run ./start.sh in another terminal first.${RST}"
  exit 1
fi
ensure_llamafile || exit 1

# Open-weights model resolved off the running llamafile (id may vary by build).
MODEL_ID=$(curl -sS "${PROBE}/models" 2>/dev/null | python3 -c '
import json,sys
try:
    d=json.loads(sys.stdin.read() or "null"); data=(d or {}).get("data") or []
    print(data[0].get("id","") if data else "")
except Exception: print("")
')
[[ -z "$MODEL_ID" ]] && MODEL_ID="local-llamafile"
OSS_MODEL="llamafile:${MODEL_ID}"

# ═════════════════════════════════════════════════════════════════════════
present "The setup" \
        "Meet our app: a Research Assistant. One box, one question." \
        "" \
        "Watch carefully: from here on, the APP CODE NEVER CHANGES." \
        "We only change where it points and which tool the gateway equips."
pause

# ───── ACT 1 ───────────────────────────────────────────────────────────────
present "Act 1 — Frontier vs. open, side by side" \
        "First the frontier: OpenAI's ${OPENAI_MODEL} with its NATIVE hosted" \
        "tools — web_search and shell. These ship with the provider;" \
        "they're a big reason teams stay. Then the SAME question to a bare" \
        "open-weights model running locally — and watch what's missing."

# — Question 1: needs current info —
present "Q1 (current info): ${Q_SEARCH}"
pause
if [[ "$SKIP" != "1" ]]; then
  echo "${BOLD}${GRN}── Frontier: OpenAI ${OPENAI_MODEL} + native web_search ──${RST}"
  echo
  openai_frontier web_search "$Q_SEARCH"
  echo
  echo "${BOLD}${RED}── Open-weights, local, NO tools ──${RST}"
  echo
  "$APP" --base-url "$BASE_URL" --model "$OSS_MODEL" --no-stream "$Q_SEARCH" || true
fi
present "The contrast" \
        "Frontier reached out and got the live answer. The open model" \
        "answered from stale training memory — and can't know what shipped" \
        "after its cutoff."
pause

# — Question 2: needs real computation —
present "Q2 (computation): ${Q_CODE}"
pause
if [[ "$SKIP" != "1" ]]; then
  echo "${BOLD}${GRN}── Frontier: OpenAI ${OPENAI_MODEL} + native shell ──${RST}"
  echo
  openai_frontier shell "$Q_CODE"
  echo
  echo "${BOLD}${RED}── Open-weights, local, NO tools ──${RST}"
  echo
  "$APP" --base-url "$BASE_URL" --model "$OSS_MODEL" --no-stream "$Q_CODE" || true
fi
present "The contrast" \
        "Frontier ran code and computed it exactly. The open model did the" \
        "date math in its head — and you can't trust that." \
        "" \
        "${BOLD}This is the fear:${RST} switch to open weights and you lose" \
        "web_search, you lose shell/code — the tools you lean on every day." \
        "Hold that thought. (Acts 2 fixes it.)"
pause

# ───── ACT 2 ───────────────────────────────────────────────────────────────
# The gateway runs one server-side tool backend per request, so we show each
# capability in its own call — same app, same model, one added line each time.
present "Act 2 — Same app, same model, now through Otari" \
        "We add ONE line to the request — a single-tool array — and let the" \
        "gateway run that tool server-side. No app logic changes." \
        "" \
        "First: web search.  tools: [ { type: otari_web_search } ]" \
        "  → live results via a SearXNG container the gateway calls." \
        "(The 'otari_' prefix is what tells the gateway to run it server-side.)"
pause
if [[ "$SKIP" != "1" ]]; then
  echo "${BOLD}$Q_SEARCH${RST}"; echo
  run_with_proof searxng --base-url "$BASE_URL" --model "$OSS_MODEL" \
         --tools otari_web_search --show-request --no-stream "$Q_SEARCH"
fi
present "Now the other tool: code execution" \
        "A different capability: exact computation. Python's 5-year support" \
        "window — when does 3.14 hit end-of-life, and how many days from today?" \
        "Same app, same model, swap the one line:" \
        "    tools: [ { type: otari_code_execution } ]" \
        "  → a sandboxed Python REPL the gateway runs and feeds back."
pause
if [[ "$SKIP" != "1" ]]; then
  echo "${BOLD}$Q_CODE${RST}"; echo
  run_with_proof sandbox --base-url "$BASE_URL" --model "$OSS_MODEL" \
         --tools otari_code_execution --show-request --no-stream "$Q_CODE"
fi
present "The headline" \
        "${GRN}The open-weights model just matched the frontier experience —${RST}" \
        "${GRN}current facts AND trustworthy math — with ZERO tool code.${RST}" \
        "One added line per capability. Otari closed the gap." \
        "" \
        "(Combining both tools in a single call is a planned refinement;" \
        "today the gateway runs one server-side tool per request.)" \
        "" \
        "And note: that model ran locally, the gateway ran locally —" \
        "the prompt never left this machine. Self-hosted = a real privacy" \
        "boundary, not a marketing one."
pause

# ───── ACT 3 ───────────────────────────────────────────────────────────────
present "Act 3 — Now scale it to a team of 50" \
        "The OSS gateway already does a lot: per-user budgets, rate limits," \
        "virtual API keys, usage + cost tracking — all via its API. Plenty" \
        "for one team that's happy running and curl-ing the gateway itself." \
        "" \
        "But a company hits the operational ceiling of an API-only engine:" \
        "" \
        "  • Admin is one shared master key — no SSO, roles, or org isolation." \
        "  • Provider keys still live in env files / config on the box." \
        "  • No UI: budgets, spend, and traces are JSON, not dashboards" \
        "    finance and team leads can actually read." \
        "  • No managed providers — every model needs your own key." \
        "  • Config is hand-edited YAML on a host, not governed per-org." \
        "" \
        "The OSS gateway is the engine. A 50-person org needs the cockpit."
pause

# ───── ACT 4 — connect to Otari.ai (platform mode) ───────────────────────
# Platform mode resolves ALL models via Otari.ai, so the local-llamafile path
# from Acts 1-3 is gone — that's why the gateway must be RESTARTED with the
# token rather than reconfigured in place. The flow pauses while you do that in
# the gateway terminal, then waits for /health to report platform mode.
present "Act 4 — Connect the gateway to Otari.ai" \
        "So far the gateway ran standalone. Now we connect it to Otari.ai by" \
        "giving it a gateway token — same stack, restarted in platform mode." \
        "" \
        "The narrative: run the SAME flow, but on a much stronger model served" \
        "through Otari.ai via Mozilla.ai Inference — no provider key in sight," \
        "billed to the org wallet, every call traced and budgeted." \
        "" \
        "${BOLD}In the GATEWAY terminal, restart in platform mode now:${RST}" \
        "  ./stop.sh && ./start-platform.sh" \
        "" \
        "(.env must have OTARI_AI_TOKEN, OTARI_USER_TOKEN, OTARI_MODEL set.)" \
        "Press Enter here once you've started it; I'll wait for it to come up."
pause

# Preflight: wait for the gateway to be back AND in platform mode.
echo "${DIM}waiting for the gateway to report platform mode…${RST}"
platform_ready=0
for _ in $(seq 1 60); do
  health=$(curl -sf "${BASE_URL}/health" 2>/dev/null || true)
  if echo "$health" | grep -q '"mode"[[:space:]]*:[[:space:]]*"platform"'; then
    platform_ready=1; break
  fi
  sleep 2
done
if [[ "$platform_ready" != "1" ]]; then
  echo "${RED}Gateway is not in platform mode (or not reachable) at ${BASE_URL}.${RST}"
  echo "${DIM}/health said: ${health:-<no response>}${RST}"
  echo "${YEL}Ensure OTARI_AI_TOKEN is set and you ran ./start-platform.sh, then re-run.${RST}"
  exit 1
fi
if echo "$health" | grep -q '"platform_reachable"[[:space:]]*:[[:space:]]*"no"'; then
  echo "${YEL}Note: platform mode is on but the gateway can't reach Otari.ai right now."
  echo "Check the token / PLATFORM_BASE_URL. Continuing…${RST}"
fi
if [[ -z "$OTARI_MODEL" ]]; then
  echo "${RED}OTARI_MODEL is not set in .env (e.g. OTARI_MODEL=mzai:<your-model>).${RST}"; exit 1
fi
if [[ -z "$OTARI_USER_TOKEN" || "$OTARI_USER_TOKEN" == *REPLACE_ME* ]]; then
  echo "${RED}OTARI_USER_TOKEN is not set in .env (project/API token = the request bearer).${RST}"; exit 1
fi

present "Act 4 — Connected to Otari.ai" \
        "The gateway is now in platform mode: ${GRN}mode=platform${RST}." \
        "Same app, same questions — but model = ${BOLD}${OTARI_MODEL}${RST}," \
        "served through Mozilla.ai Inference. There's NO provider key in the" \
        "request; Otari.ai resolves it, bills the wallet, traces the call." \
        "" \
        "Two tokens connect us: the gateway carries an Otari.ai token (server-" \
        "to-server), and the app sends an Otari.ai project token as its bearer." \
        "Neither is a raw provider key — those stay on the platform."
pause

# The app sends API_KEY as the client bearer. In platform mode the gateway
# forwards it to Otari.ai as X-User-Token, so this is the project/API token.
export API_KEY="$OTARI_USER_TOKEN"

present "Q1 on the managed model — with otari_web_search" "${Q_SEARCH}"
pause
if [[ "$SKIP" != "1" ]]; then
  "$APP" --base-url "$BASE_URL" --model "$OTARI_MODEL" \
         --tools otari_web_search --show-request --no-register --no-stream "$Q_SEARCH" || true
fi

present "Q2 on the managed model — with otari_code_execution" "${Q_CODE}"
pause
if [[ "$SKIP" != "1" ]]; then
  "$APP" --base-url "$BASE_URL" --model "$OTARI_MODEL" \
         --tools otari_code_execution --show-request --no-register --no-stream "$Q_CODE" || true
fi
present "What changed vs. Act 2" \
        "Same app, same tools — but a much stronger model" \
        "${GRN}The open path and the managed path are the same code.${RST}"
pause

present "Now the platform cockpit (switch to the browser)" \
        "Everything that call just did is visible and governable in Otari.ai:" \
        "" \
        "  1. Managed providers — frontier AND Mozilla.ai open-weights, no keys." \
        "  2. Budgets — per-provider-key caps; show a blocked over-limit call." \
        "  3. Observability — session explorer: the EXACT trace we just made," \
        "     tokens and cost per request." \
        "  4. Declarative config — the whole org as one YAML file." \
        "  5. Routing policy — cheap open model for easy asks, frontier for" \
        "     hard ones; cost drops, quality holds." \
        "" \
        "${MAG}→ Click-through details in RUNBOOK.md, Act 4.${RST}"
pause
