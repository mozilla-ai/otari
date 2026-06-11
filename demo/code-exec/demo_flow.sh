#!/usr/bin/env bash
# Walkthrough for code execution against the OSS gateway.
#
# The tool-array `type` says WHO runs the code — that's the whole point:
#   * {"type": "otari_code_execution"}        — the gateway runs it in its own
#                                               sandbox container (consistent
#                                               libs across every provider)
#   * {"type": "code_interpreter"}            — OpenAI runs it natively
#   * {"type": "code_execution_20250825"}     — Anthropic runs it natively
#
# The provider-named keywords are passed straight through to the provider; the
# gateway still does routing, observability, and billing around them. This
# demo shows both: a gateway-managed flow and a native-passthrough flow.
#
# Usage:
#   ./demo_flow.sh                                  # runs every provider that has credentials
#   ./demo_flow.sh --anthropic                      # subset of providers, in the given order
#   ./demo_flow.sh --openai --llamafile             # multiple flags compose
#
# Provider preconditions (the script checks each before running):
#   --anthropic   needs ANTHROPIC_API_KEY in .env
#   --openai      needs OPENAI_API_KEY in .env
#   --llamafile   needs a llamafile server reachable from the gateway container
#                 (LLAMAFILE_API_BASE; default http://host.docker.internal:8080/v1)
#                 If no server is already running and $LLAMAFILE_BIN points at
#                 an executable *.llamafile, the script will boot it on
#                 :8080 and kill it on exit. Otherwise it prints a download
#                 command and skips the --llamafile leg.

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OTARI_ROOT="$(cd "$HERE/../.." && pwd)"
ASK="$HERE/ask.sh"

if [[ -f "$HERE/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$HERE/.env"
  set +a
fi

# Resolve compose service -> container id (project-name-agnostic so the demo
# survives renaming the repo directory).
OTARI_CONTAINER=$(cd "$OTARI_ROOT" && docker compose ps -q otari 2>/dev/null | head -1 || true)

BOLD=$'\e[1m'; DIM=$'\e[2m'; YEL=$'\e[33m'; CYN=$'\e[36m'; GRN=$'\e[32m'; RED=$'\e[31m'; RST=$'\e[0m'

# ───── llamafile auto-start ───────────────────────────────────────────────
# Tracked across the whole script so the EXIT trap can clean up.
_LLAMAFILE_PID=""
_cleanup_llamafile() {
  if [[ -n "$_LLAMAFILE_PID" ]] && kill -0 "$_LLAMAFILE_PID" 2>/dev/null; then
    echo "${DIM}stopping auto-started llamafile (pid $_LLAMAFILE_PID)…${RST}"
    kill "$_LLAMAFILE_PID" 2>/dev/null || true
  fi
}
trap _cleanup_llamafile EXIT

# Try to boot a llamafile on port 8080 and wait until $probe/models is up.
# $probe is the URL the preflight check uses (e.g. http://127.0.0.1:8080/v1).
#
# We only auto-start when the operator has pointed $LLAMAFILE_BIN at an
# executable. Scanning the home directory would be a guess that breaks
# on every machine that doesn't match the maintainer's layout, so the
# fallback is to print a download recipe and let the user opt in.
_autostart_llamafile() {
  local probe="$1"
  if [[ -z "${LLAMAFILE_BIN:-}" || ! -x "${LLAMAFILE_BIN:-/nonexistent}" ]]; then
    cat <<EOF
${DIM}  No llamafile is reachable and \$LLAMAFILE_BIN is not set.
  To enable the --llamafile leg, download a llamafile once and point
  \$LLAMAFILE_BIN at it:

      curl -L -o /tmp/qwen3-4b.llamafile \\
          https://huggingface.co/mozilla-ai/Qwen3-4B-llamafile/resolve/main/Qwen_Qwen3-4B-Q5_K_M.llamafile
      chmod +x /tmp/qwen3-4b.llamafile
      export LLAMAFILE_BIN=/tmp/qwen3-4b.llamafile

  Then re-run this script. Subsequent runs reuse the same file.${RST}
EOF
    return 1
  fi
  echo "${DIM}auto-starting llamafile from ${LLAMAFILE_BIN##*/} (≈30s warmup)…${RST}"
  # `-ngl 0` keeps CPU-only inference predictable across machines; users who
  # want GPU offload can pre-start their own llamafile and skip auto-start.
  # `--alias local-llamafile` sets the model id surfaced via /v1/models —
  # newer llama.cpp builds leave it empty by default, which breaks the
  # demo's model-id lookup.
  "$LLAMAFILE_BIN" --server --port 8080 --host 0.0.0.0 -ngl 0 \
    --alias local-llamafile \
    > /tmp/llamafile-demo.log 2>&1 &
  _LLAMAFILE_PID=$!
  for _ in $(seq 1 90); do
    if curl -sf "${probe}/models" >/dev/null 2>&1; then
      echo "${DIM}  llamafile ready (pid $_LLAMAFILE_PID)${RST}"
      return 0
    fi
    # If the process died, fail fast instead of timing out 90s later.
    if ! kill -0 "$_LLAMAFILE_PID" 2>/dev/null; then
      echo "${YEL}  llamafile exited early — see /tmp/llamafile-demo.log${RST}"
      _LLAMAFILE_PID=""
      return 1
    fi
    sleep 1
  done
  echo "${YEL}  llamafile didn't become ready in 90s — see /tmp/llamafile-demo.log${RST}"
  return 1
}

# Provider selection ────────────────────────────────────────────────────────
PROVIDERS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --anthropic) PROVIDERS+=("anthropic"); shift ;;
    --openai)    PROVIDERS+=("openai");    shift ;;
    --llamafile) PROVIDERS+=("llamafile"); shift ;;
    -h|--help)
      # Print the leading comment block (everything between the shebang and
      # the first blank line). Stops there so inline `# …` comments later
      # in the file don't bleed into the help text.
      awk 'NR==1 && /^#!/ {next} /^$/ {exit} /^# ?/ {sub(/^# ?/, ""); print}' "$0"
      exit 0
      ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
done
# Default: all providers, in a stable order.
if [[ ${#PROVIDERS[@]} -eq 0 ]]; then
  PROVIDERS=(anthropic openai llamafile)
fi

# Preflight: drop providers without satisfied prereqs and warn the operator.
ENABLED=()
for p in "${PROVIDERS[@]}"; do
  case "$p" in
    anthropic)
      if [[ -n "${ANTHROPIC_API_KEY:-}" && "$ANTHROPIC_API_KEY" != *REPLACE_ME* ]]; then
        ENABLED+=("anthropic")
      else
        echo "${YEL}skipping --anthropic: ANTHROPIC_API_KEY not set in .env${RST}"
      fi
      ;;
    openai)
      if [[ -n "${OPENAI_API_KEY:-}" && "$OPENAI_API_KEY" != *REPLACE_ME* ]]; then
        ENABLED+=("openai")
      else
        echo "${YEL}skipping --openai: OPENAI_API_KEY not set in .env${RST}"
      fi
      ;;
    llamafile)
      base=${LLAMAFILE_API_BASE:-http://host.docker.internal:8080/v1}
      # The compose-mounted base points at host.docker.internal from inside
      # the gateway container; on the host, the same llamafile is reachable
      # at 127.0.0.1 — translate for the local reachability check.
      probe=${base/host.docker.internal/127.0.0.1}
      if curl -sf "${probe}/models" >/dev/null 2>&1; then
        ENABLED+=("llamafile")
      elif _autostart_llamafile "$probe"; then
        ENABLED+=("llamafile")
      else
        echo "${YEL}skipping --llamafile: no llamafile reachable at $probe and none to auto-start${RST}"
      fi
      ;;
  esac
done

if [[ ${#ENABLED[@]} -eq 0 ]]; then
  echo "${RED}no providers available. Set ANTHROPIC_API_KEY / OPENAI_API_KEY in .env"
  echo "or start a llamafile on http://localhost:8080.${RST}"
  exit 1
fi

# Provider → (model, tool-type, ask.sh extra flags) ─────────────────────────
provider_model() {
  case "$1" in
    anthropic) echo "anthropic:claude-sonnet-4-6" ;;
    openai)    echo "openai:gpt-4o-mini" ;;
    llamafile)
      local base probe raw model_id
      base=${LLAMAFILE_API_BASE:-http://host.docker.internal:8080/v1}
      probe=${base/host.docker.internal/127.0.0.1}
      raw=$(curl -sS --fail "${probe}/models" 2>&1) || {
        echo "ERROR_LOOKUP" >&2
        echo "${YEL}  failed to query ${probe}/models — output was: ${raw:0:200}${RST}" >&2
        return 1
      }
      # llama.cpp's /v1/models has historically been either:
      #   {"object":"list","data":[{"id":"…"}]}   (openai-compatible)
      #   {"data":[{"id":"…"}]}                   (newer builds)
      # Either way we pull data[0].id. Print the parse error to stderr so
      # the demo's run loop can flag the problem instead of silently
      # producing a `llamafile:` (empty) model string.
      model_id=$(printf '%s' "$raw" | python3 -c '
import json, sys
try:
    d = json.loads(sys.stdin.read() or "null")
    data = (d or {}).get("data") or []
    if not data:
        sys.stderr.write("models response had no data[0]\n")
        sys.exit(1)
    print(data[0].get("id", ""))
except Exception as e:
    sys.stderr.write(f"parse error: {e}\n")
    sys.exit(1)
') || {
        echo "${YEL}  /v1/models response was not the shape we expected — raw: ${raw:0:200}${RST}" >&2
        return 1
      }
      # llama.cpp's /v1/models returns id="" when no --alias was passed.
      # Fall back to a placeholder — llamafile only serves one model at a
      # time and accepts any value in the request's `model` field anyway.
      if [[ -z "$model_id" ]]; then
        model_id="local-llamafile"
      fi
      echo "llamafile:$model_id"
      ;;
  esac
}
# Native code-execution keyword each provider understands server-side. Empty
# for providers that have no native sandbox (open-weight llamafile) — those
# only work in the gateway-managed scenario.
provider_native_tool_type() {
  case "$1" in
    anthropic) echo "code_execution_20250825" ;;  # Anthropic's native versioned shape
    openai)    echo "code_interpreter" ;;          # OpenAI's native shape
    llamafile) echo "" ;;                          # no native sandbox
  esac
}
provider_extra() {
  # Llamafile doesn't stream tool calls; force non-streaming for that one.
  [[ "$1" == "llamafile" ]] && echo "--no-stream" || echo ""
}

pause() { echo; read -r -p "${DIM}  press Enter…${RST}" _; echo; }

present() {
  local title="$1"; shift
  echo
  echo "${BOLD}${YEL}═══════════════════════════════════════════════════════════════════════${RST}"
  echo "${BOLD}${YEL}  $title${RST}"
  echo "${BOLD}${YEL}═══════════════════════════════════════════════════════════════════════${RST}"
  for line in "$@"; do
    echo "${DIM}  $line${RST}"
  done
  echo
}

architecture_diagram() {
  cat <<EOF
${BOLD}what's running${RST} (all via \`docker compose up\` in the gateway repo)

                ┌──────────────────────────┐
                │  Client (./ask.sh, curl, │
                │  your app, an SDK)       │
                └─────────────┬────────────┘
                              │
                  POST /v1/chat/completions
                              │
                              ▼
            ┌──────────────────────────────────┐         ┌─────────────┐
            │   OSS Gateway                    │ ──────▶ │  Postgres   │
            │                                  │  state  └─────────────┘
            │   tool-use loop:                 │
            │    • otari_code_execution →      │
            │      gateway runs it in sandbox  │
            │    • code_interpreter /          │
            │      code_execution_<ver> →      │
            │      passed through to provider  │
            │    • stream SSE to client        │
            └────┬───────────────────────┬─────┘
                 │                       │
       Provider chat API        sandbox HTTP API
       (+ native code exec)     (otari_code_execution only)
                 │                       │
                 ▼                       ▼
       ┌─────────────────┐    ┌──────────────────────────┐
       │  Any model      │    │  Sandbox container       │
       │  any-llm        │    │  Python REPL,            │
       │  routes to      │    │  pandas/numpy/scipy …    │
       └─────────────────┘    └──────────────────────────┘
EOF
}

show_request_shapes() {
  local query="$1"
  cat <<EOF
${BOLD}the keyword decides who runs the code${RST} — look at the body, you know:

  ${DIM}# Gateway-managed — the gateway's own sandbox runs it:${RST}
  ${CYN}{${RST}
  ${CYN}  "messages": [{ "role": "user", "content": "$query" }],${RST}
  ${GRN}  "tools": [{ "type": "otari_code_execution" }]${RST}
  ${CYN}}${RST}

  ${DIM}# Native passthrough — OpenAI runs it in its own Code Interpreter:${RST}
  ${CYN}{${RST}
  ${CYN}  "messages": [{ "role": "user", "content": "$query" }],${RST}
  ${YEL}  "tools": [{ "type": "code_interpreter" }]${RST}
  ${CYN}}${RST}

  ${DIM}# Native passthrough — Anthropic runs it in its own sandbox:${RST}
  ${CYN}{${RST}
  ${CYN}  "messages": [{ "role": "user", "content": "$query" }],${RST}
  ${YEL}  "tools": [{ "type": "code_execution_20250825" }]${RST}
  ${CYN}}${RST}

  ${DIM}otari_* → gateway sandbox.  provider-named → provider's sandbox.${RST}
  ${DIM}Either way the gateway does the routing, observability, and billing.${RST}
EOF
}

show_what_llm_receives() {
  local query="$1"

  echo "${BOLD}A) bare request — gateway uses defaults${RST}"
  echo
  printf "${CYN}{${RST}\n"
  printf "${CYN}  \"messages\": [{ \"role\": \"user\", \"content\": \"%s\" }],${RST}\n" "$query"
  printf "${GRN}  \"tools\": [{ \"type\": \"otari_code_execution\" }]${RST}\n"
  printf "${CYN}}${RST}\n"
  echo
  echo "${DIM}  system message the gateway prepends:${RST}"
  docker exec $OTARI_CONTAINER python -c "
import sys
sys.path.insert(0, '/app/src')
from gateway.services.mcp_loop import inject_purpose_hints
from gateway.services.sandbox_backend import SandboxBackend

backend = SandboxBackend(sandbox_url='http://sandbox:8080')
print(inject_purpose_hints(
    [{'role': 'user', 'content': sys.argv[1]}],
    backend.purpose_hints(),
)[0]['content'])
" "$query" 2>&1 | sed "s/^/    ${GRN}/; s/$/${RST}/"

  echo
  echo "${BOLD}B) per-request override — \"purpose_hint\" on the tool entry${RST}"
  echo
  printf "${CYN}{${RST}\n"
  printf "${CYN}  \"messages\": [{ \"role\": \"user\", \"content\": \"%s\" }],${RST}\n" "$query"
  printf "${GRN}  \"tools\": [{${RST}\n"
  printf "${GRN}    \"type\": \"otari_code_execution\",${RST}\n"
  printf "${YEL}    \"purpose_hint\": \"Only use code_execution for math involving large numbers.\"${RST}\n"
  printf "${GRN}  }]${RST}\n"
  printf "${CYN}}${RST}\n"
  echo
  echo "${DIM}  system message the gateway prepends:${RST}"
  docker exec $OTARI_CONTAINER python -c "
import sys
sys.path.insert(0, '/app/src')
from gateway.services.mcp_loop import inject_purpose_hints
from gateway.services.sandbox_backend import SandboxBackend

backend = SandboxBackend(
    sandbox_url='http://sandbox:8080',
    purpose_hint='Only use code_execution for math involving large numbers.',
)
print(inject_purpose_hints(
    [{'role': 'user', 'content': sys.argv[1]}],
    backend.purpose_hints(),
)[0]['content'])
" "$query" 2>&1 | sed "s/^/    ${YEL}/; s/$/${RST}/"

  echo
  echo "${DIM}Knobs that shape the system message — priority order:${RST}"
  echo
  echo "${DIM}  Per-tool hint (e.g. 'Use this for math'):${RST}"
  echo "${DIM}    1. tools[i].purpose_hint              (per-request)${RST}"
  echo "${DIM}    2. OTARI_SANDBOX_PURPOSE_HINT       (env, per-deployment)${RST}"
  echo "${DIM}    3. built-in default${RST}"
  echo
  echo "${DIM}  List header (e.g. 'Prefer MCP tools over code_execution'):${RST}"
  echo "${DIM}    1. tools_header (top-level request field)  (per-request)${RST}"
  echo "${DIM}    2. OTARI_TOOLS_HEADER                     (env, per-deployment)${RST}"
  echo "${DIM}    3. 'You have access to the following tools:'  (built-in)${RST}"
}

# ───────────────────────────────────────────────────────────────────────
architecture_diagram
pause


# ───────────────────────────────────────────────────────────────────────
QUERY="Compute 23! and 50! using code_execution. Show both."

present "Under the hood: what the LLM actually receives" \
        "When the request asks for otari_code_execution, the gateway adds:" \
        " (a) a system message naming each tool source and its purpose hint" \
        " (b) a tools[] entry for the 'code_execution' function the model calls" \
        "The client sends otari_code_execution; the gateway injects the rest." \
        "(On a tool call the gateway runs the code in its own sandbox.)"
show_what_llm_receives "$QUERY"
pause


# ───────────────────────────────────────────────────────────────────────
present "Two ways to run code through the gateway" \
        "otari_code_execution → the gateway's sandbox runs it." \
        "code_interpreter / code_execution_<date> → passed through to the" \
        "provider, which runs it natively. The keyword alone tells you which."
show_request_shapes "$QUERY"
pause


# ───────────────────────────────────────────────────────────────────────
# Scenario 1: gateway-managed code execution (otari_code_execution).
N=${#ENABLED[@]}
present "1) Gateway-managed code execution — ${N} provider$( [[ $N -gt 1 ]] && echo s )" \
        "Every provider uses the SAME keyword (otari_code_execution) and the" \
        "SAME sandbox — identical Python libs regardless of which model ran." \
        "Watch the 'sandbox saw' lines: the gateway's container does the work."
for p in "${ENABLED[@]}"; do
  if ! model=$(provider_model "$p"); then
    echo "${YEL}── $p ── skipped (couldn't resolve a model id)${RST}"
    continue
  fi
  extra=$(provider_extra "$p")

  echo
  echo "${BOLD}${GRN}── $p ── model=$model tool-type=otari_code_execution${RST}"
  # shellcheck disable=SC2086
  "$ASK" --model "$model" --tool-type "otari_code_execution" $extra "$QUERY"
done
pause


# ───────────────────────────────────────────────────────────────────────
# Scenario 2: native provider code execution via the gateway. Only providers
# with a native sandbox participate (llamafile is open-weight, no native
# sandbox — it only works in scenario 1).
NATIVE=()
for p in "${ENABLED[@]}"; do
  [[ -n "$(provider_native_tool_type "$p")" ]] && NATIVE+=("$p")
done

present "2) Native provider code execution — via the gateway" \
        "Same gateway, but the keyword is the provider's own (code_interpreter," \
        "code_execution_<date>). The gateway passes it through untouched — the" \
        "PROVIDER runs the code. The gateway still routes, observes, and bills." \
        "Note: the 'sandbox saw' lines stay quiet — the gateway sandbox is idle."
if [[ ${#NATIVE[@]} -eq 0 ]]; then
  echo "${YEL}  no providers with a native sandbox are enabled (need --openai or --anthropic).${RST}"
else
  for p in "${NATIVE[@]}"; do
    if ! model=$(provider_model "$p"); then
      echo "${YEL}── $p ── skipped (couldn't resolve a model id)${RST}"
      continue
    fi
    tool_type=$(provider_native_tool_type "$p")
    extra=$(provider_extra "$p")

    echo
    echo "${BOLD}${GRN}── $p ── model=$model tool-type=$tool_type (provider runs it)${RST}"
    # shellcheck disable=SC2086
    "$ASK" --model "$model" --tool-type "$tool_type" $extra "$QUERY"
  done
fi
pause


# ───────────────────────────────────────────────────────────────────────
echo
echo "${BOLD}${GRN}═══════════════════════════════════════════════════════════════════════${RST}"
echo "${BOLD}${GRN}  fin — questions?${RST}"
echo "${BOLD}${GRN}═══════════════════════════════════════════════════════════════════════${RST}"
echo
echo "${DIM}  otari_code_execution → one sandbox, consistent libs, any model.${RST}"
echo "${DIM}  provider-named keyword → the provider's own sandbox, gateway proxies.${RST}"
