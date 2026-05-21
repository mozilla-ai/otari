#!/usr/bin/env bash
# Walkthrough for web search against the OSS gateway.
#
# The gateway accepts two equivalent tool-array shapes for web search:
#   * {"type": "web_search"}              — gateway short form (matches OpenAI)
#   * {"type": "web_search_20250305"}     — Anthropic, versioned (and future
#                                           web_search_20260209 etc., matched
#                                           by prefix)
#
# Both map to the same WebSearchBackend, so swapping the OpenAI/Anthropic
# SDK's `base_url` to the gateway keeps existing client code working.
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

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
GATEWAY_ROOT="$(cd "$HERE/../.." && pwd)"
ASK="$HERE/ask.sh"

if [[ -f "$HERE/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$HERE/.env"
  set +a
fi

# Resolve compose service -> container id (project-name-agnostic so the demo
# survives renaming the repo directory).
GATEWAY_CONTAINER=$(cd "$GATEWAY_ROOT" && docker compose ps -q gateway 2>/dev/null | head -1 || true)

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
        return 1
      }
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
') || return 1
      if [[ -z "$model_id" ]]; then
        model_id="local-llamafile"
      fi
      echo "llamafile:$model_id"
      ;;
  esac
}
provider_tool_type() {
  case "$1" in
    anthropic) echo "web_search_20250305" ;;  # Anthropic's native versioned shape
    openai)    echo "web_search" ;;            # OpenAI's native shape (and gateway-native)
    llamafile) echo "web_search" ;;            # gateway-native short form
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
            │    • detects web_search /        │
            │      web_search_<ver>            │
            │    • adds it to model's tools[]  │
            │    • on tool_call: HTTP to       │
            │      configured search backend   │
            │    • fetch+extract content       │
            │    • feed results back to model  │
            │    • stream SSE to client        │
            └────┬───────────────────────┬─────┘
                 │                       │
       Provider chat API        web-search HTTP API
                 │                       │
                 ▼                       ▼
       ┌─────────────────┐    ┌──────────────────────────┐
       │  Any model      │    │  SearXNG container       │
       │  any-llm        │    │  metasearch over         │
       │  routes to      │    │  DDG/mojeek/qwant/wiki   │
       └─────────────────┘    └──────────────────────────┘
EOF
}

show_engine_note() {
  cat <<EOF
${BOLD}engine selection${RST} (bundled SearXNG defaults)

  ${DIM}This demo runs a SearXNG metasearch instance over engines that don't${RST}
  ${DIM}forbid automated querying — duckduckgo, mojeek, qwant, wikipedia.${RST}
  ${DIM}Major engines (Google, Bing, Yahoo, Brave) are deliberately disabled${RST}
  ${DIM}in scripts/searxng/settings.yml. Operators who enable them should${RST}
  ${DIM}review the upstream Terms of Service first.${RST}

  ${DIM}For commercial / production use, swap the searxng container for a${RST}
  ${DIM}licensed-API backend (Tavily, Brave Search API, Exa, Linkup, Serper):${RST}
  ${DIM}any HTTP service exposing /search?format=json is a drop-in replacement.${RST}
  ${DIM}Just change GATEWAY_WEB_SEARCH_URL in the gateway's environment.${RST}
EOF
}

show_request_shapes() {
  local query="$1"
  cat <<EOF
${BOLD}two shapes the gateway accepts${RST} — same backend, different SDKs:

  ${DIM}# Gateway-native / OpenAI Responses (terse):${RST}
  ${CYN}{${RST}
  ${CYN}  "messages": [{ "role": "user", "content": "$query" }],${RST}
  ${GRN}  "tools": [{ "type": "web_search" }]${RST}
  ${CYN}}${RST}

  ${DIM}# Anthropic SDK code — versioned tool type:${RST}
  ${CYN}{${RST}
  ${CYN}  "messages": [{ "role": "user", "content": "$query" }],${RST}
  ${GRN}  "tools": [{ "type": "web_search_20250305" }]${RST}
  ${CYN}}${RST}

  ${DIM}# Optional per-tool overrides (max_results, allowed_domains, blocked_domains, purpose_hint):${RST}
  ${CYN}{${RST}
  ${GRN}  "tools": [{${RST}
  ${GRN}    "type": "web_search",${RST}
  ${YEL}    "max_results": 3,${RST}
  ${YEL}    "allowed_domains": ["docs.python.org"]${RST}
  ${GRN}  }]${RST}
  ${CYN}}${RST}
EOF
}

show_what_llm_receives() {
  local query="$1"

  echo "${BOLD}A) bare request — gateway uses defaults${RST}"
  echo
  printf "${CYN}{${RST}\n"
  printf "${CYN}  \"messages\": [{ \"role\": \"user\", \"content\": \"%s\" }],${RST}\n" "$query"
  printf "${GRN}  \"tools\": [{ \"type\": \"web_search\" }]${RST}\n"
  printf "${CYN}}${RST}\n"
  echo
  echo "${DIM}  system message the gateway prepends:${RST}"
  docker exec $GATEWAY_CONTAINER python -c "
import sys
sys.path.insert(0, '/app/src')
from gateway.services.mcp_loop import inject_purpose_hints
from gateway.services.web_search_backend import WebSearchBackend

backend = WebSearchBackend(base_url='http://searxng:8080')
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
  printf "${GRN}    \"type\": \"web_search\",${RST}\n"
  printf "${YEL}    \"purpose_hint\": \"Use web_search only for questions about events after January 2026.\"${RST}\n"
  printf "${GRN}  }]${RST}\n"
  printf "${CYN}}${RST}\n"
  echo
  echo "${DIM}  system message the gateway prepends:${RST}"
  docker exec $GATEWAY_CONTAINER python -c "
import sys
sys.path.insert(0, '/app/src')
from gateway.services.mcp_loop import inject_purpose_hints
from gateway.services.web_search_backend import WebSearchBackend

backend = WebSearchBackend(
    base_url='http://searxng:8080',
    purpose_hint='Use web_search only for questions about events after January 2026.',
)
print(inject_purpose_hints(
    [{'role': 'user', 'content': sys.argv[1]}],
    backend.purpose_hints(),
)[0]['content'])
" "$query" 2>&1 | sed "s/^/    ${YEL}/; s/$/${RST}/"

  echo
  echo "${DIM}Knobs that shape the system message — priority order:${RST}"
  echo
  echo "${DIM}  Per-tool hint (e.g. 'Use this for current events'):${RST}"
  echo "${DIM}    1. tools[i].purpose_hint                  (per-request)${RST}"
  echo "${DIM}    2. GATEWAY_WEB_SEARCH_PURPOSE_HINT        (env, per-deployment)${RST}"
  echo "${DIM}    3. built-in default${RST}"
  echo
  echo "${DIM}  List header (e.g. 'Prefer MCP tools over web_search'):${RST}"
  echo "${DIM}    1. tools_header (top-level request field)  (per-request)${RST}"
  echo "${DIM}    2. GATEWAY_TOOLS_HEADER                     (env, per-deployment)${RST}"
  echo "${DIM}    3. 'You have access to the following tools:'  (built-in)${RST}"
}

# ───────────────────────────────────────────────────────────────────────
architecture_diagram
pause


# ───────────────────────────────────────────────────────────────────────
present "Engine selection & legal landscape" \
        "Which engines the bundled SearXNG instance is configured to query," \
        "and how to swap it for a licensed-API backend in production."
show_engine_note
pause


# ───────────────────────────────────────────────────────────────────────
# Pick a query that's reliably *after* a 4B-class open-weight model's
# training cutoff. Small models with optional tools often answer from
# training memory when they think they know — which makes the demo
# silently look like web_search wasn't exercised. Asking for current
# state (top HN story, today's weather, breaking news of the day)
# forces the model to actually call the tool.
QUERY="What is the current top story on Hacker News right now? Summarize it in one sentence."

present "Under the hood: what the LLM actually receives" \
        "Two things the gateway adds before forwarding to the model:" \
        " (a) a system message naming each tool source and its purpose hint" \
        " (b) a tools[] entry for web_search" \
        "The client never sees web_search executed — the gateway runs it."
show_what_llm_receives "$QUERY"
pause


# ───────────────────────────────────────────────────────────────────────
present "Same gateway, two request shapes" \
        "The gateway accepts the keyword each SDK already emits, so swapping" \
        "the SDK's base_url to the gateway keeps existing code working."
show_request_shapes "$QUERY"
pause


# ───────────────────────────────────────────────────────────────────────
N=${#ENABLED[@]}
present "Web-search end to end — ${N} provider$( [[ $N -gt 1 ]] && echo s )" \
        "Same question, same SearXNG instance. Each call uses the SDK keyword" \
        "native to its provider — proving the gateway recognises both shapes."
for p in "${ENABLED[@]}"; do
  if ! model=$(provider_model "$p"); then
    echo "${YEL}── $p ── skipped (couldn't resolve a model id)${RST}"
    continue
  fi
  tool_type=$(provider_tool_type "$p")
  extra=$(provider_extra "$p")

  echo
  echo "${BOLD}${GRN}── $p ── model=$model tool-type=$tool_type${RST}"
  # shellcheck disable=SC2086
  "$ASK" --model "$model" --tool-type "$tool_type" $extra "$QUERY"
done
pause


# ───────────────────────────────────────────────────────────────────────
echo
echo "${BOLD}${GRN}═══════════════════════════════════════════════════════════════════════${RST}"
echo "${BOLD}${GRN}  fin — questions?${RST}"
echo "${BOLD}${GRN}═══════════════════════════════════════════════════════════════════════${RST}"
echo
echo "${DIM}  Same flow works for any model the gateway can route to —${RST}"
echo "${DIM}  open-weight or frontier. The gateway just runs the loop.${RST}"
