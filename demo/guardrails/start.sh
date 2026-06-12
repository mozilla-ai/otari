#!/usr/bin/env bash
# Bring up the OSS gateway demo with guardrails (gateway + anyguardrails +
# postgres) using the keys and ports configured in this folder's .env. Loads
# .env via docker-compose's --env-file so secrets never live in shell history.
#
#   ./start.sh                 # default: PIGuard via the encoderfile container (per-arch image)
#   ./start.sh --in-process    # InjecGuard in-process via HuggingFace (no encoderfile container)
#   ./start.sh -d              # detached; extra flags pass through to `compose up`

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OTARI_ROOT="$(cd "$HERE/../.." && pwd)"

ENV_FILE="$HERE/.env"
if [[ ! -f "$ENV_FILE" ]]; then
  echo "missing $ENV_FILE — copy .env.example to .env and fill in your keys." >&2
  exit 1
fi

if grep -qE '^[A-Z_]+=.*REPLACE_ME' "$ENV_FILE"; then
  echo "$ENV_FILE has an uncommented REPLACE_ME — fill in your real key before starting." >&2
  exit 1
fi

# Default backend is the encoderfile container (PIGuard). `--in-process` opts
# out to the in-process HuggingFace path (InjecGuard, no extra container).
INPROCESS=0
PASSTHRU=()
for arg in "$@"; do
  case "$arg" in
    --in-process) INPROCESS=1 ;;
    *) PASSTHRU+=("$arg") ;;
  esac
done

# Both backends live in the single `guardrails` profile. Default brings up
# everything (incl. the encoderfile container); --in-process lists services
# explicitly to leave the encoderfile container out.
SERVICES=()
if [[ "$INPROCESS" == 1 ]]; then
  export GUARDRAILS_CONFIG_PATH="./demo/guardrails/guardrails-service.yaml"
  SERVICES=(gateway postgres anyguardrails)
  echo "ℹ --in-process: InjecGuard via HuggingFace (no encoderfile container)."
else
  export GUARDRAILS_CONFIG_PATH="./demo/guardrails/guardrails-encoderfile-service.yaml"
  # Encoderfile images are published per-arch (arch in the tag name). Pick the
  # one matching this host. Model + version are overridable via env.
  case "$(uname -m)" in
    arm64|aarch64) ef_triple="aarch64-linux-gnu" ;;
    x86_64|amd64)  ef_triple="x86_64-linux-gnu" ;;
    *) echo "unsupported arch $(uname -m) for the encoderfile image" >&2; exit 1 ;;
  esac
  ef_model="${GUARDRAILS_ENCODERFILE_MODEL:-piguard}"
  ef_tag="${GUARDRAILS_ENCODERFILE_VERSION:-v0.6.2}"
  export OTARI_ENCODERFILE_IMAGE="docker.io/mzdotai/${ef_model}.${ef_triple}-encoderfile:${ef_tag}"
  echo "ℹ encoderfile (default): ${ef_model} via the encoderfile container (${OTARI_ENCODERFILE_IMAGE})."
fi

cd "$OTARI_ROOT"

# Pre-pull the guardrails app image; a 401 here means the Docker Hub repo is
# still private (it must be public for external users) or you're not logged in.
agr_image="${OTARI_ANYGUARDRAILS_IMAGE:-docker.io/mzdotai/otari-any-guardrail-container:latest}"
if ! docker image inspect "$agr_image" >/dev/null 2>&1 && ! docker pull "$agr_image" >/dev/null 2>&1; then
  echo "⚠ couldn't pull $agr_image — the Docker Hub repo may be private, or set OTARI_ANYGUARDRAILS_IMAGE to a tag you have." >&2
fi

# If we're on a non-main branch, the published `mzdotai/otari:latest` may not
# have the unreleased gateway code on this branch. Surface the build recipe.
branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "")
if [[ -n "$branch" && "$branch" != "main" ]]; then
  cat <<EOF
ℹ branch '$branch' detected — if it has unreleased gateway code, build the image locally first:
    docker build -t mzdotai/otari:latest .
  (skip this if you've already built, or if the branch's changes are already in mzdotai/otari:latest.)

EOF
fi

exec docker compose --env-file "$ENV_FILE" --profile guardrails up \
  ${SERVICES[@]+"${SERVICES[@]}"} ${PASSTHRU[@]+"${PASSTHRU[@]}"}
