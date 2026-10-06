#!/usr/bin/env bash
# --no-project skips the project sync, which can take longer than the verifier timeout.
set -u

if ! command -v uv >/dev/null 2>&1; then
  echo "The architecture check could not run, because its Python runner is not installed."
  exit 2
fi

uv run --quiet --no-project --python ">=3.13" python scripts/check_architecture.py
status=$?

if [ "$status" -le 1 ]; then
  exit "$status"
fi
exit 2
