#!/usr/bin/env bash
# Boot a gateway for the Playwright E2E suite: a fresh SQLite DB, migrated to
# head, serving the committed dashboard bundle in standalone mode. Invoked by
# Playwright's webServer (see web/playwright.config.ts).
set -euo pipefail

# Run from the repo root regardless of Playwright's working directory (web/).
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

config="web/e2e/otari.yml"

# A throwaway Fernet key so provider credentials can be stored during the run;
# not a secret (E2E-only, on an ephemeral DB).
export OTARI_SECRET_KEY="${OTARI_SECRET_KEY:-wdhWKyd1gwpMjxj9h4EbpW9B6pilzfrNTe0wTnwqPHg=}"

# Start each run from an empty database so onboarding and the seeded flows are
# deterministic.
rm -f web/e2e/e2e.db web/e2e/e2e.db-wal web/e2e/e2e.db-shm

uv run otari migrate --config "$config"
exec uv run otari serve --config "$config"
