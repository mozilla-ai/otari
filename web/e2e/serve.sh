#!/usr/bin/env bash
# Boot a gateway for the Playwright E2E suite: a fresh SQLite DB, migrated to
# head, serving the built dashboard bundle in standalone mode. `pnpm run e2e`
# builds it first, so the suite always runs against fresh output. Invoked by
# Playwright's webServer (see web/playwright.config.ts).
set -euo pipefail

# Run from the repo root regardless of Playwright's working directory (web/).
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

config="web/e2e/otari.yml"

# otari.yml sets no `mode`, so the mode is derived from the platform token, and a
# token in the shell of whoever is running this would silently boot the gateway
# these specs sign in to as a hybrid one, where every management route 404s. Now
# that the suite boots a hybrid gateway of its own beside this one
# (e2e/serve-hybrid.sh), the standalone half says so rather than inheriting it.
unset OTARI_AI_TOKEN

# A throwaway Fernet key so provider credentials can be stored during the run;
# not a secret (E2E-only, on an ephemeral DB).
export OTARI_SECRET_KEY="${OTARI_SECRET_KEY:-wdhWKyd1gwpMjxj9h4EbpW9B6pilzfrNTe0wTnwqPHg=}"

# Start each run from an empty database so onboarding and the seeded flows are
# deterministic.
rm -f web/e2e/e2e.db web/e2e/e2e.db-wal web/e2e/e2e.db-shm

uv run otari migrate --config "$config"
exec uv run otari serve --config "$config"
