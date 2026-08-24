#!/usr/bin/env bash
# Boot a second gateway for the Playwright E2E suite, this one in hybrid mode:
# no database, no management API, serving the built dashboard bundle as the
# landing page a hybrid deployment shows. Invoked by Playwright's webServer
# alongside serve.sh (see web/playwright.config.ts).
set -euo pipefail

# Run from the repo root regardless of Playwright's working directory (web/).
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

# An explicit `mode: hybrid` is refused without a platform token, and the token
# is what selects the mode when none is set, so hybrid mode cannot be booted
# without one. Nothing in this suite spends it: the stand-in control plane is the
# standalone gateway next door, which authenticates nothing. Set rather than
# defaulted, for the reason serve.sh unsets it: a real token in the shell of
# whoever is running this has no use here and no business being loaded into a
# throwaway process.
export OTARI_AI_TOKEN="gw_e2e_hybrid_token"

# No `otari migrate`: a hybrid gateway opens no database.
exec uv run otari serve --config web/e2e/otari.hybrid.yml
