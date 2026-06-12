# Changelog

All notable changes to this project are documented here. This project adheres to
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Security

Fixes for four budget/billing vulnerabilities. **Standalone (self-hosted) mode
only** — platform mode resolves budgets/pricing upstream and is unaffected.

- **Cross-user budget charging / impersonation via the `user` field (IDOR).** A
  non-master API key could set `"user"` to another user and charge spend to, and
  exhaust the budget of, that user. Non-master keys are now bound to their own
  user (see breaking change below).
- **Budget overspend race (TOCTOU).** Concurrent requests could all pass the
  budget check against stale spend and collectively exceed `max_budget`. Budgets
  are now enforced with atomic pre-debit reservations.
- **Unpriced models served free and unmetered.** A model with no pricing was
  served and charged $0, bypassing the budget cap. Now rejected by default (see
  breaking change below).
- **Streaming responses without usage data were not billed.** Now metered per a
  configurable policy.

### Added

- `users.reserved` column (migration `b2f4c6d8e0a1`) holding in-flight budget
  reservations; spend reconciles to actual cost on completion.
- Config `require_pricing` (default `true`), `stream_missing_usage_policy`
  (`estimate` | `fail` | `allow_free`, default `estimate`),
  `reject_user_mismatch` (default `true`), and
  `budget_estimate_default_output_tokens`.
- Startup warning when `require_pricing` is enabled but no pricing is configured.
- `SECURITY.md`.

### Changed

- **Renamed "Otari Gateway" to "Otari" across the product surface.** The CLI is
  now `otari` (the `gateway` command stays as a legacy alias). Environment
  variables use the `OTARI_` prefix; the legacy `GATEWAY_` prefix still works as a
  fallback, and `OTARI_` wins when both are set. The OpenAPI title, the default
  SQLite filename (`otari.db`), the Docker Compose service (`otari`), and the CI
  workflow files (`otari-*.yml`) were renamed to match. The Python import package
  stays `gateway`: the `otari` distribution name on PyPI belongs to the Otari
  client SDK. Existing `GATEWAY_*` variables and the `gateway` command keep
  working, so no configuration change is required.

  Upgrade notes (on-disk state is not auto-migrated):
  - **SQLite default path renamed `otari-gateway.db` -> `otari.db`.** Standalone
    deployments that relied on the default `database_url` (no explicit value) must
    rename the existing `otari-gateway.db` file to `otari.db`, or set
    `database_url` to the old path. Otherwise the app starts against a new, empty
    database: previously issued API keys return 401 and existing users, budgets,
    and usage history are orphaned in the old file. Deployments that set
    `database_url` explicitly (for example Postgres) are unaffected.
  - **Docker Compose Postgres credentials renamed `gateway` -> `otari`.** The
    `postgres` service now initializes user/password/db as `otari`. An existing
    `postgres_data` volume was created with the old `gateway` credentials and will
    fail auth, so drop it locally (`docker compose down -v`). If you copied
    `config.example.yml` to a local `config.yml` before this change, also update
    its `database_url` to `postgresql://otari:otari@postgres:5432/otari` so it
    matches the new Postgres credentials.

### Changed (BREAKING — standalone mode)

- **`require_pricing` defaults to `true`:** requests for models with no pricing
  row are rejected with **HTTP 402**. Operators running genuinely free or
  self-hosted models must add an explicit `$0` pricing entry, or set
  `require_pricing=false`. Audio and moderation endpoints are exempt.
- **The client `user` field is no longer trusted for non-master keys:** a
  request naming a user other than the key's own is rejected with **HTTP 403**.
  Set `reject_user_mismatch=false` to instead bind spend to the key's own user
  while still forwarding `user` to the provider (OpenAI-style end-user tag) — use
  this if clients send arbitrary `user` values for abuse tracking. The master key
  may still bill an arbitrary user.
- **`stream_missing_usage_policy` defaults to `estimate`:** a streamed response
  that completes without provider usage data is billed an estimated cost rather
  than served free. Set to `allow_free` for the previous behavior.
- The usage-log writer no longer updates `users.spend`; reconciliation of the
  budget reservation is now the sole authority for spend.
