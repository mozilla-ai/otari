---
applyTo: "src/gateway/api/**/*.py,src/gateway/auth/**/*.py,src/gateway/services/**/*.py,src/gateway/core/config.py,src/gateway/models/**/*.py,src/gateway/streaming.py,alembic/versions/**/*.py"
---

# Security review instructions

Review the enclosing request or transaction, not only the changed lines.
Otari's highest-risk failures are cross-tenant access, credential exposure, and
usage that bypasses billing or leaves budget reservations stuck.

## Budget, billing, and tenant isolation

### Bind identity to authentication

The OpenAI `user` field is an untrusted provider tag. A non-master request is
always billed to the API key's user and workspace. Lenient mismatch handling may
forward a different tag, but it never changes attribution.

Every object lookup must include the tenant predicate. A row outside the
caller's organization or workspace returns 404 so its ID is not disclosed.

### Choose the correct authority

Management routes use one of two authorization shapes:

- Deployment-wide routes declare `require_deployment_operator`.
- Tenant routes authenticate with `verify_master_key`, resolve the current
  identity, and authorize it against the organization or workspace in the
  service layer.

Data-plane routes use `verify_api_key_or_master_key`. They never accept a
dashboard cookie, including a superuser session. A header master key is billed
through the deployment's default workspace, but accepting a cookie here would
let any signed-in organization member spend that default workspace's provider
credentials and budget without holding a data-plane key.
Read-only catalog routes use `verify_catalog_reader`.

Using the operator gate on a tenant route is also wrong; it prevents members
from managing resources their role permits.

### Preserve the reservation lifecycle

Budgeted request paths use `reserve_budget` before dispatch and then exactly
one of:

- `reconcile_reservation` after cost is known
- `refund_reservation` on every failure or incomplete stream

Check provider errors, tool failures, iteration limits, HTTP exceptions,
cancellation, and client disconnect. A leaked reservation permanently reduces
available budget.

Do not replace the conditional reservation update with a read, provider call,
and later write. Concurrent requests would pass the same stale check. Scoped
budget rows must be acquired and compensated in their established total order.

Some retrospective or asynchronously settled paths intentionally do not provide
a hard real-time cap. Do not broaden that exception silently.

### Meter fail closed

`require_pricing` defaults to true for budgeted token and image traffic.
Missing provider usage follows `stream_missing_usage_policy`. Reconciliation,
not the usage-log writer, is the authority for spend.

Treat zero as a value and `None` as missing. Falsy checks around cost, token
counts, or limits can create free usage.

New security-affecting configuration must default to the fail-closed behavior
and reject unknown values during config loading.

### Respect deployment mode

Standalone serves both planes, hosted serves only the control plane, and hybrid
serves the data plane without a local management database. Check new behavior in
all applicable modes and avoid reporting or charging the same usage twice.

Keep the status contract stable: 402 is insufficient funds or missing pricing;
403 is forbidden, blocked, or over budget; 404 hides absent or foreign
resources.

## Secrets and public errors

Never log or return provider keys, API keys, master keys, bearer tokens, raw
provider bodies, prompts, responses, or tool payloads.

Caller-fixable upstream 400, 404, and 422 errors may pass through only after
`redact_upstream_message` and length limiting. Credential failures, provider
billing failures, 5xx responses, and unknown failures use fixed public text.
Expanding the pass-through set is a security change.

API keys are stored as one-way SHA-256 hashes, never as plaintext. Recoverable
provider and tool credentials are encrypted, and responses expose only safe
metadata such as `last4` or `has_token`. Key validation must not echo the
submitted secret.
Sentry or other telemetry must scrub request headers and bodies.

## SSRF and untrusted model context

Provider, MCP, guardrail, sandbox, and search URLs follow their existing URL
safety policy. Credentials require HTTPS where the service contract says so.
Do not let an ordinary request turn the gateway into an unrestricted HTTP
client.

MCP results, web pages, sandbox output, and tool responses are untrusted data.
Bound their size, preserve tool allow-lists, and keep loop limits. Do not insert
external content into system instructions as trusted text.

## Schema and dependency changes

A model change needs a migration. Review defaults and backfills for populated
tables, foreign-key deletion behavior, downgrade safety, and tenant indexes.

New dependencies require lockfile review and a supply-chain check. Do not load
untrusted pickle data, unsafe YAML, or shell commands built from request input.

## Findings

Use Critical for authentication bypass, cross-tenant access or charging, remote
code execution, and exposed credentials. Use High for budget bypass,
reservation leaks, SSRF, or sensitive payload exposure. Use Medium for mode
gaps, migration hazards, and weaker abuse controls.

Each finding must name the file and line, show the reachable bad outcome, explain
impact, and recommend a concrete fix. Require a regression test that fails
before the fix.
