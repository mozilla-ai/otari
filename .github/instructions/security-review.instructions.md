# Security Review Instructions

How to review otari for security regressions. otari is a self-hosted, OpenAI-compatible
**LLM gateway**: it authenticates callers, enforces per-user budgets, meters spend, and
proxies requests to upstream providers (via `any_llm`). Its highest-value security
properties are **tenant isolation** and **budget/billing integrity** — most of this file
is about those, because that is where this project's real incidents have been.

**Applies to paths**:
- `src/gateway/api/routes/**` — request handlers (the billable + admin surface)
- `src/gateway/api/deps.py`, `src/gateway/auth/**` — authentication
- `src/gateway/services/**` — budget, pricing, metering, log writing, tool/sandbox backends
- `src/gateway/core/config.py` — configuration and security defaults
- `src/gateway/models/entities.py`, `alembic/**` — schema and migrations
- `src/gateway/streaming.py` — streaming/SSE billing hooks

When reviewing a PR, read the diff **and** the enclosing functions — a touched billable
route that fails to settle a budget reservation on one error path is a real finding even
if the changed lines look fine.

---

## 0. Budget, billing & tenant isolation (otari-specific — read first)

This gateway holds money-adjacent state (per-user `spend`, `budget`, `reserved`) and
serves multiple tenants behind one deployment. The following are the classes of bug that
have actually shipped here; treat any new code on a billable path against this list.

### 0.1 Never trust a client-supplied identifier for authz or billing — CWE-639 (IDOR)
The OpenAI-compatible `user` field is **client-controlled** and is an end-user *tag*, not
an identity. Spend and budget must bind to the **authenticated principal**, never to a
value from the request body.

- ✅ **Check**: a non-master API key resolves to its own user only. `resolve_user_id`
  (`api/routes/_helpers.py`) is the single chokepoint — a non-master key naming a
  *different* `user` must be rejected (or bound to the key's own user under
  `reject_user_mismatch=false`), **never** charged to the named user.
- ✅ **Check**: only the **master key** may act on behalf of an arbitrary `user`.
- ✅ **Check**: every billable route resolves identity through `resolve_user_id` — no route
  reads `request.user` (or `metadata.user_id`) directly to attribute spend.
- ✅ **Check**: list/read/delete endpoints scope to the caller's own resources; a user key
  cannot read or mutate another user's keys, budget, usage, or spend.
- **Severity**: Critical

### 0.2 Enforce budgets atomically — no check-then-act (CWE-367 TOCTOU)
Budget caps must hold under concurrency. The fixed-and-correct pattern is **atomic
pre-debit reservation**, not "check the budget, call the provider, write spend later."

- ✅ **Check**: budget is enforced via `reserve_budget` (`services/budget_service.py`),
  which holds an estimate in `users.reserved` with a single conditional `UPDATE`
  (`spend + reserved + estimate <= max_budget`). A "read spend → if ok → proceed → add
  cost afterwards" shape is a TOCTOU bug — concurrent requests all pass the stale check.
- ✅ **Check**: a budget row lock (`FOR UPDATE`) is **not** released before the spend it
  guards is committed. Don't `validate(...)` then `db.rollback()` then call the provider.
- ✅ **Check**: budget resets on the request path are atomic (CAS), since the reservation
  path holds no row lock — a read-modify-write reset races at the reset boundary.
- **Severity**: High

### 0.3 Reservations must always settle — reconcile or refund on every path
A reservation that is never released leaks, permanently shrinking the user's effective
budget (`spend + reserved` creeps toward `max_budget`).

- ✅ **Check**: every billable handler calls **`reconcile_reservation`** on success and
  **`refund_reservation`** on **every** error branch (provider error, tool-iteration cap,
  sandbox/web-search unreachable, generic `except`, and `except HTTPException`).
- ✅ **Check**: the streaming path settles on completion **and** on client disconnect —
  `streaming_generator` (`streaming.py`) must call `on_incomplete` when the generator is
  closed mid-stream, or the hold leaks.
- ✅ **Check**: a new billable route added without the full reserve→reconcile/refund
  lifecycle is a finding. Prefer routing settlement through the shared helpers, not
  hand-rolled per-route.
- **Severity**: High

### 0.4 Fail closed on metering — unmetered usage is a budget bypass
Any billable request that can complete **without recording cost** is a way to get free,
uncapped usage.

- ✅ **Check**: unpriced models are rejected by default (`require_pricing`, default
  `true`) rather than served at $0. A code path that serves a model with no pricing and
  silently records no cost is a bypass.
- ✅ **Check**: streamed responses with no provider usage data are metered per
  `stream_missing_usage_policy` (not silently billed $0).
- ✅ **Check**: spend has a **single authority**. Reconciliation writes `users.spend`; the
  usage-log writer (`services/log_writer.py`) must **not** also add spend (double-charge),
  and must not be the *sole* authority (the batch writer flushes asynchronously, so the
  next request's budget check would see stale spend).
- ✅ **Check**: falsy-zero traps — `if cost:` / `if max_tokens:` treat a legitimate `0` as
  "missing." Use `is None` for "absent" vs "zero."
- **Severity**: High

### 0.5 Standalone vs hybrid mode
Local enforcement lives in the standalone branch; hybrid mode resolves credentials and
reports usage upstream.

- ✅ **Check**: `config.is_hybrid_mode` is auto-detected from a platform token. New
  billable logic that must apply locally belongs in the standalone (`db is not None`)
  branch; logic that must not double-count belongs out of the hybrid path. Verify new
  code is correct (or correctly gated) in **both** modes.
- **Severity**: Medium

### 0.6 Status-code contract
Client SDKs and the platform map gateway status codes to typed errors — keep them stable.

- ✅ **Check**: `402` means **insufficient funds / no pricing** (SDKs surface it as an
  insufficient-funds error); `403` is blocked/over-budget/forbidden-user; `404` is
  not-found. Don't repurpose a code without checking the SDK/platform mapping.
- **Severity**: Low

---

## OWASP-style checklist (backend API gateway)

### Injection — CWE-89 / CWE-78
- ✅ **Check**: all DB access uses the SQLAlchemy ORM / parameterized queries — no string-built SQL.
- ✅ **Check**: no `eval`/`exec`/`os.system`/`subprocess(..., shell=True)` with request data.
- **Severity**: Critical

### Broken authentication — CWE-287
- ✅ **Check**: API keys are stored hashed (`auth/models.py` uses SHA-256), never in plaintext; key material is not logged.
- ✅ **Check**: key validation enforces active + non-expired; the master key is compared in constant time and never logged.
- ✅ **Check**: `verify_api_key_or_master_key` (`api/deps.py`) is applied to every non-public route.
- **Severity**: Critical

### Broken access control — CWE-862 / CWE-639
- ✅ **Check**: admin/management endpoints (`/v1/users`, `/v1/budgets`, `/v1/keys`, `/v1/pricing`, `/v1/usage`) require the **master key** (user keys → 401/403).
- ✅ **Check**: object-level authorization — see §0.1. No cross-user read/write/charge.
- **Severity**: Critical

### Sensitive data exposure — CWE-200 / CWE-532
- ✅ **Check**: no provider keys, master key, or API keys in code, logs, or error messages — config via env (`.env` gitignored).
- ✅ **Check**: **no prompt/response content or other user payloads in logs** — log opaque IDs (request id, user id), token counts, model/provider names, status. Never log `messages`, `input`, completion text, or full request bodies.
- ✅ **Check**: upstream provider errors are not leaked verbatim — routes return a generic message (e.g. "The request could not be completed by the provider"); the raw upstream error/stack is logged server-side only. (Covered by `tests/integration/test_error_detail_leakage.py`.)
- **Severity**: Critical

### SSRF & request forwarding — CWE-918
otari forwards to upstream providers and optional sandbox / web-search backends.
- ✅ **Check**: sandbox / web-search backend URLs are **operator-controlled** (`GATEWAY_SANDBOX_URL`, `GATEWAY_WEB_SEARCH_URL`) — a per-request URL override must **not** be honored (it would turn the gateway into an open HTTP client). See the threat-model comment in `api/routes/chat.py`.
- ✅ **Check**: provider/base-URL selection is not driven by unvalidated request fields.
- **Severity**: High

### Insecure deserialization — CWE-502
- ✅ **Check**: no `pickle`/`yaml.load` of untrusted data; all request bodies validated by Pydantic models with constraints (types, `min_length`, bounds).
- **Severity**: Critical

### Security misconfiguration — CWE-16
- ✅ **Check**: debug mode off in production; docs endpoints gated as intended (`enable_docs`).
- ✅ **Check**: CORS is not `allow_origins=["*"]` for a deployment that uses cookauth; review `main.py` CORS.
- ✅ **Check**: new config flags that affect security **fail closed by default** and are validated at load (reject unknown values), e.g. the `stream_missing_usage_policy` validator in `core/config.py`.
- **Severity**: High

### Rate limiting & abuse — CWE-770
- ✅ **Check**: rate limiting (`rate_limit.py`) applies to billable endpoints; per-user and/or per-key limits are enforced where configured.
- **Severity**: Medium

### Components with known vulnerabilities — CWE-1035
- ✅ **Check**: dependencies current (Dependabot); no known CVEs. Run `uv pip ... audit` / `pip-audit` when adding deps.
- **Severity**: Varies

### Insufficient logging & monitoring — CWE-778
- ✅ **Check**: security-relevant events (auth failures, budget-exceeded, blocked users) are observable (metrics/logs) **without** user content.
- **Severity**: Medium

---

## LLM & prompt-injection security
otari relays tool outputs, MCP server responses, sandbox results, and web-search content
back into model context. All of these are **untrusted**.

- ✅ **Check**: tool/MCP/web-search/sandbox outputs are treated as data, not instructions — they may contain adversarial content attempting to override the system prompt or exfiltrate other tools' results.
- ✅ **Check**: the MCP tool loop bounds iterations (`max_tool_iterations` / cap) so a malicious tool can't drive unbounded provider calls (also a budget concern — see §0.2/§0.4 on sizing the reservation for the worst case).
- ✅ **Check**: no raw dumps of external objects into prompts; format and bound only the fields needed.
- **Severity**: High (CWE-74 is the closest catalog entry; there is no dedicated prompt-injection CWE.)

---

## Schema & migration safety (Alembic)
- ✅ **Check**: a model change in `models/entities.py` ships with a matching migration in `alembic/versions/`, chained to the current head.
- ✅ **Check**: new non-nullable columns have a `server_default` (existing rows) — e.g. `users.reserved` defaults to `0`.
- ✅ **Check**: every FK to `users.user_id` has an explicit `ondelete` policy and account deletion leaves no orphaned billable rows (`tests/integration/test_user_delete_preserve_logs.py` is the template — usage logs are intentionally preserved via `SET NULL`).
- ✅ **Check**: a reversible `downgrade()`.
- **Severity**: Medium

---

## Severity guidelines
- **Critical** — auth bypass, cross-user data access or charging (IDOR), injection/RCE, hardcoded secrets.
- **High** — budget bypass / overspend (unmetered usage, TOCTOU, reservation leak), SSRF, prompt injection via unsanitized external data, sensitive-data exposure.
- **Medium** — missing rate limit, info disclosure (non-sensitive), mode-gating gaps, migration hazards.
- **Low** — best-practice gaps, status-code contract drift, minor misconfig.

## Finding format
```markdown
## [Severity] — [CWE-XXX] [Title]
**File**: `src/gateway/.../file.py:line`
**Description**: what is wrong and why.
**Proof of concept**: concrete request/state → wrong outcome (e.g. "ALICE key with body {"user":"bob"} → bob's spend increases").
**Impact**: what an attacker achieves (free usage, drained budget, cross-tenant access...).
**Recommendation**: the fix, ideally with a snippet.
```

## Validating a security fix
1. Add a test that reproduces the issue and proves the fix (service-level and/or an HTTP/route test).
2. `uv run ruff check` and `uv run mypy` (mypy checks `tests/` too — annotate test helpers).
3. `uv run pytest tests/unit tests/integration` (integration spins up Postgres via testcontainers).
4. `uv run python scripts/generate_openapi.py --check` if you changed any response/request model.

## References
- [OWASP Top 10](https://owasp.org/www-project-top-ten/) · [OWASP API Security Top 10](https://owasp.org/API-Security/) · [CWE Top 25](https://cwe.mitre.org/top25/)
