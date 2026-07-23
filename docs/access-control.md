# Access control: users, keys, and budgets

Standalone Otari decides three things about every request: who is calling (the **user**), what credential they presented (the **API key**), and whether they have room to spend (the **budget**). This guide is a task-oriented tour of those three, with the management endpoints that drive them. Everything here is standalone-only and authenticated with the master key; hybrid mode delegates identity and spend to otari.ai.

All management endpoints require the master key. Send it as `Otari-Key: <master-key>` or `Authorization: Bearer <master-key>`. The same actions are available in the dashboard's **Access** section; see the [Admin dashboard guide](dashboard.md).

## How the pieces fit

- A **user** is the identity that spend and usage attach to. A user carries an optional default model allow-list and an optional budget.
- An **API key** is a credential a client sends to Otari. Each key belongs to a user. A key can narrow which models it may call.
- A **budget** is a spending limit with an optional reset period. It is a per-user cap; assign it to one user or share it across many.

A request is authenticated to a key, the key resolves to its user, the user's budget is checked and reserved before the provider call, and the usage is billed to that user afterward.

## Users

Create a user, optionally with a default model allow-list and a budget:

```bash
curl -X POST http://localhost:8000/v1/users \
  -H "Otari-Key: <master-key>" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "alice",
    "alias": "Alice (research)",
    "budget_id": "<budget-id>",
    "allowed_models": ["openai:gpt-4o-mini", "anthropic:*"]
  }'
```

- `user_id` is the stable identifier you choose; it is what spend and usage key on.
- `allowed_models` is the default access-list the user's keys inherit. `null` (or omitted) means any model, `[]` denies everything, and a list restricts to canonical `instance:model` entries, with `instance:*` and `instance:prefix*` wildcards.
- `blocked: true` stops the user from making requests without deleting anything; their calls are rejected until you unblock them.

Manage users with `GET /v1/users`, `GET /v1/users/{user_id}`, `PATCH /v1/users/{user_id}` (update alias, budget, `blocked`, or `allowed_models`), and `DELETE /v1/users/{user_id}`. A user's response includes `spend` and `reserved` (in-flight spend held by accepted but not-yet-settled requests); the committed total is `spend + reserved`. `GET /v1/users/{user_id}/usage` returns that user's request log.

### The default user

A key created with no `user_id` is bound to a shared user called `default`, created on first use. All such keys share one identity, so they share budget, usage, and files. Give a key an explicit `user_id` whenever you want to track or cap it separately.

## API keys

Create a key for a user. The plaintext key (a `gw-...` value) is returned once and never again; store it immediately.

```bash
curl -X POST http://localhost:8000/v1/keys \
  -H "Otari-Key: <master-key>" \
  -H "Content-Type: application/json" \
  -d '{
    "key_name": "alice-laptop",
    "user_id": "alice",
    "expires_at": "2026-12-31T23:59:59Z",
    "allowed_models": ["openai:gpt-4o-mini"]
  }'
```

- `expires_at` is an optional expiry; omit it for a key that never expires. Expired keys are rejected.
- `allowed_models` restricts this one key. The resolution is an override, not an intersection: a key's own list wins when set, a key with no list of its own inherits its user's default, and no list anywhere means unrestricted. A key can only narrow, never widen: creating or updating a key with a list broader than its user's default is rejected with `400`.
- The listing endpoints never return the plaintext again; they show only a `key_prefix` fingerprint (the key's leading characters).

Manage keys with `GET /v1/keys`, `GET /v1/keys/{key_id}`, `PATCH /v1/keys/{key_id}` (rename, toggle `is_active`, change expiry or `allowed_models`), and `DELETE /v1/keys/{key_id}`. To replace a key's secret without changing its identity or settings, use `POST /v1/keys/{key_id}/rotate`; it returns a new plaintext once and invalidates the old secret.

### Requests that name another user

By default a non-master key that names a `user` other than its own in the request body is rejected with `403`. This is the `reject_user_mismatch` setting (default `true`). Set it to `false` when a trusted client (for example Claude Code, which attaches its own `user_id`) must be allowed to pass a different label; spend is still bound to the key's own user. The master key may always bill an arbitrary user.

## Budgets

A budget is a spending cap with an optional reset period. `max_budget` is the limit **per user**, so a budget shared by several users caps each of them at that amount rather than in aggregate.

```bash
curl -X POST http://localhost:8000/v1/budgets \
  -H "Otari-Key: <master-key>" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "daily-10",
    "max_budget": 10.0,
    "budget_duration_sec": 86400
  }'
```

- `max_budget` is the per-user ceiling in your pricing currency. Otari reserves an estimated cost before each call and reconciles it after, so a request that would exceed the cap is rejected before it runs.
- `budget_duration_sec` is the reset period in seconds (for example `86400` for daily, `604800` for weekly). Omit it for a cap that never resets. On each period boundary the user's spend rolls back to zero and a reset is recorded.

Assign a budget to a user by setting `budget_id` on the user (at create time or via `PATCH /v1/users/{user_id}`). Manage budgets with `GET /v1/budgets`, `GET /v1/budgets/{budget_id}`, `PATCH /v1/budgets/{budget_id}`, and `DELETE /v1/budgets/{budget_id}`. A budget's response rolls up the users assigned to it: `user_count`, `total_spend`, and `total_reserved`. `GET /v1/budgets/{budget_id}/reset-logs` returns the per-user reset history.

The enforcement strategy is configurable with `OTARI_BUDGET_STRATEGY` (`for_update` row-lock, `cas` compare-and-swap, or `disabled`); see [Configuration](configuration.md).

## See also

- [Admin dashboard](dashboard.md): the same users, keys, and budgets in the browser UI.
- [Configuration](configuration.md): `reject_user_mismatch`, `budget_strategy`, and related settings.
- [API reference](api-reference.md): the full endpoint and schema listing.
