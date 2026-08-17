---
name: backend-standards
description: Backend conventions for the otari gateway (`src/gateway/`), async SQLAlchemy 2.0, FastAPI, budget/reservation lifecycle, Alembic migrations, config layering. Use when writing or reviewing gateway request handling, services, models, or migrations.
---

# Backend Standards: otari gateway (`src/gateway/`)

The gateway is an async FastAPI service: request handlers in `api/routes/`, business logic in
`services/`, ORM in `models/entities.py`, migrations in `alembic/versions/`. This guide is the
backend counterpart to the frontend skill and to the path-scoped review instructions in
`.github/instructions/` (performance and security). `AGENTS.md` is the source of truth for
build/test/lint commands and the two-mode architecture; read it first. This file captures the
conventions that keep new backend code correct and consistent.

## Async SQLAlchemy 2.0: the house style

Everything is async. Match the shapes already in `services/`:

```python
from sqlalchemy import select, func

# scalar list
rows = (await db.execute(select(ModelAlias))).scalars().all()

# single row (or None)
existing = (await db.execute(select(APIKey.id).limit(1))).scalar_one_or_none()

# count without loading rows
count = (await db.execute(select(func.count()).select_from(ModelPricing))).scalar_one()
```

- Use `await db.execute(select(...))` + `.scalars()` / `.scalar_one_or_none()` /
  `.scalar_one()`. Don't fetch rows to count them (`len(all())`), use `func.count()`.
- ORM columns are typed with `Mapped[...]` + `mapped_column(...)`. Follow the existing style:
  modern generics (`str | None`, `list[str]`), timezone-aware `DateTime(timezone=True)`.
- Sessions come from the `get_db` dependency in routes; non-request code uses
  `create_session()` (`core/database.py`). Don't open ad-hoc engines.

## The SQLModel half: the reconciled control plane's tables

`models/tenancy.py` (organizations, workspaces, identities, memberships) is SQLModel rather
than `entities.py`'s declarative style, because its `Create`/`Update`/`Public` schemas are the
endpoint contracts the generated dashboard client is built from. Same session, same chain, three
extra rules:

- **Wrap every column reference in `sqlmodel.col()`.** On a SQLModel class the attribute's
  static type is the annotation, so `col(Organization.slug) == slug` typechecks where
  `Organization.slug == slug` reads as `bool` and mypy rejects it. Applies to `where`,
  `order_by`, `join` conditions, and `.in_(...)`.
- **Inherit `BaseRepository`** (`repositories/base_repository.py`) for `get`/`get_all`/`create`/
  `update`/`delete`/`count`, and put tenancy repositories in `repositories/tenancy/`. Every
  repository write **flushes and never commits**: the service owns the commit boundary, because
  it is the layer that knows when a unit of work is complete.
- **Declare no `relationship()`.** Lazy loading raises `MissingGreenlet` on an `AsyncSession` at
  attribute access rather than at the query; join explicitly and return
  `(model, related)` tuples instead.

A column that needs a SQLAlchemy construct `Field()` cannot express (`use_alter`, a custom type
instance) takes an explicit `sa_column=Column(...)`, but never on a mixin: one `Column` instance
cannot attach to more than one table, so a shared mixin passes `sa_type` plus
`sa_column_kwargs` and lets SQLModel build a fresh column per model.

## Layering

- **Routes** (`api/routes/`) stay thin: parse the request, resolve identity, call a service,
  shape the response. Keep request/response Pydantic models near the handler; return typed
  models, not raw dicts; use `fastapi.status` constants.
- **Services** (`services/`, one concern per `*_service.py`) hold the business logic and own
  the DB work.
- Service-specific exceptions live beside their service (e.g. `UnsafeURLError`,
  `GuardrailsNotReachableError`). Raise `HTTPException` with a clear `detail` in the API layer;
  prefer specific exceptions (`ValueError`, `SQLAlchemyError`) over broad `except Exception`.

## The budget / reservation lifecycle is load-bearing

Billable routes hold money-adjacent state. The invariant (detailed in
[../../instructions/security-review.instructions.md](../../instructions/security-review.instructions.md#0-budget-billing--tenant-isolation-otari-specific--read-first))
is: **reserve before the provider call, then reconcile on success or refund on every error
path**, including provider errors, tool-iteration caps, unreachable sandbox/web-search,
generic `except`, and `except HTTPException`, plus streaming completion and client disconnect.
A reservation that never settles leaks and permanently shrinks the user's budget.

- Bind spend to the **authenticated principal** via `resolve_user_id`, never to the
  client-supplied `user` field.
- Enforce budgets atomically (the reservation is a single conditional `UPDATE`), not
  check-then-act. Use `is None` for "absent" vs a legitimate `0` (falsy-zero traps).
- New billable logic must be correct in **both** standalone and hybrid mode, verify which
  branch (`db is not None`) it belongs in.

## Migrations (Alembic)

- A change to `models/entities.py` ships with a matching migration in `alembic/versions/`,
  chained to the current head, in the same PR.
- New non-nullable columns need a `server_default` for existing rows (e.g. `users.reserved`
  defaults to `"0"`).
- Every foreign key needs an explicit `ondelete` policy; index it (`index=True`), see the
  performance instructions. Account deletion must leave no orphaned billable rows.
- Provide a real, reversible `downgrade()`.
- The chain runs on SQLite *and* PostgreSQL, so keep it dialect-neutral: `sa.func.now()` rather
  than a literal `now()`/`CURRENT_TIMESTAMP`, and no `ALTER TABLE ... ADD CONSTRAINT`, which
  SQLite does not have. Adding a constraint to an existing table goes through
  `op.batch_alter_table(..., copy_from=<the sa.Table>)`; `copy_from` is what keeps SQLite's
  table rebuild from dropping what reflection could not see. Verify both engines locally
  (upgrade, downgrade, upgrade): the integration suite migrates PostgreSQL only, so SQLite is
  covered only where a test asks for it (`tests/unit/test_tenancy_schema_chain.py` is the pattern).

## Config & env

`GatewayConfig` (`core/config.py`) loads `config.yml` then layers env vars, under the
user-facing `OTARI_` prefix. New security-relevant
flags **fail closed by default** and are validated at load (reject unknown values), like the
`stream_missing_usage_policy` validator. Don't read `os.getenv` directly on a hot path; route
through the config / `otari_env()`.

## Logging

- Use the module logger from `gateway.log_config` with `%s` placeholders.
- **Never log secrets or user payloads**: no API keys, no `messages`/`input`/completion text,
  no full request bodies. Log opaque ids, token counts, model/provider names, status. (The
  one sanctioned exception is the intentional one-time bootstrap key print.)

## Before you finish

- Add happy-path **and** error-path tests next to the changed behavior (unit for pure logic,
  integration for route/DB behavior; integration spins up Postgres via testcontainers).
- If you touched request/response models, run `uv run python scripts/generate_openapi.py
  --check`.
- Run `make lint` and `make typecheck` (ruff + mypy strict over `src`, `tests`, `scripts`).

## Related instructions

- [performance-review.instructions.md](../../instructions/performance-review.instructions.md): N+1, indexes, pagination limits, transaction atomicity, async efficiency.
- [security-review.instructions.md](../../instructions/security-review.instructions.md): budget/tenant isolation, auth, SSRF, prompt injection, migration safety.
