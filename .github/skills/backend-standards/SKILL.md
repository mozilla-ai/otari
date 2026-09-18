---
name: backend-standards
description: Backend conventions for the otari gateway (`src/gateway/`), async SQLAlchemy 2.0, FastAPI, budget/reservation lifecycle, Alembic migrations, config layering. Use when writing or reviewing gateway request handling, services, models, or migrations.
---

# Backend Standards: otari gateway (`src/gateway/`)

The gateway is an async FastAPI service: request handlers in `api/routes/`, business logic in
`services/`, queries in `repositories/`, ORM in `models/`, migrations in `alembic/versions/`.
This guide is the backend counterpart to the frontend skill and to the path-scoped review
instructions in `.github/instructions/` (architecture, performance and security). `AGENTS.md`
is the source of truth for build/test/lint commands and runtime modes; read it first. This file
captures the conventions that keep new backend code correct and consistent.

## Async SQLAlchemy 2.0: the house style

Everything is async. A query belongs in a repository (see [Layering](#layering)). Match these shapes:

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
- A request's session comes from the `get_db` dependency, and non-request code uses
  `create_session()` (`core/database.py`). Don't open ad-hoc engines.

## The SQLModel half: the reconciled control plane's tables

`models/tenancy.py` (organizations, workspaces, identities, memberships) is SQLModel rather
than the declarative `Base` style of the other domain modules, because its `Create`/`Update`/`Public` schemas are the
endpoint contracts the generated dashboard client is built from. Same session, same chain, three
extra rules:

- **Wrap every column reference in `sqlmodel.col()`.** On a SQLModel class the attribute's
  static type is the annotation, so `col(Organization.slug) == slug` typechecks where
  `Organization.slug == slug` reads as `bool` and mypy rejects it. Applies to `where`,
  `order_by`, `join` conditions, and `.in_(...)`.
- **Inherit `BaseRepository`** (`repositories/base_repository.py`) for `get`/`get_all`/`create`/
  `update`/`delete`/`count`, and put a domain's repositories in `repositories/<domain>/`. Every
  repository write **flushes and never commits**. The commit belongs to the service (see
  [Who commits](#who-commits)).
- **Declare no `relationship()`.** Lazy loading raises `MissingGreenlet` on an `AsyncSession` at
  attribute access rather than at the query; join explicitly and return
  `(model, related)` tuples instead.

A column that needs a SQLAlchemy construct `Field()` cannot express (`use_alter`, a custom type
instance) takes an explicit `sa_column=Column(...)`, but never on a mixin: one `Column` instance
cannot attach to more than one table, so a shared mixin passes `sa_type` plus
`sa_column_kwargs` and lets SQLModel build a fresh column per model.

## Layering

The backend is a modular monolith. [ARCHITECTURE.md](../../../ARCHITECTURE.md#the-modular-monolith)
names the shape and gives the layer and import rules.
[docs/domains.md](../../../docs/domains.md#the-target-shape) gives what each layer holds, how a
domain fits together and the domain test, assigns every module to its domain, and gives the steps
for moving one domain into the shape. This section adds the house style for code in those layers.

**New and moved code follows the target shape. Most existing code does not, so never copy the
module beside yours.** `SERVICE_DATABASE_IMPORT_BASELINE`, `ROUTE_DATABASE_IMPORT_BASELINE` and `FLAT_MODULE_BASELINE` in
`scripts/check_architecture.py` name the code still in the old shape. A baseline only shrinks:
remove a name when you move its code, and never add one.

- **Routes** return typed schemas, not raw dicts, and use `fastapi.status` constants.
- **Services** never import `sqlalchemy` or `sqlmodel`. A service that handles a database failure catches
  `DATABASE_ERRORS` from `core/database.py`, which also covers the bare `TimeoutError` a
  connect timeout raises.
- **Reacting to another domain.** Dependencies between domains run one way. A domain that must
  react to a change in a domain that does not depend on it receives a listener interface by
  constructor injection, defined by the domain where the change happens. The listener runs in the
  caller's transaction and never commits.
- **Repositories** inherit `BaseRepository`, flush and never commit. A repository that turns a
  specific database error, such as `IntegrityError`, into a domain error is the module that
  imports it.
- **Exceptions.** Each error class carries its own `status_code`, and one registered handler
  renders its family as FastAPI's `{"detail": ...}` shape, so a route needs no `try`/`except`.
  A 5xx member has its message logged and a generic detail returned. One family exists:
  `TenancyError` and the four status bases under it, defined in `exceptions/_base.py`, imported
  from `gateway.exceptions` and rendered by `_tenancy_error_handler` in `gateway.main`. A
  domain's own error module subclasses those bases, as `exceptions/budget_exceptions.py` does.

Catch specific exceptions, not a broad `except Exception`.

### How a service is built

A domain's service is its Service Layer, and [docs/domains.md](../../../docs/domains.md#the-target-shape)
gives its shape: one service per domain with a small public API, built by constructor injection,
from a builder in `api/deps.py`. Two rules add to it:

- **Hybrid mode.** A hybrid gateway has no local database, so a service that needs one is not
  built there.
- **Moving old code.** A module-level function that takes a session becomes a method of its
  domain's service, or a repository method when all it does is run a query. A helper that
  needs no database stays a plain function.

### Who commits

A Unit of Work (`core/unit_of_work.py`) marks where a business transaction starts, commits and
rolls back.

- Each request, and each worker job, has one Unit of Work over its one session, and every service
  built in that scope shares it. A request gets it from `get_unit_of_work` in `api/deps.py`. A
  worker job gets it from `create_unit_of_work()`, or from `create_log_unit_of_work()` for work
  on the metering pool.
- A business step is one `async with uow:` block, an async context manager. The block commits
  when it ends, and rolls back and re-raises on an error. Either way the connection goes back to
  the pool.
- Blocks nest. An inner block joins the outer one, and only the outermost block commits, so a step
  that writes to two domains is atomic.
- A step in which an inner block failed is rolled back as a whole, even when the code around that
  block caught the error: the outermost block rolls back and raises `UnitOfWorkRolledBackError`.
- If the rollback itself fails with a database error, that error is logged, and the error that
  ended the step is still the one the caller sees.
- A repository reaches the session only through `session_for(uow)`, which raises
  `OutsideUnitOfWorkError` when no block is open. A Unit of Work exposes no session of its own.
  A `BaseRepository` built on a Unit of Work calls `session_for` for every operation.
- Only a service opens a block. A repository never commits, and a route never opens a block.
- A request whose database work all runs in blocks needs no `release_session` before it
  dispatches upstream.
- Hybrid mode has no local database and no Unit of Work.

Code still in the old shape commits in its services, and some routes still commit. A domain's
commits move into blocks when the domain moves
([step 4](../../../docs/domains.md#what-one-domain-change-does)).

The boundary check refuses a `commit()` or `rollback()` call outside `core/unit_of_work.py`, except
in the modules on `TRANSACTION_CONTROL_BASELINE`, which names that code. It also refuses an import
of `session_for` outside `repositories/`.

### Sources

- Service Layer (Randy Stafford), Repository (Edward Hieatt and Rob Mee) and Unit of Work, in
  Martin Fowler, *Patterns of Enterprise Application Architecture* (2002):
  <https://martinfowler.com/eaaCatalog/>
- Harry Percival and Bob Gregory, *Architecture Patterns with Python* (2020), chapter 2
  (Repository), chapter 4 (Service Layer) and chapter 6 (Unit of Work):
  <https://www.cosmicpython.com/book/>
- Constructor injection: Martin Fowler, "Inversion of Control Containers and the Dependency
  Injection pattern" (2004): <https://martinfowler.com/articles/injection.html>
- Deep modules: John Ousterhout, *A Philosophy of Software Design* (2018), chapter 4, "Modules
  Should Be Deep"
- Async context manager: PEP 492, <https://peps.python.org/pep-0492/>

## The budget / reservation lifecycle is load-bearing

Billable routes hold money-adjacent state. The invariant (detailed in
[../../instructions/security-review.instructions.md](../../instructions/security-review.instructions.md#budget-billing-and-tenant-isolation))
is: **reserve before the provider call, then reconcile on success or refund on every error
path**, including provider errors, tool-iteration caps, unreachable sandbox/web-search,
generic `except`, and `except HTTPException`, plus streaming completion and client disconnect.
A reservation that never settles leaks and permanently shrinks the user's budget.

- Bind spend to the **authenticated principal** via `resolve_user_id`, never to the
  client-supplied `user` field.
- Enforce budgets atomically (the reservation is a single conditional `UPDATE`), not
  check-then-act. Use `is None` for "absent" vs a legitimate `0` (falsy-zero traps).
- New billable logic must be correct in every applicable runtime mode. Verify whether it
  belongs in the local control-plane branch (`db is not None`), the data plane, or both.

## Migrations (Alembic)

- A change to anything under `models/` ships with a matching migration in `alembic/versions/`,
  chained to the current head, in the same PR.
- New non-nullable columns need a `server_default` for existing rows (e.g. `users.reserved`
  defaults to `"0"`).
- Every foreign key needs an explicit `ondelete` policy; index it (`index=True`), see the
  performance instructions. Account deletion must leave no orphaned billable rows.
- Provide a real, reversible `downgrade()`.
- Name each table in `op.create_table` or `op.rename_table` with a string literal or a module-level
  constant. Autogenerate reads those names to tell otari's tables from another chain's tables in the
  same database, and it refuses a revision whose table name it cannot read.
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
- [backend-architecture.instructions.md](../../instructions/backend-architecture.instructions.md): the layer and import rules above, restated for review.
- [security-review.instructions.md](../../instructions/security-review.instructions.md): budget/tenant isolation, auth, SSRF, prompt injection, migration safety.
