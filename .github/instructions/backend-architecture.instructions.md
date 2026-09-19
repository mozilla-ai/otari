---
applyTo: "src/gateway/**/*.py"
---

# Backend architecture review instructions

The backend is a modular monolith. The layers are the top-level folders under
`src/gateway/`, and each layer holds one package or module per domain.
`docs/domains.md` assigns every module to its domain.

## Old shape and new shape

Most existing modules are still in the old shape.
A service or route that handles a database failure catches `DATABASE_ERRORS`
from `gateway.core.database` rather than importing `sqlalchemy`. Review new and moved code
against the rules below, and do not accept "the module next to it does the
same" as a reason.

`SERVICE_DATABASE_IMPORT_BASELINE`, `ROUTE_DATABASE_IMPORT_BASELINE` and `FLAT_MODULE_BASELINE` in
`scripts/check_architecture.py` name the code still in the old shape.

- Do not flag an existing baseline entry the PR does not touch.
- Flag a PR that adds a name to any baseline.
- A PR that moves code out of the old shape removes its names from the
  baseline. The check fails until it does.

## Layer rules

| Layer | Path | Does | Flag when it |
| --- | --- | --- | --- |
| Routes | `api/routes/<domain>.py` | Parses the request, calls one service method, returns a schema | Imports `sqlalchemy` or `sqlmodel`, builds a query, holds a business rule, defines a Pydantic model, imports a repository, or commits |
| Schemas | `schemas/<domain>.py` | Holds Pydantic request and response models, and their mapping from ORM rows | Holds anything else |
| Services | `services/<domain>/` | Runs use cases: business rules and orchestration | Imports `sqlalchemy` or `sqlmodel`, builds a query, takes or holds a session in a class or in a module-level function, touches HTTP, or imports another domain's repository |
| Repositories | `repositories/<domain>/`, modules ending in `_repository.py` | Runs every query, over `BaseRepository`, and flushes | Commits, or holds a business rule |
| Exceptions | `exceptions/<domain>_exceptions.py` | Declares error classes, each with its own `status_code` | Handles an error |
| Models | `models/<domain>.py` | Declares ORM tables, and the closed vocabulary of each string column that has one | Holds logic |

## How a service is built

- One service per domain. The package's `__init__.py` exports the service and
  the types its public methods use, and nothing else. Each public method is one
  use case, and helpers sit in private modules whose names start with `_`.
- A domain whose service cannot offer a small public API is more than one
  domain. Flag a service that grows a wide interface.
- The service receives its own domain's repositories, the Unit of Work, config,
  ports and other domains' services through its constructor. Flag a service
  that receives the session or another domain's repository.
- A builder in `api/deps.py` builds the service. Flag a new service builder
  defined anywhere else.
- Flag a new module-level function under `services/` that takes an
  `AsyncSession`, including one added to a module already on
  `SERVICE_DATABASE_IMPORT_BASELINE`. The check does not catch that case.

## Commits

- A commit happens only when a Unit of Work block ends, and only a service
  opens a block. Flag a route or a repository that opens a block.
- The check refuses a `commit()` or `rollback()` call outside
  `core/unit_of_work.py`, except in a module on
  `TRANSACTION_CONTROL_BASELINE`. Flag such a call added to a module already
  on that baseline. The check does not catch that case.
- A repository reaches the session only through `session_for(uow)`. The check
  refuses that import anywhere else. Flag other code that reaches
  `session_for` without importing it by name.
- A request gets its Unit of Work from `get_unit_of_work`, and a worker job
  from `create_unit_of_work()` or `create_log_unit_of_work()`.
- Code still in the old shape commits in its services. Do not flag a commit
  the PR does not add.

## Imports between domains

- Code outside a domain imports its service only through the package root,
  `gateway.services.<domain>`. Flag an import of a module whose name starts
  with `_` from outside its package.
- Only the domain's own service package and `api/deps.py` import
  `gateway.repositories.<domain>`.
- Flag an import that makes two domain services depend on each other in a
  cycle. A domain that must react to a change in a domain that does not depend
  on it receives a listener interface by constructor injection, defined by the
  domain where the change happens.
- Flag a listener implementation that commits or rolls back. The caller owns
  the transaction.

The boundary check does not enforce these import rules yet, so review is the
only gate for them.

## Errors

A domain error carries its own `status_code`, and one registered handler
renders its family. One family exists: `TenancyError` and the status bases
under it, defined in `exceptions/_base.py`, imported from `gateway.exceptions`
and rendered by `_tenancy_error_handler` in `gateway.main`. A domain's own
error module subclasses those bases.

- Flag a route that catches a tenancy error to turn it into an
  `HTTPException`.

## Module size

Divider comments that cut a module into sections mean the module is more than
one module. Suggest splitting a module that a PR grows along such a divider.
