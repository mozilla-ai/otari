"""ORM models. Importing any module in this package registers every table.

Alembic autogenerate and ``create_all`` need the whole schema in ``Base.metadata``.
A table module missing from the import list below looks deleted to Alembic, which
then proposes dropping its tables.

Put a table in its domain's model module. A module that declares tables must be
in the list; one that declares none (`base`, `mcp`) stays out.
"""

from gateway.models import (  # noqa: F401
    api_keys,
    budgets,
    files,
    guardrails,
    inference,
    platform,
    playground,
    pricing,
    provider_keys,
    providers,
    routing,
    tenancy,
    tools,
    usage,
    users,
)
