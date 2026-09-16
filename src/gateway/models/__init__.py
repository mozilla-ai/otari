"""ORM models, and the one place that guarantees the schema is whole.

Importing any module in this package runs this file first, which imports every
model module in turn. That is what keeps ``Base.metadata`` (shared with
``SQLModel.metadata``, see `base`) complete for the three operations that
are only correct against the entire schema: Alembic's autogenerate comparison,
``create_all``, and ``drop_all``. Without it, whether a table exists in the
metadata would depend on which model modules the caller happened to import, so
Alembic would propose dropping the tables it could not see, and a test's
``drop_all`` teardown would leave the rest behind for the next test to collide
with.

A new model module that declares tables belongs in the import list below; the
schema-less request/response module beside them (`mcp`) contributes nothing
to the metadata and stays out of it.
"""

from gateway.models import (  # noqa: F401
    api_keys,
    budgets,
    entities,
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
)
