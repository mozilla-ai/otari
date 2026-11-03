"""Turn stored policies on a retired router into plain ones.

The ``knn`` and ``noop`` router backends are gone, and a policy naming either
already served its default target on every request. This removes that ``router``
entry from each stored policy's ``select``, which leaves the policy serving the
same model, editable in the dashboard instead of shown read-only. A policy whose
``select`` then holds only its ``default`` is an ordinary one-model policy.

Only stored policies are rewritten. A ``config.yml`` policy naming either router
lives in a file this process does not own; it keeps serving its default and warns
once, like any backend a build lacks.

The downgrade does not restore the entries: the router they named is gone either
way, and the policy served its default before and after.

Revision ID: a79a97473478
Revises: c4c7c29847fa
Create Date: 2026-11-03
"""

import json
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "a79a97473478"
down_revision: str | Sequence[str] | None = "c4c7c29847fa"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_RETIRED_ROUTERS = frozenset({"knn", "noop"})


def _is_retired(entry: object) -> bool:
    router = entry.get("router") if isinstance(entry, dict) else None
    return isinstance(router, str) and router.strip().lower() in _RETIRED_ROUTERS


def upgrade() -> None:
    conn = op.get_bind()
    rows = conn.execute(sa.text("SELECT id, spec FROM routing_policies")).mappings().all()
    for row in rows:
        # ``spec`` is ``sa.JSON``, which a plain ``text()`` select hands back as a
        # string on SQLite and decoded on PostgreSQL.
        spec = json.loads(row["spec"]) if isinstance(row["spec"], str) else row["spec"]
        select = spec.get("select") if isinstance(spec, dict) else None
        if not isinstance(select, list) or not any(_is_retired(entry) for entry in select):
            continue
        spec["select"] = [entry for entry in select if not _is_retired(entry)]
        conn.execute(
            sa.text("UPDATE routing_policies SET spec = :spec WHERE id = :id").bindparams(
                sa.bindparam("spec", type_=sa.JSON)
            ),
            {"spec": spec, "id": row["id"]},
        )


def downgrade() -> None:
    pass
