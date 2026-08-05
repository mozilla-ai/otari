"""Move stored aliases into routing_policies.

An alias is the one-target case of a routing policy, so keeping both stores meant
two tables, two endpoints, and two dashboard pages for one concept. This moves
every ``model_aliases`` row to ``routing_policies`` as ``{select: [{default:
<target>}]}``, which is the same thing the compiler would have produced for it.

Scope and timestamps are carried across unchanged, so a user-scoped alias stays
scoped to that user and "created" dates do not reset.

Rows are **moved**, not copied. Leaving them behind would put the same name in
both stores, and alias resolution runs before policy resolution, so the stale
alias would win and silently shadow every later edit made through the policy API.
The downgrade moves them back, so this is reversible in both directions; a
one-target policy created after the upgrade downgrades into an alias, which is a
faithful representation of it.

Only *stored* aliases are affected. The ``aliases:`` block in ``config.yml`` lives
in a file this process does not own; it keeps working and is still documented as
the one-target shorthand.

Revision ID: b5d7f9a1c3e6
Revises: f4c6a8b0d2e5
Create Date: 2026-08-05 00:00:00.000000

"""

import json
import uuid
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b5d7f9a1c3e6"
down_revision: str | Sequence[str] | None = "f4c6a8b0d2e5"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _spec_for(target: str) -> str:
    """The policy document an alias is equivalent to.

    Serialized here rather than handed over as a dict because the column is
    ``sa.JSON`` and this runs through a plain ``text()`` insert, which does no
    type coercion of its own.
    """
    return json.dumps({"spec_version": 1, "select": [{"default": target}], "on_failure": [], "guardrails": []})


def upgrade() -> None:
    """Upgrade schema."""
    conn = op.get_bind()
    rows = conn.execute(
        sa.text("SELECT name, target, user_id, created_at, updated_at FROM model_aliases")
    ).mappings().all()

    for row in rows:
        # A policy of the same name and scope already existing would make this
        # insert violate the unique constraint. Refuse rather than skip: alias
        # resolution used to win, so carrying on would delete the alias below and
        # silently hand the name to a policy that may serve a different model.
        # Which of the two the operator meant is not knowable from here, and this
        # object decides where money is spent, so it is theirs to resolve.
        clash = conn.execute(
            sa.text(
                "SELECT 1 FROM routing_policies WHERE name = :name "
                "AND (user_id = :user_id OR (user_id IS NULL AND :user_id IS NULL))"
            ),
            {"name": row["name"], "user_id": row["user_id"]},
        ).first()
        if clash is not None:
            scope = f"user {row['user_id']}" if row["user_id"] is not None else "global scope"
            raise RuntimeError(
                f"Cannot move alias {row['name']!r} ({scope}) into routing_policies: a policy of that "
                "name and scope already exists. The alias currently wins at request time, so this "
                "migration would change which model that name serves. Delete whichever of the two is "
                "obsolete, then re-run the migration."
            )
        conn.execute(
            sa.text(
                "INSERT INTO routing_policies (id, name, spec, user_id, created_at, updated_at) "
                "VALUES (:id, :name, :spec, :user_id, :created_at, :updated_at)"
            ),
            {
                "id": str(uuid.uuid4()),
                "name": row["name"],
                "spec": _spec_for(row["target"]),
                "user_id": row["user_id"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            },
        )

    conn.execute(sa.text("DELETE FROM model_aliases"))


def downgrade() -> None:
    """Downgrade schema.

    Moves back every policy that an alias can represent, which is one whose
    ``select`` is a single ``default`` entry with no failure chain and no
    guardrails. A policy doing more than that has no alias form, so it is left in
    place rather than silently flattened into one; a rolled-back binary does not
    know about ``routing_policies`` and will simply not see it.
    """
    conn = op.get_bind()
    rows = conn.execute(
        sa.text("SELECT id, name, spec, user_id, created_at, updated_at FROM routing_policies")
    ).mappings().all()

    for row in rows:
        spec = row["spec"]
        if isinstance(spec, str):
            spec = json.loads(spec)
        select = spec.get("select") or []
        if len(select) != 1 or "default" not in select[0]:
            continue
        if spec.get("on_failure") or spec.get("guardrails"):
            continue
        clash = conn.execute(
            sa.text(
                "SELECT 1 FROM model_aliases WHERE name = :name "
                "AND (user_id = :user_id OR (user_id IS NULL AND :user_id IS NULL))"
            ),
            {"name": row["name"], "user_id": row["user_id"]},
        ).first()
        if clash is not None:
            # Symmetric with the upgrade: deleting the policy without writing the
            # alias would drop the policy's target on the floor, and the surviving
            # alias may point somewhere else.
            scope = f"user {row['user_id']}" if row["user_id"] is not None else "global scope"
            raise RuntimeError(
                f"Cannot move policy {row['name']!r} ({scope}) back into model_aliases: an alias of that "
                "name and scope already exists. Delete whichever of the two is obsolete, then re-run the "
                "downgrade."
            )
        conn.execute(
            sa.text(
                "INSERT INTO model_aliases (id, name, target, user_id, created_at, updated_at) "
                "VALUES (:id, :name, :target, :user_id, :created_at, :updated_at)"
            ),
            {
                "id": str(uuid.uuid4()),
                "name": row["name"],
                "target": select[0]["default"],
                "user_id": row["user_id"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            },
        )
        conn.execute(sa.text("DELETE FROM routing_policies WHERE id = :id"), {"id": row["id"]})
