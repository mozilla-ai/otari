"""Add routing_policies table.

The runtime counterpart of the ``routing.policies`` config block, so policies can
be managed through the API and the dashboard instead of only by editing a file and
restarting.

Scoping and uniqueness mirror ``model_aliases`` deliberately (a policy and an
alias are the same concept at different complexities): a composite constraint for
one row per (name, user), plus a partial unique index for one *global* row per
name, which the composite constraint cannot enforce because both SQLite and
PostgreSQL treat NULL ``user_id`` values as distinct.

``spec`` is JSON rather than columns: it is a nested, versioned document, so
flattening it would mean a migration per schema addition and would still need JSON
for the ``when`` conditions.

Revision ID: e8b1d3f5a7c9
Revises: d7b9f1a3c5e8
Create Date: 2026-08-04 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "e8b1d3f5a7c9"
down_revision: str | Sequence[str] | None = "d7b9f1a3c5e8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "routing_policies",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("spec", sa.JSON(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id", name="routing_policies_pkey"),
        sa.UniqueConstraint("name", "user_id", name="uq_routing_policies_name_user"),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.user_id"],
            name="fk_routing_policies_user_id_users",
            ondelete="CASCADE",
        ),
    )
    op.create_index("ix_routing_policies_user_id", "routing_policies", ["user_id"])
    # No plain index on `name`: uq_routing_policies_name_user already leads with it.
    op.create_index(
        "uq_routing_policies_global_name",
        "routing_policies",
        ["name"],
        unique=True,
        sqlite_where=sa.text("user_id IS NULL"),
        postgresql_where=sa.text("user_id IS NULL"),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("uq_routing_policies_global_name", table_name="routing_policies")
    op.drop_index("ix_routing_policies_user_id", table_name="routing_policies")
    op.drop_table("routing_policies")
