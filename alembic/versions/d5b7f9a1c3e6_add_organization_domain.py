"""Add the organization email-domain claim table.

Backs email-domain auto-join: an organization claims a domain, proves control
of it with a DNS TXT record, and anyone signing in afterwards with a verified
address at that domain joins as the claim's ``default_role``.

One table for the claim and its proof, rather than the platform's two
migrations (``b7d2f4a1c9e0`` then ``a3e9f2c1d8b4``, which added verification
later). Nothing here has ever shipped without the proof, and a claim without one
grants nothing, so there is no state worth reproducing the split for.

``UNIQUE(domain)`` is deployment-wide and not per organization. Auto-join
resolves an address to exactly one organization, and two organizations holding
the same claim has no defensible answer; the constraint is also what settles the
race between two admins claiming a domain at once.

No index on ``domain`` beyond that constraint's own: the sign-in lookup is an
equality match on it, which the unique index already serves.

Revision ID: d5b7f9a1c3e6
Revises: c9f2a6b4e8d7
Create Date: 2026-09-03
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "d5b7f9a1c3e6"
down_revision: str | Sequence[str] | None = "c9f2a6b4e8d7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "organization_domain",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("organization_id", sa.Uuid(), nullable=False),
        sa.Column("domain", sa.String(length=255), nullable=False),
        sa.Column("default_role", sa.String(length=32), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("verification_token", sa.String(length=64), nullable=False),
        sa.Column("verified_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["organization_id"], ["organization.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("domain", name="uq_organization_domain_domain"),
    )
    op.create_index(
        op.f("ix_organization_domain_organization_id"),
        "organization_domain",
        ["organization_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_organization_domain_organization_id"), table_name="organization_domain")
    op.drop_table("organization_domain")
