"""Add the organization email-domain claim table.

Backs email-domain auto-join: an organization claims a domain, proves control
of it with a DNS TXT record, and anyone signing in afterwards with a verified
address at that domain joins as the claim's ``default_role``.

One table for the claim and its proof, rather than the platform's two
migrations (``b7d2f4a1c9e0`` then ``a3e9f2c1d8b4``, which added verification
later). Nothing here has ever shipped without the proof, and a claim without one
grants nothing, so there is no state worth reproducing the split for.

Uniqueness is **partial**, over verified rows alone. Auto-join has to resolve an
address to exactly one organization, so two *proven* claims on a domain has no
defensible answer, and the index is what settles the race between two
organizations verifying at once. Unproven claims are deliberately not unique: a
plain ``UNIQUE(domain)`` would make claiming first-come-first-served, so anyone
who can create an organization could permanently lock the domain's real owner
out of ever claiming it.

``domain`` carries a plain index of its own for that reason: the partial unique
one covers only verified rows, and both the claim conflict check and the
displacement sweep on verify read unverified ones.

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

_VERIFIED_DOMAIN = "uq_organization_domain_verified_domain"


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
    )
    op.create_index(
        op.f("ix_organization_domain_organization_id"),
        "organization_domain",
        ["organization_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_organization_domain_domain"),
        "organization_domain",
        ["domain"],
        unique=False,
    )
    op.create_index(
        _VERIFIED_DOMAIN,
        "organization_domain",
        ["domain"],
        unique=True,
        postgresql_where=sa.text("verified_at IS NOT NULL"),
        sqlite_where=sa.text("verified_at IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_index(_VERIFIED_DOMAIN, table_name="organization_domain")
    op.drop_index(op.f("ix_organization_domain_domain"), table_name="organization_domain")
    op.drop_index(op.f("ix_organization_domain_organization_id"), table_name="organization_domain")
    op.drop_table("organization_domain")
