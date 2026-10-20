"""Add api_key_federation_rules.

The OpenID Connect identities that may authenticate as an API key in place of
its secret (workload identity). Each rule names an issuer, an audience, and a
subject or an email; the check constraint keeps an issuer and audience match
alone from ever being enough. Rules go with their key.

Revision ID: 557fc0f5bfda
Revises: a3c5e7f9b1d4
Create Date: 2026-10-15 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "557fc0f5bfda"
down_revision: str | Sequence[str] | None = "a3c5e7f9b1d4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "api_key_federation_rules",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("api_key_id", sa.String(), nullable=False),
        sa.Column("issuer", sa.String(), nullable=False),
        sa.Column("audience", sa.String(), nullable=False),
        sa.Column("subject", sa.String(), nullable=True),
        sa.Column("email", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint("subject IS NOT NULL OR email IS NOT NULL", name="ck_api_key_federation_rules_principal"),
        sa.ForeignKeyConstraint(["api_key_id"], ["api_keys.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_api_key_federation_rules_api_key_id", "api_key_federation_rules", ["api_key_id"], unique=False)
    op.create_index(
        "ix_api_key_federation_rules_issuer_audience",
        "api_key_federation_rules",
        ["issuer", "audience"],
        unique=False,
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_api_key_federation_rules_issuer_audience", table_name="api_key_federation_rules")
    op.drop_index("ix_api_key_federation_rules_api_key_id", table_name="api_key_federation_rules")
    op.drop_table("api_key_federation_rules")
