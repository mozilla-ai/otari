"""Add member_managed to api_key_federation_rules.

Marks a rule a member gave their own key, which names any public issuer and
matches only in the token exchange that names it. Every existing rule was
created by the operator, so the column defaults to false and nothing changes on
upgrade.

Revision ID: c4f1a9d27e6b
Revises: b7d9e1f3a5c2
Create Date: 2026-10-22
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "c4f1a9d27e6b"
down_revision: str | Sequence[str] | None = "b7d9e1f3a5c2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "api_key_federation_rules",
        sa.Column("member_managed", sa.Boolean(), server_default=sa.false(), nullable=False),
    )


def downgrade() -> None:
    op.drop_column("api_key_federation_rules", "member_managed")
