"""Add the spending limit and lifetime a person approves for an OAuth app key.

``oauth_app_grants`` carries them from the consent screen to the token exchange,
which caps the key and sets its expiry. Both nullable, so a grant in flight
across the upgrade redeems to a key with no limit and no expiry, as before.

Revision ID: e8c2a4f6b1d3
Revises: d5a7c9e1b3f4
Create Date: 2026-11-18
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "e8c2a4f6b1d3"
down_revision: str | Sequence[str] | None = "d5a7c9e1b3f4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("oauth_app_grants") as batch_op:
        batch_op.add_column(sa.Column("spend_limit", sa.Numeric(precision=18, scale=6), nullable=True))
        batch_op.add_column(sa.Column("key_lifetime_sec", sa.Integer(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("oauth_app_grants") as batch_op:
        batch_op.drop_column("key_lifetime_sec")
        batch_op.drop_column("spend_limit")
