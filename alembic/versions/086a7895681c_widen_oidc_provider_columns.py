"""Widen the three provider columns so a customer's own OIDC issuer URL fits.

``oauth_identity_link.provider``, ``user.oauth_provider`` and
``oauth_pending_state.provider`` each held a short fixed vocabulary
(``google``, ``github``) at their old widths. A customer's own OIDC
connection is named by its issuer URL instead, which does not fit in 50 or
32 characters, so all three widen to 255 alongside
``gateway.models.identity.MAX_PROVIDER_LENGTH``.

``batch_alter_table`` is used on all three tables because SQLite cannot
``ALTER COLUMN`` in place; it recreates the table on SQLite and renders a
plain ``ALTER COLUMN`` on PostgreSQL, so one form works on both engines.

Revision ID: 086a7895681c
Revises: 90b7e153c41f
Create Date: 2026-11-03
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "086a7895681c"
down_revision: str | Sequence[str] | None = "d1145eb686f8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # NOTE: SQLite's batch reflection preserves this table's named unique
    # constraint and composite primary key, so copy_from is not needed here.
    with op.batch_alter_table("oauth_identity_link") as batch_op:
        batch_op.alter_column(
            "provider", existing_type=sa.String(length=50), type_=sa.String(length=255), existing_nullable=False
        )
    with op.batch_alter_table("user") as batch_op:
        batch_op.alter_column(
            "oauth_provider", existing_type=sa.String(length=50), type_=sa.String(length=255), existing_nullable=True
        )
    with op.batch_alter_table("oauth_pending_state") as batch_op:
        batch_op.alter_column(
            "provider", existing_type=sa.String(length=32), type_=sa.String(length=255), existing_nullable=False
        )


def downgrade() -> None:
    with op.batch_alter_table("oauth_pending_state") as batch_op:
        batch_op.alter_column(
            "provider", existing_type=sa.String(length=255), type_=sa.String(length=32), existing_nullable=False
        )
    with op.batch_alter_table("user") as batch_op:
        batch_op.alter_column(
            "oauth_provider", existing_type=sa.String(length=255), type_=sa.String(length=50), existing_nullable=True
        )
    with op.batch_alter_table("oauth_identity_link") as batch_op:
        batch_op.alter_column(
            "provider", existing_type=sa.String(length=255), type_=sa.String(length=50), existing_nullable=False
        )
