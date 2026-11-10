"""Add an organization's SSO enforcement switch and its break-glass exception.

The two break-glass columns tell "no exception chosen yet" apart from "we chose to have none".
The service layer requires one of them before it sets ``sso_required``.

``sso_break_glass_user_id`` is a second foreign key from this table to ``user``, so it closes the
same cycle its existing one does and needs the same named, altered constraint to stay droppable.
``batch_alter_table`` keeps every statement here valid on SQLite too.

Revision ID: 10727bf67704
Revises: b7c3e9d1a4f8
Create Date: 2026-11-05
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "10727bf67704"
down_revision: str | Sequence[str] | None = "b7c3e9d1a4f8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_FK = "fk_organization_sso_break_glass_user_id"
_IX = "ix_organization_sso_break_glass_user_id"


def upgrade() -> None:
    with op.batch_alter_table("organization") as batch_op:
        batch_op.add_column(sa.Column("sso_required", sa.Boolean(), nullable=False, server_default=sa.false()))
        batch_op.add_column(sa.Column("sso_break_glass_user_id", sa.Uuid(), nullable=True))
        batch_op.add_column(
            sa.Column("sso_break_glass_waived", sa.Boolean(), nullable=False, server_default=sa.false())
        )
        batch_op.create_index(_IX, ["sso_break_glass_user_id"])
        batch_op.create_foreign_key(_FK, "user", ["sso_break_glass_user_id"], ["id"], ondelete="SET NULL")


def downgrade() -> None:
    with op.batch_alter_table("organization") as batch_op:
        batch_op.drop_constraint(_FK, type_="foreignkey")
        batch_op.drop_index(_IX)
        batch_op.drop_column("sso_break_glass_waived")
        batch_op.drop_column("sso_break_glass_user_id")
        batch_op.drop_column("sso_required")
