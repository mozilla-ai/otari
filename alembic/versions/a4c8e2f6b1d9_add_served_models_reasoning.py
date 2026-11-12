"""Keep whether a model listing describes the model's reasoning options.

A server built on vLLM can list a model with a ``reasoning`` object (the
efforts it accepts and the default one), which says the model reasons when
models.dev does not know the model.

Revision ID: a4c8e2f6b1d9
Revises: d8a491bc02fa
Create Date: 2026-11-12
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "a4c8e2f6b1d9"
down_revision: str | Sequence[str] | None = "d8a491bc02fa"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("served_models", sa.Column("reasoning", sa.Boolean(), server_default=sa.false(), nullable=False))


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table("served_models") as batch_op:
        batch_op.drop_column("reasoning")
