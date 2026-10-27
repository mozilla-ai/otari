"""Record how an API key was issued, and whether an operator switched it off.

Adds ``self_service`` to ``api_keys``: true for a key a member minted for
themselves, false for one an operator issued or ``config.yml`` declared, and
null for every key predating this column, whose origin is not recorded. Adds
``operator_disabled``: true while a key is off because an operator, or a
removal from its workspace or organization, switched it off, which its owner
may then not undo.

Revision ID: e5a7c9d1f3b2
Revises: e3a1c5d7f9b2
Create Date: 2026-10-27
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "e5a7c9d1f3b2"
down_revision: str | Sequence[str] | None = "e3a1c5d7f9b2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add the two columns; existing keys are of unrecorded origin and not operator-disabled."""
    op.add_column("api_keys", sa.Column("self_service", sa.Boolean(), nullable=True))
    op.add_column("api_keys", sa.Column("operator_disabled", sa.Boolean(), nullable=False, server_default=sa.false()))


def downgrade() -> None:
    """Drop the two columns."""
    with op.batch_alter_table("api_keys") as batch:
        batch.drop_column("operator_disabled")
        batch.drop_column("self_service")
