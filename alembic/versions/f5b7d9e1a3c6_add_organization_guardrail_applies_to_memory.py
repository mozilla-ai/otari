"""Let an organization guardrail check, or not, what agents propose to agent memory.

- ``applies_to_memory``: true (the default, and every existing mandate) also runs
  it over a proposal's text before the consolidator keeps it; false checks
  requests only, for a check that misreads memory notes.

Revision ID: f5b7d9e1a3c6
Revises: e3a5c7f9b2d4
Create Date: 2026-11-13
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "f5b7d9e1a3c6"
down_revision: str | Sequence[str] | None = "e3a5c7f9b2d4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

ORGANIZATION_GUARDRAILS = "organization_guardrails"


def upgrade() -> None:
    with op.batch_alter_table(ORGANIZATION_GUARDRAILS) as batch:
        batch.add_column(sa.Column("applies_to_memory", sa.Boolean(), server_default=sa.true(), nullable=False))


def downgrade() -> None:
    with op.batch_alter_table(ORGANIZATION_GUARDRAILS) as batch:
        batch.drop_column("applies_to_memory")
