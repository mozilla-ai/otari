"""Store what an organization's guardrail reads.

A mandate checked the caller's input and nothing else. ``directions`` (the
mandate's ``on``, a word SQL reserves) names what it reads, as a request's
guardrail entry already can: the input, tool results, tool calls or the model's
answer. Existing rows keep reading the input alone.

Revision ID: d8e1f3a5b7c9
Revises: a7f2c91d4e60
Create Date: 2026-11-10
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "d8e1f3a5b7c9"
down_revision: str | Sequence[str] | None = "a7f2c91d4e60"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "organization_guardrails",
        sa.Column("directions", sa.JSON(), server_default='["input"]', nullable=False),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("organization_guardrails", "directions")
