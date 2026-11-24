"""Let a hosted provider declare where its inference runs, for routing policies with eligibility.

``routing_locality`` is the operator's declaration (``local``, ``private`` or
``cloud``) that a routing policy's ``eligibility`` reads. Null for every
existing row: nothing has been declared yet, and an undeclared provider is
never eligible.

Revision ID: c9d1e3f5a7b2
Revises: f6b8d0a2c4e7
Create Date: 2026-11-24
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "c9d1e3f5a7b2"
down_revision: str | Sequence[str] | None = "f6b8d0a2c4e7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("hosted_providers", sa.Column("routing_locality", sa.String(length=16), nullable=True))


def downgrade() -> None:
    op.drop_column("hosted_providers", "routing_locality")
