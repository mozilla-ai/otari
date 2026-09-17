"""Hold a guardrail definition in Otari rather than in a sidecar's YAML.

A row is the definition: the ``any_guardrail`` class plus the arguments that
build and call it, named by the profile a caller sends.

Two columns for the constructor arguments rather than one column per argument.
The guardrails a hosted API reaches carry between zero and three secret
constructor arguments each, so typing them would mean chasing every guardrail
upstream adds; instead the non-secret ones stay plain and every secret goes into
one map encrypted as a single string.

Nothing on the request path reads it yet. A downgrade drops the table and loses
the stored definitions with it.

Revision ID: d3f5a7c9e1b4
Revises: b2d4f6a8c0e2
Create Date: 2026-09-16
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "d3f5a7c9e1b4"
down_revision: str | Sequence[str] | None = "b2d4f6a8c0e2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "guardrail_credentials",
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("guardrail_name", sa.String(), nullable=False),
        sa.Column("create_kwargs", sa.JSON(), nullable=False),
        sa.Column("encrypted_create_secrets", sa.Text(), nullable=True),
        sa.Column("validate_kwargs", sa.JSON(), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("name"),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("guardrail_credentials")
