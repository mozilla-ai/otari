"""Store the country file an operator uploads.

``country_database`` holds at most one MaxMind DB country file, so every
replica reads the same one. ``content`` is the raw file: ``BYTEA`` on
PostgreSQL, ``BLOB`` on SQLite.

Revision ID: e3a1c5d7f9b2
Revises: db1bdb2e9a93
Create Date: 2026-10-27
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "e3a1c5d7f9b2"
down_revision: str | Sequence[str] | None = "db1bdb2e9a93"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "country_database",
        sa.Column("id", sa.Integer(), autoincrement=False, nullable=False),
        sa.Column("content", sa.LargeBinary(), nullable=False),
        sa.Column("database_type", sa.String(length=255), nullable=False),
        sa.Column("built_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column("uploaded_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("id = 1", name="ck_country_database_one_row"),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    op.drop_table("country_database")
