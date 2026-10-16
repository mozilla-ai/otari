"""Make ``used`` the default ``tpm_admission`` for a stored rate limit rule.

A rule stored before keeps the value it has.

Revision ID: c9e4a7d2f8b6
Revises: b3e8f1a6c9d2
Create Date: 2026-10-06
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "c9e4a7d2f8b6"
down_revision: str | Sequence[str] | None = "b3e8f1a6c9d2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("rate_limit_rules") as batch:
        batch.alter_column("tpm_admission", existing_type=sa.String(), server_default="used")


def downgrade() -> None:
    with op.batch_alter_table("rate_limit_rules") as batch:
        batch.alter_column("tpm_admission", existing_type=sa.String(), server_default="estimate")
