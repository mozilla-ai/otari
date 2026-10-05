"""Let a stored rate-limit rule admit on tokens used rather than on an estimate.

Adds ``tpm_admission``: ``estimate`` (what every rule did so far) or ``used``.

Revision ID: b3e8f1a6c9d2
Revises: d8f1b3a5c7e9
Create Date: 2026-10-05
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "b3e8f1a6c9d2"
down_revision: str | Sequence[str] | None = "d8f1b3a5c7e9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "rate_limit_rules",
        sa.Column("tpm_admission", sa.String(), nullable=False, server_default="estimate"),
    )


def downgrade() -> None:
    with op.batch_alter_table("rate_limit_rules") as batch:
        batch.drop_column("tpm_admission")
