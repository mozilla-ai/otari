"""Keep the deployment's one attempt to connect to otari.ai.

Connecting a gateway to otari.ai is a device authorization: the gateway asks
for a code, the operator approves it on otari.ai, and the gateway asks for the
key. The attempt is kept in one row so it survives a restart and every replica
reads the same one. The device code is stored encrypted.

Revision ID: c4f9e2a7d5b8
Revises: b3e8d1f6a2c9
Create Date: 2026-11-10
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "c4f9e2a7d5b8"
down_revision: str | Sequence[str] | None = "b3e8d1f6a2c9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLE = "platform_connect_attempts"


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        _TABLE,
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("attempt_id", sa.Uuid(), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("encrypted_device_code", sa.String(), nullable=True),
        sa.Column("user_code", sa.String(length=64), nullable=False),
        sa.Column("verification_uri_complete", sa.String(length=2048), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("interval_seconds", sa.Integer(), nullable=False),
        sa.Column("next_poll_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("organization_id", sa.Uuid(), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("id = 1", name="ck_platform_connect_attempts_one_row"),
        sa.CheckConstraint(
            "status IN ('pending', 'connected', 'expired', 'denied')", name="ck_platform_connect_attempts_status"
        ),
        sa.ForeignKeyConstraint(["organization_id"], ["organization.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_platform_connect_attempts_organization_id", _TABLE, ["organization_id"], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_platform_connect_attempts_organization_id", table_name=_TABLE)
    op.drop_table(_TABLE)
