"""Add the edge-proxy credential an organization's provider key carries.

An organization that points a key at its own ``api_base`` often sits behind a
proxy that requires a second credential. All five columns are nullable, so an
existing key keeps dialing with its provider credential alone.

Revision ID: c7e1b9d4a260
Revises: f6b8d0a2c4e7
Create Date: 2026-11-20
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "c7e1b9d4a260"
down_revision: str | Sequence[str] | None = "f6b8d0a2c4e7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLE = "org_provider_keys"


def upgrade() -> None:
    with op.batch_alter_table(_TABLE) as batch_op:
        batch_op.add_column(sa.Column("endpoint_auth_method", sa.String(length=32), nullable=True))
        batch_op.add_column(sa.Column("endpoint_auth_header_name", sa.String(length=64), nullable=True))
        batch_op.add_column(sa.Column("encrypted_endpoint_auth_secret", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("endpoint_auth_last4", sa.String(length=8), nullable=True))
        batch_op.add_column(sa.Column("endpoint_auth_expires_at", sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table(_TABLE) as batch_op:
        batch_op.drop_column("endpoint_auth_expires_at")
        batch_op.drop_column("endpoint_auth_last4")
        batch_op.drop_column("encrypted_endpoint_auth_secret")
        batch_op.drop_column("endpoint_auth_header_name")
        batch_op.drop_column("endpoint_auth_method")
