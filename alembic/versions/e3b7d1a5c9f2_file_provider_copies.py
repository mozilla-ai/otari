"""Record the copy a provider holds of a stored file.

A provider-native feature that names a file, such as Anthropic's code
execution, takes the provider's own file ID rather than Otari's. The copy is a
cache the provider expires, so the row holds the expiry it was given.

The key identifies the account the copy is in, because a provider file ID exists
only inside the account of the credential that uploaded it, and both the
configured instance and the dispatching workspace select that credential.

Revision ID: e3b7d1a5c9f2
Revises: a9c4e7b2d5f8
Create Date: 2026-09-24
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "e3b7d1a5c9f2"
down_revision: str | Sequence[str] | None = "a9c4e7b2d5f8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLE = "file_provider_copies"
_WORKSPACE_INDEX = "ix_file_provider_copies_credential_workspace_id"


def upgrade() -> None:
    op.create_table(
        _TABLE,
        sa.Column("file_id", sa.String(), nullable=False),
        sa.Column("provider", sa.String(), nullable=False),
        sa.Column("provider_instance", sa.String(), nullable=False),
        sa.Column("credential_workspace_id", sa.Uuid(), nullable=False),
        sa.Column("provider_file_id", sa.String(), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["file_id"], ["file_objects.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["credential_workspace_id"], ["workspace.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("file_id", "provider", "provider_instance", "credential_workspace_id"),
    )
    # The workspace foreign key cascades and the primary key leads with the
    # file, so without this a workspace deletion scans the whole table.
    op.create_index(_WORKSPACE_INDEX, _TABLE, ["credential_workspace_id"])


def downgrade() -> None:
    op.drop_index(_WORKSPACE_INDEX, table_name=_TABLE)
    op.drop_table(_TABLE)
