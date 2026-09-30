"""Record the copy a provider holds of a stored file.

A provider-native feature that names a file, such as Anthropic's code
execution, takes the provider's own file ID rather than Otari's. The copy is a
cache the provider expires, so the row holds the expiry it was given.

A row is recorded pending before its copy is uploaded, so no copy Otari makes
goes unnamed, and each copy has a row of its own. The account the copy is in is
named by an identity derived from the credential that made it, because a
provider file ID exists only inside that account.

Revision ID: e3b7d1a5c9f2
Revises: b7d2e9f4a6c1
Create Date: 2026-09-24
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "e3b7d1a5c9f2"
down_revision: str | Sequence[str] | None = "b7d2e9f4a6c1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLE = "file_provider_copies"
_ACCOUNT_INDEX = "ix_file_provider_copies_file_account"
_WORKSPACE_INDEX = "ix_file_provider_copies_credential_workspace_id"
_PENDING_INDEX = "ix_file_provider_copies_pending_since"
_EXPIRY_INDEX = "ix_file_provider_copies_expires_at"


def upgrade() -> None:
    op.create_table(
        _TABLE,
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("file_id", sa.String(), nullable=False),
        sa.Column("account_identity", sa.String(), nullable=False),
        sa.Column("provider", sa.String(), nullable=False),
        sa.Column("provider_instance", sa.String(), nullable=False),
        sa.Column("credential_workspace_id", sa.Uuid(), nullable=False),
        sa.Column("provider_file_id", sa.String(), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("pending_since", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "(pending_since IS NULL AND provider_file_id IS NOT NULL AND expires_at IS NOT NULL)"
            " OR (pending_since IS NOT NULL AND provider_file_id IS NULL)",
            name="ck_file_provider_copies_pending_or_confirmed",
        ),
        sa.ForeignKeyConstraint(["file_id"], ["file_objects.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["credential_workspace_id"], ["workspace.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(_ACCOUNT_INDEX, _TABLE, ["file_id", "account_identity"])
    op.create_index(_WORKSPACE_INDEX, _TABLE, ["credential_workspace_id"])
    op.create_index(
        _PENDING_INDEX,
        _TABLE,
        ["pending_since"],
        postgresql_where=sa.text("pending_since IS NOT NULL"),
        sqlite_where=sa.text("pending_since IS NOT NULL"),
    )
    op.create_index(
        _EXPIRY_INDEX,
        _TABLE,
        ["expires_at"],
        postgresql_where=sa.text("pending_since IS NULL"),
        sqlite_where=sa.text("pending_since IS NULL"),
    )


def downgrade() -> None:
    op.drop_index(_EXPIRY_INDEX, table_name=_TABLE)
    op.drop_index(_PENDING_INDEX, table_name=_TABLE)
    op.drop_index(_WORKSPACE_INDEX, table_name=_TABLE)
    op.drop_index(_ACCOUNT_INDEX, table_name=_TABLE)
    op.drop_table(_TABLE)
