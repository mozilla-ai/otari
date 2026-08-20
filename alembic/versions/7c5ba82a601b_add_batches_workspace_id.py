"""Add workspace_id to batches.

CodeRabbit review follow-up on otari#643: ``retrieve_batch``/``cancel_batch``/
``retrieve_batch_results`` resolved organization-scoped provider credentials
from the *caller's* current workspace, which is wrong for a master-key
retrieval or a legitimately cross-workspace one -- it can use the wrong
organization's key, or find none, exactly the failure ``api_key_id`` going
NULL on key revocation already risks for ownership. This column lets those
lifecycle calls resolve credentials from the workspace the batch was actually
*created* in.

Nullable, no backfill: batches created before this column existed (and any
row where the originating workspace has since been deleted, via SET NULL)
carry NULL here, and `api/routes/batches.py` falls back to the caller's own
workspace for those, matching how it already falls back to the metadata
marker for ownership on record-less legacy batches.

Revision ID: 7c5ba82a601b
Revises: e1c3a5b7d9f2
Create Date: 2026-08-20 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "7c5ba82a601b"
down_revision: str | Sequence[str] | None = "e1c3a5b7d9f2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("batches", sa.Column("workspace_id", sa.Uuid(), nullable=True))
    op.create_index(op.f("ix_batches_workspace_id"), "batches", ["workspace_id"], unique=False)
    # batch_alter_table: SQLite has no ALTER TABLE ADD CONSTRAINT, matching
    # `d5e7f1a2b3c4`'s precedent for adding a workspace FK to an existing table.
    with op.batch_alter_table("batches") as batch:
        batch.create_foreign_key(
            "fk_batches_workspace_id",
            "workspace",
            ["workspace_id"],
            ["id"],
            ondelete="SET NULL",
        )


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table("batches") as batch:
        batch.drop_constraint("fk_batches_workspace_id", type_="foreignkey")
    op.drop_index(op.f("ix_batches_workspace_id"), table_name="batches")
    op.drop_column("batches", "workspace_id")
