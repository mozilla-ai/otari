"""Add the workspace executor pin and let a file row stand for a provider-held file.

Three things this branch needs, in one revision because they land together:

- ``workspace_code_execution_policies.executor``, the workspace's pin on who
  runs a provider-native code-execution declaration (``auto``, ``otari`` or
  ``provider``), over the deployment's default and over the request's header.
  No backfill: NULL is "no pin", the state every existing row is in, so a
  deployment upgrading onto this revision keeps deciding exactly as it did.
- ``file_objects.provider``, ``file_objects.provider_instance`` and
  ``file_objects.provider_container_id``, with ``storage_ref`` made nullable. A
  provider-native code execution keeps what it produced in the provider's own
  container, so the row records who may read that id (the provider only
  authenticates the deployment's credential) and which provider, through which
  configured instance, to fetch it from, with no local blob to point at.
- Three indexes on ``file_objects``. One on ``expires_at``, because the sweep
  selects ``deleted_at IS NOT NULL OR expires_at < now`` and the existing
  ``deleted_at`` index cannot serve the ``OR`` on its own, so without it every
  tick scanned the table. Two composites for the paged listing, the tenant
  predicates then the keyset its cursor pages on:
  ``(user_id, workspace_id, created_at, id)`` for a keyed request, and
  ``(user_id, created_at, id)`` for a master-key listing that names no
  workspace, which the first cannot order. Without them every page sorts the
  user's whole set.

The ``ADD COLUMN``s need no table rebuild (see ``a4d7f1c9e2b6``), but dropping
``storage_ref``'s NOT NULL does on SQLite, which has no ``ALTER COLUMN``, so
that one statement goes through ``batch_alter_table``.

Revision ID: f2a6c81d9b47
Revises: d5f8b2a4c6e9
Create Date: 2026-09-21
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "f2a6c81d9b47"
down_revision: str | Sequence[str] | None = "d5f8b2a4c6e9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_POLICY_TABLE = "workspace_code_execution_policies"
_FILES_TABLE = "file_objects"


def upgrade() -> None:
    op.add_column(_POLICY_TABLE, sa.Column("executor", sa.String(length=16), nullable=True))

    op.add_column(_FILES_TABLE, sa.Column("provider", sa.String(), nullable=True))
    op.add_column(_FILES_TABLE, sa.Column("provider_instance", sa.String(), nullable=True))
    op.add_column(_FILES_TABLE, sa.Column("provider_container_id", sa.String(), nullable=True))
    with op.batch_alter_table(_FILES_TABLE) as batch:
        batch.alter_column("storage_ref", existing_type=sa.String(), nullable=True)

    op.create_index("ix_file_objects_expires_at", _FILES_TABLE, ["expires_at"])
    op.create_index(
        "ix_file_objects_user_workspace_created",
        _FILES_TABLE,
        ["user_id", "workspace_id", "created_at", "id"],
    )
    op.create_index("ix_file_objects_user_created", _FILES_TABLE, ["user_id", "created_at", "id"])


def downgrade() -> None:
    op.drop_index("ix_file_objects_user_created", table_name=_FILES_TABLE)
    op.drop_index("ix_file_objects_user_workspace_created", table_name=_FILES_TABLE)
    op.drop_index("ix_file_objects_expires_at", table_name=_FILES_TABLE)

    # A provider-held row has no blob to point at, so it cannot survive the
    # column going back to NOT NULL.
    op.execute(sa.text("DELETE FROM file_objects WHERE storage_ref IS NULL"))
    with op.batch_alter_table(_FILES_TABLE) as batch:
        batch.alter_column("storage_ref", existing_type=sa.String(), nullable=False)
    op.drop_column(_FILES_TABLE, "provider_container_id")
    op.drop_column(_FILES_TABLE, "provider_instance")
    op.drop_column(_FILES_TABLE, "provider")

    op.drop_column(_POLICY_TABLE, "executor")
