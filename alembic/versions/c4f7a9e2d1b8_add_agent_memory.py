"""Add agent memory: a workspace's data keys, its entries, the proposals in front of them, and their use.

- ``memory_keys``: each workspace's data keys, wrapped by the deployment's
  content key. Entries and proposals name the key that sealed them.
- ``memory_entries``: what agents recall, one row per version, its text and
  embedding sealed.
- ``memory_proposals``: what agents offered, sealed, deleted once decided.
- ``memory_signals``: each member's confirmation or flag of an entry.
- ``memory_reads``: who recalled which entries, with no content.
- ``memory_workspace_settings``: each workspace's switches. No row is memory off.

On PostgreSQL every table but the settings is put under row-level security,
forced so that the owner the app connects as is held to it too: a row is visible
only while ``otari.memory_workspace`` names its workspace, and with the setting
unset nothing is. SQLite has no row-level security, so there the repositories'
workspace filter and the sealed content's associated data are the bound.

Revision ID: c4f7a9e2d1b8
Revises: c6e8a0b2d4f1
Create Date: 2026-11-10
"""

from collections.abc import Sequence
from datetime import datetime

import sqlalchemy as sa
from alembic import op

revision: str = "c4f7a9e2d1b8"
down_revision: str | Sequence[str] | None = "c6e8a0b2d4f1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

MEMORY_KEYS = "memory_keys"
MEMORY_ENTRIES = "memory_entries"
MEMORY_PROPOSALS = "memory_proposals"
MEMORY_SIGNALS = "memory_signals"
MEMORY_READS = "memory_reads"
MEMORY_WORKSPACE_SETTINGS = "memory_workspace_settings"

# The tables a workspace's rows are fenced in, in the order they are created.
FENCED_TABLES = (MEMORY_KEYS, MEMORY_ENTRIES, MEMORY_PROPOSALS, MEMORY_SIGNALS, MEMORY_READS)
# NULLIF, because an unset custom setting reads as '' once the session has set it at all.
_CURRENT_WORKSPACE = "NULLIF(current_setting('otari.memory_workspace', true), '')::uuid"


def _timestamp(name: str) -> sa.Column[datetime]:
    return sa.Column(name, sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False)


def _provenance() -> list[sa.Column[str]]:
    return [
        sa.Column("source_api_key_id", sa.String(length=255), nullable=True),
        sa.Column("source_session_id", sa.String(length=255), nullable=True),
        sa.Column("source_model", sa.String(length=255), nullable=True),
    ]


def upgrade() -> None:
    op.create_table(
        MEMORY_KEYS,
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("workspace_id", sa.Uuid(), nullable=False),
        sa.Column("wrapper", sa.String(length=16), nullable=False),
        sa.Column("wrapped", sa.Text(), nullable=False),
        _timestamp("created_at"),
        sa.ForeignKeyConstraint(["workspace_id"], ["workspace.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_memory_keys_workspace_created_at", MEMORY_KEYS, ["workspace_id", "created_at"])

    op.create_table(
        MEMORY_ENTRIES,
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("workspace_id", sa.Uuid(), nullable=False),
        sa.Column("scope", sa.String(length=16), nullable=False),
        sa.Column("owner_user_id", sa.Uuid(), nullable=True),
        sa.Column("author_user_id", sa.Uuid(), nullable=True),
        sa.Column("kind", sa.String(length=16), nullable=False),
        sa.Column("status", sa.String(length=16), server_default="active", nullable=False),
        sa.Column("trust", sa.String(length=16), nullable=False),
        sa.Column("version", sa.Integer(), server_default="1", nullable=False),
        sa.Column("supersedes_id", sa.Uuid(), nullable=True),
        sa.Column("key_id", sa.Uuid(), nullable=False),
        sa.Column("content_sealed", sa.Text(), nullable=False),
        sa.Column("embedding_sealed", sa.Text(), nullable=True),
        sa.Column("embedding_model", sa.String(length=255), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("confirmations", sa.Integer(), server_default="0", nullable=False),
        sa.Column("flags", sa.Integer(), server_default="0", nullable=False),
        *_provenance(),
        sa.Column("use_count", sa.Integer(), server_default="0", nullable=False),
        _timestamp("last_used_at"),
        _timestamp("created_at"),
        _timestamp("updated_at"),
        sa.CheckConstraint("(scope = 'member') = (owner_user_id IS NOT NULL)", name="check_memory_entries_owner"),
        sa.CheckConstraint("kind IN ('fact', 'decision', 'procedure')", name="check_memory_entries_kind"),
        sa.CheckConstraint("scope IN ('workspace', 'member')", name="check_memory_entries_scope"),
        sa.CheckConstraint(
            "status IN ('active', 'superseded', 'retired', 'quarantined')", name="check_memory_entries_status"
        ),
        sa.CheckConstraint("trust IN ('reviewed', 'agent', 'external')", name="check_memory_entries_trust"),
        sa.CheckConstraint("version > 0", name="check_memory_entries_version"),
        sa.ForeignKeyConstraint(["author_user_id"], ["user.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["key_id"], ["memory_keys.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["owner_user_id"], ["user.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["supersedes_id"], ["memory_entries.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["workspace_id"], ["workspace.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_memory_entries_author_user_id", MEMORY_ENTRIES, ["author_user_id"])
    op.create_index("ix_memory_entries_key_id", MEMORY_ENTRIES, ["key_id"])
    op.create_index("ix_memory_entries_supersedes_id", MEMORY_ENTRIES, ["supersedes_id"])
    op.create_index("ix_memory_entries_workspace_last_used_at", MEMORY_ENTRIES, ["workspace_id", "last_used_at"])
    op.create_index(
        "ix_memory_entries_workspace_owner",
        MEMORY_ENTRIES,
        ["workspace_id", "owner_user_id"],
        postgresql_where=sa.text("owner_user_id IS NOT NULL"),
        sqlite_where=sa.text("owner_user_id IS NOT NULL"),
    )
    op.create_index("ix_memory_entries_workspace_status", MEMORY_ENTRIES, ["workspace_id", "status"])

    op.create_table(
        MEMORY_PROPOSALS,
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("workspace_id", sa.Uuid(), nullable=False),
        sa.Column("scope", sa.String(length=16), nullable=False),
        sa.Column("owner_user_id", sa.Uuid(), nullable=True),
        sa.Column("author_user_id", sa.Uuid(), nullable=True),
        sa.Column("kind", sa.String(length=16), nullable=False),
        sa.Column("trust", sa.String(length=16), nullable=False),
        sa.Column("state", sa.String(length=16), server_default="pending", nullable=False),
        sa.Column("key_id", sa.Uuid(), nullable=False),
        sa.Column("content_sealed", sa.Text(), nullable=False),
        sa.Column("review_reason", sa.String(length=64), nullable=True),
        *_provenance(),
        sa.Column("attempts", sa.Integer(), server_default="0", nullable=False),
        _timestamp("created_at"),
        sa.CheckConstraint("(scope = 'member') = (owner_user_id IS NOT NULL)", name="check_memory_proposals_owner"),
        sa.CheckConstraint("kind IN ('fact', 'decision', 'procedure')", name="check_memory_proposals_kind"),
        sa.CheckConstraint("scope IN ('workspace', 'member')", name="check_memory_proposals_scope"),
        sa.CheckConstraint("state IN ('pending', 'review')", name="check_memory_proposals_state"),
        sa.CheckConstraint("trust IN ('reviewed', 'agent', 'external')", name="check_memory_proposals_trust"),
        sa.ForeignKeyConstraint(["author_user_id"], ["user.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["key_id"], ["memory_keys.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["owner_user_id"], ["user.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["workspace_id"], ["workspace.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_memory_proposals_author_user_id", MEMORY_PROPOSALS, ["author_user_id"])
    op.create_index("ix_memory_proposals_key_id", MEMORY_PROPOSALS, ["key_id"])
    op.create_index("ix_memory_proposals_owner_user_id", MEMORY_PROPOSALS, ["owner_user_id"])
    op.create_index("ix_memory_proposals_state_created_at", MEMORY_PROPOSALS, ["state", "created_at"])
    op.create_index("ix_memory_proposals_workspace_state", MEMORY_PROPOSALS, ["workspace_id", "state"])

    op.create_table(
        MEMORY_SIGNALS,
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("workspace_id", sa.Uuid(), nullable=False),
        sa.Column("entry_id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("signal", sa.String(length=16), nullable=False),
        sa.Column("reason", sa.String(length=64), nullable=True),
        _timestamp("created_at"),
        sa.CheckConstraint("signal IN ('confirm', 'flag')", name="check_memory_signals_signal"),
        sa.ForeignKeyConstraint(["entry_id"], ["memory_entries.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["user.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["workspace_id"], ["workspace.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("entry_id", "user_id", name="uq_memory_signals_entry_user"),
    )
    op.create_index("ix_memory_signals_user_id", MEMORY_SIGNALS, ["user_id"])
    op.create_index("ix_memory_signals_workspace_id", MEMORY_SIGNALS, ["workspace_id"])

    op.create_table(
        MEMORY_READS,
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("workspace_id", sa.Uuid(), nullable=False),
        sa.Column("reader_user_id", sa.Uuid(), nullable=True),
        sa.Column("api_key_id", sa.String(length=255), nullable=True),
        sa.Column("tool", sa.String(length=64), nullable=False),
        sa.Column("entry_ids", sa.JSON(), nullable=False),
        _timestamp("read_at"),
        sa.ForeignKeyConstraint(["reader_user_id"], ["user.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["workspace_id"], ["workspace.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_memory_reads_reader_user_id", MEMORY_READS, ["reader_user_id"])
    op.create_index("ix_memory_reads_workspace_read_at", MEMORY_READS, ["workspace_id", "read_at"])

    op.create_table(
        MEMORY_WORKSPACE_SETTINGS,
        sa.Column("workspace_id", sa.Uuid(), nullable=False),
        sa.Column("enabled", sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.Column("workspace_scope_enabled", sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.Column("write_policy", sa.String(length=16), server_default="review", nullable=False),
        sa.Column("hide_external", sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.Column("retention_days", sa.Integer(), nullable=True),
        _timestamp("updated_at"),
        sa.CheckConstraint("write_policy IN ('auto', 'review')", name="check_memory_settings_write_policy"),
        sa.CheckConstraint("retention_days IS NULL OR retention_days > 0", name="check_memory_settings_retention"),
        sa.ForeignKeyConstraint(["workspace_id"], ["workspace.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("workspace_id"),
    )

    if op.get_bind().dialect.name == "postgresql":
        for table in FENCED_TABLES:
            op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
            op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
            op.execute(
                f"CREATE POLICY {table}_workspace ON {table} "
                f"USING (workspace_id = {_CURRENT_WORKSPACE}) WITH CHECK (workspace_id = {_CURRENT_WORKSPACE})"
            )


def downgrade() -> None:
    # Dropping a table drops its policies with it.
    op.drop_table(MEMORY_WORKSPACE_SETTINGS)
    op.drop_index("ix_memory_reads_workspace_read_at", table_name=MEMORY_READS)
    op.drop_index("ix_memory_reads_reader_user_id", table_name=MEMORY_READS)
    op.drop_table(MEMORY_READS)
    op.drop_index("ix_memory_signals_workspace_id", table_name=MEMORY_SIGNALS)
    op.drop_index("ix_memory_signals_user_id", table_name=MEMORY_SIGNALS)
    op.drop_table(MEMORY_SIGNALS)
    op.drop_index("ix_memory_proposals_workspace_state", table_name=MEMORY_PROPOSALS)
    op.drop_index("ix_memory_proposals_state_created_at", table_name=MEMORY_PROPOSALS)
    op.drop_index("ix_memory_proposals_owner_user_id", table_name=MEMORY_PROPOSALS)
    op.drop_index("ix_memory_proposals_key_id", table_name=MEMORY_PROPOSALS)
    op.drop_index("ix_memory_proposals_author_user_id", table_name=MEMORY_PROPOSALS)
    op.drop_table(MEMORY_PROPOSALS)
    op.drop_index("ix_memory_entries_workspace_status", table_name=MEMORY_ENTRIES)
    op.drop_index("ix_memory_entries_workspace_owner", table_name=MEMORY_ENTRIES)
    op.drop_index("ix_memory_entries_workspace_last_used_at", table_name=MEMORY_ENTRIES)
    op.drop_index("ix_memory_entries_supersedes_id", table_name=MEMORY_ENTRIES)
    op.drop_index("ix_memory_entries_key_id", table_name=MEMORY_ENTRIES)
    op.drop_index("ix_memory_entries_author_user_id", table_name=MEMORY_ENTRIES)
    op.drop_table(MEMORY_ENTRIES)
    op.drop_index("ix_memory_keys_workspace_created_at", table_name=MEMORY_KEYS)
    op.drop_table(MEMORY_KEYS)
