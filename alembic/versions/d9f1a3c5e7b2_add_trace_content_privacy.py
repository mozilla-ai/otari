"""Add what keeping trace content privately needs: the workspace switch, shares, barriers and break-glass records.

- ``workspace.trace_content_enabled``: off for every workspace, existing ones
  included. A workspace keeps what its agent sessions were about only once its
  managers turn it on, and only where the deployment's ``trace_content_enabled``
  allows it.
- ``trace_content_shares`` and ``trace_content_reads``: a session's content
  opens only for its owner. A share lets the organization's admins, or the
  deployment's operator, read it until the owner ends it or loses the workspace, and
  every read through it is kept for the owner to see. Both go with the session.
- ``trace_content_barriers``: each workspace's content generation. Turning
  content off, or deleting it, moves the generation on, and a write stores
  content only when it was captured on the current generation.
- ``trace_user_barriers``: when each user's traces were last erased. A write
  locks the row of each user it stores for, so no replica stores spans of an
  erased user it was still holding.
- ``trace_break_glass_reads``: each opening of a session's content without its
  owner's share, recorded before the content opens, kept apart from the session
  and the organization so it outlives both, and deleted after its retention.

Revision ID: d9f1a3c5e7b2
Revises: e5a7c9d1f3b2
Create Date: 2026-10-23
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "d9f1a3c5e7b2"
down_revision: str | Sequence[str] | None = "e5a7c9d1f3b2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "workspace", sa.Column("trace_content_enabled", sa.Boolean(), nullable=False, server_default=sa.false())
    )

    op.create_table(
        "trace_content_shares",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("workspace_id", sa.Uuid(), nullable=False),
        sa.Column("trace_id", sa.String(length=64), nullable=False),
        sa.Column("audience", sa.String(length=32), nullable=False),
        sa.Column("note", sa.String(length=500), nullable=True),
        sa.Column("shared_by_user_id", sa.Uuid(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "audience IN ('organization_admins', 'deployment_operator')", name="check_trace_content_shares_audience"
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id", "trace_id"],
            ["traces.workspace_id", "traces.trace_id"],
            ondelete="CASCADE",
            name="fk_trace_content_shares_trace",
        ),
        sa.ForeignKeyConstraint(["shared_by_user_id"], ["user.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_trace_content_shares_trace", "trace_content_shares", ["workspace_id", "trace_id"])
    op.create_index("ix_trace_content_shares_shared_by_user_id", "trace_content_shares", ["shared_by_user_id"])
    op.create_table(
        "trace_content_reads",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("share_id", sa.Uuid(), nullable=True),
        sa.Column("workspace_id", sa.Uuid(), nullable=False),
        sa.Column("trace_id", sa.String(length=64), nullable=False),
        sa.Column("session_user_id", sa.String(), nullable=True),
        sa.Column("audience", sa.String(length=32), nullable=False),
        sa.Column("reader_user_id", sa.Uuid(), nullable=True),
        sa.Column("reader_email", sa.String(length=320), nullable=False),
        sa.Column("read_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["share_id"], ["trace_content_shares.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["reader_user_id"], ["user.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_trace_content_reads_share_id", "trace_content_reads", ["share_id"])
    op.create_index("ix_trace_content_reads_reader_user_id", "trace_content_reads", ["reader_user_id"])
    op.create_index("ix_trace_content_reads_share_read_at", "trace_content_reads", ["share_id", "read_at"])
    op.create_index("ix_trace_content_reads_session_user_id", "trace_content_reads", ["session_user_id"])

    op.create_table(
        "trace_content_barriers",
        sa.Column("workspace_id", sa.Uuid(), nullable=False),
        sa.Column("generation", sa.BigInteger(), server_default="0", nullable=False),
        sa.ForeignKeyConstraint(["workspace_id"], ["workspace.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("workspace_id"),
    )

    op.create_table(
        "trace_user_barriers",
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("erased_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("user_id"),
    )

    op.create_table(
        "trace_break_glass_reads",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("kind", sa.String(length=16), nullable=False),
        sa.Column("organization_id", sa.Uuid(), nullable=False),
        sa.Column("workspace_id", sa.Uuid(), nullable=False),
        sa.Column("trace_id", sa.String(length=64), nullable=False),
        sa.Column("session_user_id", sa.String(), nullable=True),
        sa.Column("reader_user_id", sa.Uuid(), nullable=True),
        sa.Column("reader_email", sa.String(length=320), nullable=False),
        sa.Column("reason", sa.String(length=1000), nullable=False),
        sa.Column("reference", sa.String(length=200), nullable=True),
        sa.Column("read_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint("kind IN ('organization', 'deployment')", name="check_trace_break_glass_reads_kind"),
        sa.ForeignKeyConstraint(["reader_user_id"], ["user.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_trace_break_glass_reads_organization_read_at", "trace_break_glass_reads", ["organization_id", "read_at"]
    )
    op.create_index("ix_trace_break_glass_reads_read_at", "trace_break_glass_reads", ["read_at"])
    op.create_index("ix_trace_break_glass_reads_reader_user_id", "trace_break_glass_reads", ["reader_user_id"])
    op.create_index("ix_trace_break_glass_reads_session_user_id", "trace_break_glass_reads", ["session_user_id"])


def downgrade() -> None:
    # Destroys the record of every break-glass opening, which is otherwise kept for a
    # year and past the organization's deletion: export it before downgrading past here.
    op.drop_index("ix_trace_break_glass_reads_session_user_id", table_name="trace_break_glass_reads")
    op.drop_index("ix_trace_break_glass_reads_reader_user_id", table_name="trace_break_glass_reads")
    op.drop_index("ix_trace_break_glass_reads_read_at", table_name="trace_break_glass_reads")
    op.drop_index("ix_trace_break_glass_reads_organization_read_at", table_name="trace_break_glass_reads")
    op.drop_table("trace_break_glass_reads")
    op.drop_table("trace_user_barriers")
    op.drop_table("trace_content_barriers")
    op.drop_index("ix_trace_content_reads_session_user_id", table_name="trace_content_reads")
    op.drop_index("ix_trace_content_reads_share_read_at", table_name="trace_content_reads")
    op.drop_index("ix_trace_content_reads_reader_user_id", table_name="trace_content_reads")
    op.drop_index("ix_trace_content_reads_share_id", table_name="trace_content_reads")
    op.drop_table("trace_content_reads")
    op.drop_index("ix_trace_content_shares_shared_by_user_id", table_name="trace_content_shares")
    op.drop_index("ix_trace_content_shares_trace", table_name="trace_content_shares")
    op.drop_table("trace_content_shares")
    op.drop_column("workspace", "trace_content_enabled")
