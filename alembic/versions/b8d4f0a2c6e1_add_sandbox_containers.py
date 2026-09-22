"""Add ``sandbox_containers``, the leases a caller resumes a code-execution sandbox by.

One row per container id the gateway has handed out and not yet let go of. It
maps the id to the provider's own session handle and binds it to the user and
workspace that leased it, which is the whole tenant-isolation story for reuse:
a resume by anyone else is answered as an unknown id.

Two expiry columns because there are two clocks. ``expires_at`` is the idle
clock, moved forward on every use and equal to what the provider was told to
hold the sandbox for. ``hard_expires_at`` is set at the first lease and never
moves, so a client resuming forever still gives the sandbox back. The sweep
selects on either, hence the index on ``expires_at``; ``hard_expires_at`` is
always later than the idle clock at write time, so the one index serves both.

``in_use_until`` is the third, and it is a claim rather than a clock: one
request at a time may hold a sandbox, because two sharing one workspace would
interleave their code and each collect the other's files. A request takes the
claim with a conditional update at admission and gives it back when it records
the lease; the timestamp is what releases a claim whose gateway died holding it.

Revision ID: b8d4f0a2c6e1
Revises: b4d7f1a9c2e6
Create Date: 2026-09-22
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "b8d4f0a2c6e1"
down_revision: str | Sequence[str] | None = "b4d7f1a9c2e6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLE = "sandbox_containers"


def upgrade() -> None:
    op.create_table(
        _TABLE,
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("user_id", sa.String(), sa.ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False),
        # CASCADE, unlike the durable rows that reference a workspace: a lease is
        # ephemeral and nothing depends on it, so it must not block a deletion for
        # as long as its hard clock runs.
        sa.Column("workspace_id", sa.Uuid(), sa.ForeignKey("workspace.id", ondelete="CASCADE"), nullable=False),
        sa.Column("provider", sa.String(), nullable=False),
        sa.Column("provider_session_id", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("hard_expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("in_use_until", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_sandbox_containers_user_id", _TABLE, ["user_id"])
    op.create_index("ix_sandbox_containers_workspace_id", _TABLE, ["workspace_id"])
    op.create_index("ix_sandbox_containers_expires_at", _TABLE, ["expires_at"])


def downgrade() -> None:
    op.drop_index("ix_sandbox_containers_expires_at", table_name=_TABLE)
    op.drop_index("ix_sandbox_containers_workspace_id", table_name=_TABLE)
    op.drop_index("ix_sandbox_containers_user_id", table_name=_TABLE)
    op.drop_table(_TABLE)
