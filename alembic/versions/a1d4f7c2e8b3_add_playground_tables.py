"""Add the Playground's five tables.

The Playground is back (otari-ai#1947, mozilla-ai/otari#663): an in-product chat
page with model comparison, a saved-transcript history, and pinned models. Its
completions run through the ordinary pipeline and write the ordinary
``usage_logs`` row, so nothing here is on the request path. These tables hold
only what the page remembers between visits:

- ``playground_consent``: content-retention consent, one row per identity,
  keyed on ``user_id`` because an identity has one answer or none. Absence
  means not agreed, so nothing provisions a row until somebody agrees.
- ``playground_conversation`` / ``playground_message``: a saved single-panel
  transcript and its turns, ordered by an explicit ``position`` rather than by
  insertion clock, with a unique constraint over ``(conversation_id, position)``
  so a transcript cannot read differently on two loads.
- ``playground_comparison``: one rated A/B exchange, both answers in full,
  which is what the comparison consent flag discloses.
- ``playground_favorite_model``: the model keys an identity pins, per
  workspace, with ``position`` carrying the order the client sent.

Every owner FK is ``user.id`` with CASCADE, not ``users.user_id``: the tenancy
identity is who wrote this content, and unlike the soft-deleted spend row it has
no ledger duty to outlive the account. Workspace FKs are CASCADE for the reason
``org_provider_keys`` gives, that these are workspace-owned resources rather
than durable request-plane history; the request-plane row for the same
completion is the ``usage_logs`` row, which keeps its RESTRICT.

Purely additive, with no legacy rows to migrate: the hosted original's tables
were dropped on the platform side (otari-ai#1920) and never existed here. So
there is no data step and the downgrade is five drops.

Revision ID: a1d4f7c2e8b3
Revises: f1c4a8e2d6b9
Create Date: 2026-09-14 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a1d4f7c2e8b3"
down_revision: str | Sequence[str] | None = "f1c4a8e2d6b9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_CONVERSATION_RECENT_INDEX = "ix_playground_conversation_owner_recent"
_COMPARISON_RECENT_INDEX = "ix_playground_comparison_owner_recent"
_FAVORITE_POSITION_INDEX = "ix_playground_favorite_model_owner_position"


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "playground_consent",
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("store_conversations", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("store_comparisons", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["user.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("user_id"),
    )

    op.create_table(
        "playground_conversation",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("workspace_id", sa.Uuid(), nullable=False),
        sa.Column("title", sa.String(length=200), nullable=False),
        sa.Column("model", sa.String(length=512), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["user.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["workspace_id"], ["workspace.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_playground_conversation_user_id"), "playground_conversation", ["user_id"], unique=False)
    op.create_index(
        op.f("ix_playground_conversation_workspace_id"),
        "playground_conversation",
        ["workspace_id"],
        unique=False,
    )
    # The list query's whole predicate plus its sort. Without the trailing sort
    # column, "my conversations in this workspace, newest first" is a filesort
    # over everything the predicate matched.
    op.create_index(
        _CONVERSATION_RECENT_INDEX,
        "playground_conversation",
        ["user_id", "workspace_id", "created_at"],
        unique=False,
    )

    op.create_table(
        "playground_message",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("conversation_id", sa.Uuid(), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("role", sa.String(length=16), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("reasoning", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["conversation_id"], ["playground_conversation.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("conversation_id", "position", name="uq_playground_message_conversation_position"),
    )
    op.create_index(
        op.f("ix_playground_message_conversation_id"), "playground_message", ["conversation_id"], unique=False
    )

    op.create_table(
        "playground_comparison",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("workspace_id", sa.Uuid(), nullable=False),
        sa.Column("user_question", sa.Text(), nullable=False),
        sa.Column("model_a", sa.String(length=512), nullable=False),
        sa.Column("model_b", sa.String(length=512), nullable=False),
        sa.Column("model_a_answer", sa.Text(), nullable=False),
        sa.Column("model_b_answer", sa.Text(), nullable=False),
        sa.Column("preference", sa.String(length=16), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["user.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["workspace_id"], ["workspace.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_playground_comparison_user_id"), "playground_comparison", ["user_id"], unique=False)
    op.create_index(
        op.f("ix_playground_comparison_workspace_id"),
        "playground_comparison",
        ["workspace_id"],
        unique=False,
    )
    op.create_index(
        _COMPARISON_RECENT_INDEX,
        "playground_comparison",
        ["user_id", "workspace_id", "created_at"],
        unique=False,
    )

    op.create_table(
        "playground_favorite_model",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("workspace_id", sa.Uuid(), nullable=False),
        sa.Column("model_key", sa.String(length=512), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["user.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["workspace_id"], ["workspace.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id", "workspace_id", "model_key", name="uq_playground_favorite_model_owner_key"),
    )
    op.create_index(
        op.f("ix_playground_favorite_model_user_id"),
        "playground_favorite_model",
        ["user_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_playground_favorite_model_workspace_id"),
        "playground_favorite_model",
        ["workspace_id"],
        unique=False,
    )
    op.create_index(
        _FAVORITE_POSITION_INDEX,
        "playground_favorite_model",
        ["user_id", "workspace_id", "position"],
        unique=False,
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(_FAVORITE_POSITION_INDEX, table_name="playground_favorite_model")
    op.drop_index(op.f("ix_playground_favorite_model_workspace_id"), table_name="playground_favorite_model")
    op.drop_index(op.f("ix_playground_favorite_model_user_id"), table_name="playground_favorite_model")
    op.drop_table("playground_favorite_model")

    op.drop_index(_COMPARISON_RECENT_INDEX, table_name="playground_comparison")
    op.drop_index(op.f("ix_playground_comparison_workspace_id"), table_name="playground_comparison")
    op.drop_index(op.f("ix_playground_comparison_user_id"), table_name="playground_comparison")
    op.drop_table("playground_comparison")

    op.drop_index(op.f("ix_playground_message_conversation_id"), table_name="playground_message")
    op.drop_table("playground_message")

    op.drop_index(_CONVERSATION_RECENT_INDEX, table_name="playground_conversation")
    op.drop_index(op.f("ix_playground_conversation_workspace_id"), table_name="playground_conversation")
    op.drop_index(op.f("ix_playground_conversation_user_id"), table_name="playground_conversation")
    op.drop_table("playground_conversation")

    op.drop_table("playground_consent")
