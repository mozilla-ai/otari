"""add routing policies projects traces

Revision ID: b7c9d0e1f2a3
Revises: a1b2c3d4e5f6
Create Date: 2026-05-27 08:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b7c9d0e1f2a3"
down_revision: str | Sequence[str] | None = "a1b2c3d4e5f6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "routing_policies",
        sa.Column("policy_id", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("strategy", sa.String(), nullable=False),
        sa.Column("config", sa.JSON(), nullable=False),
        sa.Column("is_default", sa.Boolean(), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("policy_id"),
    )
    op.create_index(op.f("ix_routing_policies_is_default"), "routing_policies", ["is_default"], unique=False)
    op.create_index(op.f("ix_routing_policies_status"), "routing_policies", ["status"], unique=False)

    op.create_table(
        "routing_policy_revisions",
        sa.Column("revision_id", sa.String(), nullable=False),
        sa.Column("policy_id", sa.String(), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("action", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("strategy", sa.String(), nullable=False),
        sa.Column("config", sa.JSON(), nullable=False),
        sa.Column("is_default", sa.Boolean(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("change_note", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("revision_id"),
    )
    op.create_index(
        "ix_routing_policy_revisions_policy_id_created_at",
        "routing_policy_revisions",
        ["policy_id", "created_at"],
        unique=False,
    )
    op.create_index(
        "ix_routing_policy_revisions_policy_id_revision",
        "routing_policy_revisions",
        ["policy_id", "revision"],
        unique=True,
    )
    op.create_index(
        op.f("ix_routing_policy_revisions_policy_id"),
        "routing_policy_revisions",
        ["policy_id"],
        unique=False,
    )

    op.create_table(
        "projects",
        sa.Column("project_id", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=True),
        sa.Column("routing_policy_id", sa.String(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["routing_policy_id"],
            ["routing_policies.policy_id"],
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("project_id"),
    )
    op.create_index(op.f("ix_projects_is_active"), "projects", ["is_active"], unique=False)
    op.create_index(op.f("ix_projects_routing_policy_id"), "projects", ["routing_policy_id"], unique=False)

    op.create_table(
        "route_traces",
        sa.Column("trace_id", sa.String(), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("api_key_id", sa.String(), nullable=True),
        sa.Column("user_id", sa.String(), nullable=True),
        sa.Column("project_id", sa.String(), nullable=True),
        sa.Column("policy_id", sa.String(), nullable=True),
        sa.Column("requested_model", sa.String(), nullable=False),
        sa.Column("selected_model", sa.String(), nullable=True),
        sa.Column("selected_provider", sa.String(), nullable=True),
        sa.Column("strategy", sa.String(), nullable=True),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("error_message", sa.String(), nullable=True),
        sa.Column("selected_reason", sa.String(), nullable=True),
        sa.Column("estimated_prompt_tokens", sa.Integer(), nullable=True),
        sa.Column("estimated_output_tokens", sa.Integer(), nullable=True),
        sa.Column("estimated_cost", sa.Float(), nullable=True),
        sa.Column("fallback_enabled", sa.Boolean(), nullable=False),
        sa.Column("policy_source", sa.String(), nullable=True),
        sa.Column("tags", sa.JSON(), nullable=False),
        sa.Column("candidates", sa.JSON(), nullable=False),
        sa.Column("attempts", sa.JSON(), nullable=False),
        sa.ForeignKeyConstraint(["api_key_id"], ["api_keys.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["policy_id"], ["routing_policies.policy_id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["project_id"], ["projects.project_id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["user_id"], ["users.user_id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("trace_id"),
    )
    op.create_index(op.f("ix_route_traces_api_key_id"), "route_traces", ["api_key_id"], unique=False)
    op.create_index(op.f("ix_route_traces_policy_id"), "route_traces", ["policy_id"], unique=False)
    op.create_index(op.f("ix_route_traces_project_id"), "route_traces", ["project_id"], unique=False)
    op.create_index(
        "ix_route_traces_project_id_timestamp",
        "route_traces",
        ["project_id", "timestamp"],
        unique=False,
    )
    op.create_index(op.f("ix_route_traces_timestamp"), "route_traces", ["timestamp"], unique=False)
    op.create_index(op.f("ix_route_traces_user_id"), "route_traces", ["user_id"], unique=False)
    op.create_index(
        "ix_route_traces_user_id_timestamp",
        "route_traces",
        ["user_id", "timestamp"],
        unique=False,
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_route_traces_user_id_timestamp", table_name="route_traces")
    op.drop_index(op.f("ix_route_traces_user_id"), table_name="route_traces")
    op.drop_index(op.f("ix_route_traces_timestamp"), table_name="route_traces")
    op.drop_index("ix_route_traces_project_id_timestamp", table_name="route_traces")
    op.drop_index(op.f("ix_route_traces_project_id"), table_name="route_traces")
    op.drop_index(op.f("ix_route_traces_policy_id"), table_name="route_traces")
    op.drop_index(op.f("ix_route_traces_api_key_id"), table_name="route_traces")
    op.drop_table("route_traces")
    op.drop_index(op.f("ix_projects_routing_policy_id"), table_name="projects")
    op.drop_index(op.f("ix_projects_is_active"), table_name="projects")
    op.drop_table("projects")
    op.drop_index(op.f("ix_routing_policy_revisions_policy_id"), table_name="routing_policy_revisions")
    op.drop_index("ix_routing_policy_revisions_policy_id_revision", table_name="routing_policy_revisions")
    op.drop_index("ix_routing_policy_revisions_policy_id_created_at", table_name="routing_policy_revisions")
    op.drop_table("routing_policy_revisions")
    op.drop_index(op.f("ix_routing_policies_status"), table_name="routing_policies")
    op.drop_index(op.f("ix_routing_policies_is_default"), table_name="routing_policies")
    op.drop_table("routing_policies")
