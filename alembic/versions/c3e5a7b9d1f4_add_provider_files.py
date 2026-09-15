"""Add provider-native file bindings and cleanup operations.

Revision ID: c3e5a7b9d1f4
Revises: b2d4f6a8c0e2
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
import sqlmodel

revision: str = "c3e5a7b9d1f4"
down_revision: str | None = "b2d4f6a8c0e2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "provider_account_generations",
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("provider", sqlmodel.sql.sqltypes.AutoString(length=32), nullable=False),
        sa.Column("credential_source", sqlmodel.sql.sqltypes.AutoString(length=32), nullable=False),
        sa.Column("credential_ref", sqlmodel.sql.sqltypes.AutoString(length=255), nullable=False),
        sa.Column("organization_id", sa.Uuid(), nullable=False),
        sa.Column("upstream_identity_ciphertext", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("generation", sa.Integer(), nullable=False),
        sa.Column("status", sqlmodel.sql.sqltypes.AutoString(length=16), nullable=False),
        sa.Column("retired_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("lease_id", sa.Uuid(), nullable=True),
        sa.Column("lease_token_hash", sqlmodel.sql.sqltypes.AutoString(length=64), nullable=True),
        sa.Column("lease_gateway_id", sqlmodel.sql.sqltypes.AutoString(length=255), nullable=True),
        sa.Column("lease_deadline", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "organization_id",
            "credential_source",
            "credential_ref",
            "generation",
            name="uq_provider_account_generation",
        ),
    )
    op.create_index(
        op.f("ix_provider_account_generations_credential_ref"),
        "provider_account_generations",
        ["credential_ref"],
        unique=False,
    )
    op.create_index(
        op.f("ix_provider_account_generations_organization_id"),
        "provider_account_generations",
        ["organization_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_provider_account_generations_status"), "provider_account_generations", ["status"], unique=False
    )
    op.create_table(
        "provider_file_rate_windows",
        sa.Column("workspace_id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sqlmodel.sql.sqltypes.AutoString(length=255), nullable=False),
        sa.Column("window", sa.Integer(), nullable=False),
        sa.Column("count", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("workspace_id", "user_id"),
    )
    op.create_table(
        "provider_file_output_operations",
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("provider_account_generation_id", sa.Uuid(), nullable=False),
        sa.Column("organization_id", sa.Uuid(), nullable=False),
        sa.Column("workspace_id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sqlmodel.sql.sqltypes.AutoString(length=255), nullable=False),
        sa.Column("initiating_gateway_id", sqlmodel.sql.sqltypes.AutoString(length=255), nullable=False),
        sa.Column("request_id", sqlmodel.sql.sqltypes.AutoString(length=255), nullable=False),
        sa.Column("attempt_id", sqlmodel.sql.sqltypes.AutoString(length=255), nullable=False),
        sa.Column("cleanup_token_hash", sqlmodel.sql.sqltypes.AutoString(length=64), nullable=False),
        sa.Column("deadline", sa.DateTime(timezone=True), nullable=False),
        sa.Column("state", sqlmodel.sql.sqltypes.AutoString(length=16), nullable=False),
        sa.Column("reserved_files", sa.Integer(), nullable=False),
        sa.Column("reserved_bytes", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ["provider_account_generation_id"], ["provider_account_generations.id"], ondelete="RESTRICT"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_provider_file_output_operations_organization_id"),
        "provider_file_output_operations",
        ["organization_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_provider_file_output_operations_provider_account_generation_id"),
        "provider_file_output_operations",
        ["provider_account_generation_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_provider_file_output_operations_user_id"), "provider_file_output_operations", ["user_id"], unique=False
    )
    op.create_index(
        op.f("ix_provider_file_output_operations_workspace_id"),
        "provider_file_output_operations",
        ["workspace_id"],
        unique=False,
    )
    op.create_table(
        "provider_file_bindings",
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("provider_file_id", sqlmodel.sql.sqltypes.AutoString(length=255), nullable=True),
        sa.Column("provider_account_generation_id", sa.Uuid(), nullable=False),
        sa.Column("output_operation_id", sa.Uuid(), nullable=True),
        sa.Column("organization_id", sa.Uuid(), nullable=False),
        sa.Column("workspace_id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sqlmodel.sql.sqltypes.AutoString(length=255), nullable=False),
        sa.Column("encrypted_metadata", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("downloadable", sa.Boolean(), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("provider_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("operation_deadline", sa.DateTime(timezone=True), nullable=False),
        sa.Column("initiating_gateway_id", sqlmodel.sql.sqltypes.AutoString(length=255), nullable=False),
        sa.Column("cleanup_token_hash", sqlmodel.sql.sqltypes.AutoString(length=64), nullable=False),
        sa.Column("provider_outcome_unknown", sa.Boolean(), nullable=False),
        sa.Column("state", sqlmodel.sql.sqltypes.AutoString(length=32), nullable=False),
        sa.Column("cleanup_reason", sqlmodel.sql.sqltypes.AutoString(length=32), nullable=True),
        sa.Column("cleanup_attempts", sa.Integer(), nullable=False),
        sa.Column("cleanup_after", sa.DateTime(timezone=True), nullable=True),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("lease_id", sa.Uuid(), nullable=True),
        sa.ForeignKeyConstraint(["output_operation_id"], ["provider_file_output_operations.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(
            ["provider_account_generation_id"], ["provider_account_generations.id"], ondelete="RESTRICT"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("provider_account_generation_id", "provider_file_id", name="uq_provider_file_account_id"),
    )
    op.create_index(op.f("ix_provider_file_bindings_lease_id"), "provider_file_bindings", ["lease_id"], unique=False)
    op.create_index(
        op.f("ix_provider_file_bindings_organization_id"), "provider_file_bindings", ["organization_id"], unique=False
    )
    op.create_index(
        op.f("ix_provider_file_bindings_output_operation_id"),
        "provider_file_bindings",
        ["output_operation_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_provider_file_bindings_provider_account_generation_id"),
        "provider_file_bindings",
        ["provider_account_generation_id"],
        unique=False,
    )
    op.create_index(op.f("ix_provider_file_bindings_user_id"), "provider_file_bindings", ["user_id"], unique=False)
    op.create_index(
        op.f("ix_provider_file_bindings_workspace_id"), "provider_file_bindings", ["workspace_id"], unique=False
    )
    op.create_index("ix_provider_files_cleanup", "provider_file_bindings", ["state", "cleanup_after"], unique=False)
    op.create_index(
        "ix_provider_files_owner_page",
        "provider_file_bindings",
        ["workspace_id", "user_id", "state", "created_at", "id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_provider_files_owner_page", table_name="provider_file_bindings")
    op.drop_index("ix_provider_files_cleanup", table_name="provider_file_bindings")
    op.drop_index(op.f("ix_provider_file_bindings_workspace_id"), table_name="provider_file_bindings")
    op.drop_index(op.f("ix_provider_file_bindings_user_id"), table_name="provider_file_bindings")
    op.drop_index(op.f("ix_provider_file_bindings_provider_account_generation_id"), table_name="provider_file_bindings")
    op.drop_index(op.f("ix_provider_file_bindings_output_operation_id"), table_name="provider_file_bindings")
    op.drop_index(op.f("ix_provider_file_bindings_organization_id"), table_name="provider_file_bindings")
    op.drop_index(op.f("ix_provider_file_bindings_lease_id"), table_name="provider_file_bindings")
    op.drop_table("provider_file_bindings")
    op.drop_index(op.f("ix_provider_file_output_operations_workspace_id"), table_name="provider_file_output_operations")
    op.drop_index(op.f("ix_provider_file_output_operations_user_id"), table_name="provider_file_output_operations")
    op.drop_index(
        op.f("ix_provider_file_output_operations_provider_account_generation_id"),
        table_name="provider_file_output_operations",
    )
    op.drop_index(
        op.f("ix_provider_file_output_operations_organization_id"), table_name="provider_file_output_operations"
    )
    op.drop_table("provider_file_output_operations")
    op.drop_table("provider_file_rate_windows")
    op.drop_index(op.f("ix_provider_account_generations_status"), table_name="provider_account_generations")
    op.drop_index(op.f("ix_provider_account_generations_organization_id"), table_name="provider_account_generations")
    op.drop_index(op.f("ix_provider_account_generations_credential_ref"), table_name="provider_account_generations")
    op.drop_table("provider_account_generations")
