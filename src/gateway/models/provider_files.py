"""Durable provider accounts and file operations; no file bytes or provider secrets."""

import uuid
from datetime import datetime
from typing import Literal

from sqlalchemy import BigInteger, CheckConstraint, Column, Index, String, UniqueConstraint
from sqlmodel import Field, SQLModel

from gateway.models.base import CreatedAtMixin, PrimaryKeyMixin, UpdatedAtMixin, UtcDateTime

CredentialSource = Literal["organization_key", "hosted_backend"]
AccountStatus = Literal["active", "retiring", "retired"]
BindingState = Literal["pending_upload", "active", "pending_cleanup", "deleted"]
OutputOperationState = Literal["active", "revoked", "completed"]


class ProviderAccountGeneration(SQLModel, PrimaryKeyMixin, CreatedAtMixin, table=True):
    __tablename__ = "provider_account_generations"
    __table_args__ = (
        CheckConstraint(
            "credential_source IN ('organization_key', 'hosted_backend')", name="ck_provider_account_source"
        ),
        CheckConstraint("status IN ('active', 'retiring', 'retired')", name="ck_provider_account_status"),
        UniqueConstraint(
            "organization_id",
            "credential_source",
            "credential_ref",
            "generation",
            name="uq_provider_account_generation",
        ),
    )

    provider: str = Field(default="anthropic", max_length=32)
    credential_source: CredentialSource = Field(sa_column=Column(String(32), nullable=False))
    credential_ref: str = Field(max_length=255, index=True)
    organization_id: uuid.UUID = Field(index=True)
    upstream_identity_ciphertext: str | None = None
    generation: int = Field(default=1)
    status: AccountStatus = Field(default="active", sa_column=Column(String(16), nullable=False, index=True))
    retired_at: datetime | None = Field(default=None, sa_type=UtcDateTime)
    lease_id: uuid.UUID | None = None
    lease_token_hash: str | None = Field(default=None, max_length=64)
    lease_gateway_id: str | None = Field(default=None, max_length=255)
    lease_deadline: datetime | None = Field(default=None, sa_type=UtcDateTime)


class ProviderFileBinding(SQLModel, PrimaryKeyMixin, CreatedAtMixin, UpdatedAtMixin, table=True):
    __tablename__ = "provider_file_bindings"
    __table_args__ = (
        CheckConstraint(
            "state IN ('pending_upload', 'active', 'pending_cleanup', 'deleted')", name="ck_provider_file_state"
        ),
        UniqueConstraint("provider_account_generation_id", "provider_file_id", name="uq_provider_file_account_id"),
        Index("ix_provider_files_owner_page", "workspace_id", "user_id", "state", "created_at", "id"),
        Index("ix_provider_files_cleanup", "state", "cleanup_after"),
    )

    provider_file_id: str | None = Field(default=None, max_length=255)
    provider_account_generation_id: uuid.UUID = Field(
        foreign_key="provider_account_generations.id", ondelete="RESTRICT", index=True
    )
    output_operation_id: uuid.UUID | None = Field(
        default=None, foreign_key="provider_file_output_operations.id", ondelete="RESTRICT", index=True
    )
    # Ownership survives tenant removal so cleanup never relies on a deleted row.
    organization_id: uuid.UUID = Field(index=True)
    workspace_id: uuid.UUID = Field(index=True)
    user_id: str = Field(max_length=255, index=True)
    encrypted_metadata: str | None = None
    purpose: str | None = Field(default=None, max_length=255)
    provider_created_at: datetime | None = Field(default=None, sa_type=UtcDateTime)
    size_bytes: int = Field(default=0, sa_type=BigInteger)
    downloadable: bool = False
    expires_at: datetime = Field(sa_type=UtcDateTime)
    provider_expires_at: datetime | None = Field(default=None, sa_type=UtcDateTime)
    operation_deadline: datetime = Field(sa_type=UtcDateTime)
    initiating_gateway_id: str = Field(max_length=255)
    cleanup_token_hash: str = Field(max_length=64)
    provider_outcome_unknown: bool = False
    state: BindingState = Field(default="pending_upload", sa_column=Column(String(32), nullable=False))
    cleanup_reason: str | None = Field(default=None, max_length=32)
    cleanup_attempts: int = 0
    cleanup_after: datetime | None = Field(default=None, sa_type=UtcDateTime)
    deleted_at: datetime | None = Field(default=None, sa_type=UtcDateTime)
    lease_id: uuid.UUID | None = Field(default=None, index=True)


class ProviderFileOutputOperation(SQLModel, PrimaryKeyMixin, CreatedAtMixin, table=True):
    __tablename__ = "provider_file_output_operations"
    __table_args__ = (
        CheckConstraint("state IN ('active', 'revoked', 'completed')", name="ck_provider_file_output_state"),
    )

    provider_account_generation_id: uuid.UUID = Field(
        foreign_key="provider_account_generations.id", ondelete="RESTRICT", index=True
    )
    organization_id: uuid.UUID = Field(index=True)
    workspace_id: uuid.UUID = Field(index=True)
    user_id: str = Field(max_length=255, index=True)
    initiating_gateway_id: str = Field(max_length=255)
    request_id: str = Field(max_length=255)
    attempt_id: str = Field(max_length=255)
    cleanup_token_hash: str = Field(max_length=64)
    deadline: datetime = Field(sa_type=UtcDateTime)
    state: OutputOperationState = Field(default="active", sa_column=Column(String(16), nullable=False))
    reserved_files: int
    reserved_bytes: int = Field(sa_type=BigInteger)


class ProviderFileRateWindow(SQLModel, table=True):
    __tablename__ = "provider_file_rate_windows"

    workspace_id: uuid.UUID = Field(primary_key=True)
    user_id: str = Field(primary_key=True, max_length=255)
    window: int
    count: int = 0
