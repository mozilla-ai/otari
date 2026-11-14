"""ORM tables for organization invitations, and for who invited each member."""

import uuid
from datetime import UTC, datetime
from typing import TypedDict

from sqlalchemy import JSON, ForeignKey, Index, Integer, String, UniqueConstraint, Uuid
from sqlalchemy.orm import Mapped, mapped_column

from gateway.models.base import Base, UtcDateTime
from gateway.models.tenancy import OrganizationMemberRole, WorkspaceMemberRole


class WorkspaceAssignment(TypedDict):
    """The role that an invited person gets in one workspace when they join."""

    workspace_id: str
    role: WorkspaceMemberRole


class OrganizationInvitation(Base):
    """An invitation to join an organization that has not been accepted.

    An accepted or canceled invitation has no row.
    ``next_send_at`` is when the invitation email is next due, and is None when no email is due.
    ``delivery_attempts`` counts the attempts to deliver the invitation email.
    """

    __tablename__ = "organization_invitation"
    __table_args__ = (
        UniqueConstraint("organization_id", "email", name="uq_organization_invitation_organization_email"),
        Index("ix_organization_invitation_expires_at", "expires_at"),
        Index("ix_organization_invitation_next_send_at", "next_send_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    organization_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("organization.id", ondelete="CASCADE"), nullable=False
    )
    email: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[OrganizationMemberRole] = mapped_column(String(32), nullable=False)
    invited_by_user_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid, ForeignKey("user.id", ondelete="SET NULL"), index=True
    )
    token_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    workspace_assignments: Mapped[list[WorkspaceAssignment]] = mapped_column(JSON, default=list, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(UtcDateTime(), nullable=False)
    declined_at: Mapped[datetime | None] = mapped_column(UtcDateTime())
    send_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    last_send_requested_at: Mapped[datetime | None] = mapped_column(UtcDateTime())
    next_send_at: Mapped[datetime | None] = mapped_column(UtcDateTime())
    delivery_attempts: Mapped[int] = mapped_column(Integer, default=0, server_default="0", nullable=False)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime(), default=lambda: datetime.now(UTC), nullable=False)


class OrganizationMembershipOrigin(Base):
    """An audit record of who invited a member, when they were invited, and when they joined."""

    __tablename__ = "organization_membership_origin"

    organization_member_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("organization_member.id", ondelete="CASCADE"), primary_key=True
    )
    invited_by_user_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid, ForeignKey("user.id", ondelete="SET NULL"), index=True
    )
    invited_at: Mapped[datetime] = mapped_column(UtcDateTime(), nullable=False)
    joined_at: Mapped[datetime] = mapped_column(UtcDateTime(), nullable=False)
