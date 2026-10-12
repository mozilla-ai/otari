"""Tables the identity domain owns."""

import uuid
from datetime import UTC, datetime

from sqlalchemy import ForeignKey, String, UniqueConstraint, Uuid, func
from sqlalchemy.orm import Mapped, mapped_column

from gateway.models.base import Base, UtcDateTime


class OAuthIdentityLink(Base):
    """A provider identity linked to one account.

    The primary key is the provider's ``(provider, subject)`` pair, which names one person at that provider.
    An account holds at most one link per provider.

    NOTE: ``email_at_link`` and ``email_verified_at_link`` record what the provider said when the link was made.
    They must not be used to look an account up.
    """

    __tablename__ = "oauth_identity_link"
    __table_args__ = (UniqueConstraint("user_id", "provider", name="uq_oauth_identity_link_user_provider"),)

    provider: Mapped[str] = mapped_column(String(50), primary_key=True)
    subject: Mapped[str] = mapped_column(String(255), primary_key=True)
    # The unique constraint above leads with this column, so it also serves as the column's index.
    user_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("user.id", ondelete="CASCADE"), nullable=False)
    email_at_link: Mapped[str | None] = mapped_column(String(255))
    email_verified_at_link: Mapped[bool]
    linked_at: Mapped[datetime] = mapped_column(
        UtcDateTime(), default=lambda: datetime.now(UTC), server_default=func.now()
    )


__all__ = ["OAuthIdentityLink"]
