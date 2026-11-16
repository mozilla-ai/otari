"""ORM table for stored prompts."""

import uuid
from datetime import UTC, datetime

from sqlalchemy import DateTime, String, Text, Uuid
from sqlalchemy.orm import Mapped, mapped_column

from gateway.models.base import Base, UtcDateTime


class StoredPrompt(Base):
    """A named prompt stored at runtime through the dashboard.

    Prompts are operator-scoped: they live at the deployment level and are
    available to every workspace. A caller references a prompt by name instead
    of embedding the text inline, so the content can change without redeploying.
    """

    __tablename__ = "stored_prompts"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid, primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        UtcDateTime, default=lambda: datetime.now(UTC)
    )
    updated_at: Mapped[datetime] = mapped_column(
        UtcDateTime,
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )
