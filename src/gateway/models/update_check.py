"""ORM table for the check for new otari releases and security fixes."""

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, Boolean, CheckConstraint, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

from gateway.models.base import Base, UtcDateTime

STATE_ROW_ID = 1


class UpdateCheckState(Base):
    """The one row holding the install id, the operator's answer, the schedule and the last answer received.

    ``next_check_at`` is also the claim: every replica's worker polls this row,
    and a check is sent only by the replica whose conditional update moved it forward.
    """

    __tablename__ = "update_check_state"
    __table_args__ = (CheckConstraint(f"id = {STATE_ROW_ID}", name="ck_update_check_state_one_row"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=STATE_ROW_ID)
    # Random, and never derived from anything about the host. Sent with every check.
    install_id: Mapped[uuid.UUID] = mapped_column(default=uuid.uuid4)
    # What an operator answered in the dashboard. Null until one does; ``update_check`` in config overrides it.
    enabled: Mapped[bool | None] = mapped_column(Boolean, default=None)
    answered_at: Mapped[datetime | None] = mapped_column(UtcDateTime(), default=None)
    next_check_at: Mapped[datetime | None] = mapped_column(UtcDateTime(), default=None)
    # Consecutive failed checks, which set how long the next retry waits.
    failures: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    last_checked_at: Mapped[datetime | None] = mapped_column(UtcDateTime(), default=None)
    # The version this deployment ran when ``latest_version`` and ``advisories`` were received.
    checked_version: Mapped[str | None] = mapped_column(String, default=None)
    latest_version: Mapped[str | None] = mapped_column(String, default=None)
    release_url: Mapped[str | None] = mapped_column(String, default=None)
    advisories: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    updated_at: Mapped[datetime] = mapped_column(
        UtcDateTime(),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
        server_default=func.now(),
    )
