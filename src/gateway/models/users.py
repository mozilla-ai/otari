"""ORM table for the gateway's billing identity.

API keys, budgets, and usage rows attach to this ``User``. Gotcha: another model
class named ``User`` is the dashboard sign-in identity.
"""

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import JSON, BigInteger, DateTime, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from gateway.models.base import Base
from gateway.models.money import UsdCost


class User(Base):
    """User/Customer model for end-user tracking."""

    __tablename__ = "users"

    user_id: Mapped[str] = mapped_column(primary_key=True)
    alias: Mapped[str | None] = mapped_column()
    # The spend ledger, exact to the micro-dollar like the ``usage_logs`` rows
    # that sum into it (mozilla-ai/otari#691). As a float it drifted: four
    # completions whose settled costs were each exact left this at
    # 0.6619999999999999, and the drift accumulated across every reconcile until
    # the budget reset.
    spend: Mapped[Decimal] = mapped_column(UsdCost(), default=Decimal(0))
    # The committed amount is ``spend + reserved``, where ``reserved`` holds requests that have not settled.
    reserved: Mapped[Decimal] = mapped_column(UsdCost(), default=Decimal(0), server_default="0")
    # The token and request counters, gated by the same budget's ``token_limit``
    # and ``request_limit`` the way the pair above is gated by ``max_budget``.
    # Each axis names itself rather than extending the bare ``spend``/``reserved``
    # pair, which is USD and predates them.
    current_tokens: Mapped[int] = mapped_column(BigInteger(), default=0, server_default="0")
    reserved_tokens: Mapped[int] = mapped_column(BigInteger(), default=0, server_default="0")
    current_requests: Mapped[int] = mapped_column(BigInteger(), default=0, server_default="0")
    reserved_requests: Mapped[int] = mapped_column(BigInteger(), default=0, server_default="0")
    # Indexed: the budgets list groups users by this column to build each budget's
    # usage rollup, so an unindexed FK turns that page into a users table scan.
    budget_id: Mapped[str | None] = mapped_column(ForeignKey("budgets.budget_id"), index=True)
    # Default model access-list every one of this user's keys inherits when the
    # key has no list of its own. null = unrestricted, [] = deny all, else
    # canonical instance:model entries (see services/model_access.py). A key may
    # narrow this default but never broaden it (validated on key write).
    allowed_models: Mapped[list[str] | None] = mapped_column(JSON)
    budget_started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    next_budget_reset_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    blocked: Mapped[bool] = mapped_column(default=False)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)

    budget = relationship("Budget", back_populates="users")
    api_keys = relationship("APIKey", back_populates="user", passive_deletes=True)
    usage_logs = relationship("UsageLog", back_populates="user", passive_deletes=True)
    reset_logs = relationship("BudgetResetLog", back_populates="user", passive_deletes=True)

    def to_dict(self) -> dict[str, Any]:
        """Convert model to dictionary."""
        return {
            "user_id": self.user_id,
            "alias": self.alias,
            "spend": self.spend,
            "reserved": self.reserved,
            "budget_id": self.budget_id,
            "allowed_models": self.allowed_models,
            "budget_started_at": self.budget_started_at.isoformat() if self.budget_started_at else None,
            "next_budget_reset_at": self.next_budget_reset_at.isoformat() if self.next_budget_reset_at else None,
            "blocked": self.blocked,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "metadata": self.metadata_,
        }
