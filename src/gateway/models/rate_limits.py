"""ORM table for the rate-limit rules an operator adds through the dashboard."""

from datetime import UTC, datetime

from sqlalchemy import JSON, CheckConstraint, String, func
from sqlalchemy.orm import Mapped, mapped_column

from gateway.models.base import Base, UtcDateTime


class StoredRateLimitRule(Base):
    """A ``rate_limits`` rule added at runtime, counted beside the config-file rules.

    A name a config-file rule already uses is refused on write and skipped on
    load, so ``config.yml`` stays the whole truth about the rules it declares.
    """

    __tablename__ = "rate_limit_rules"
    __table_args__ = (
        # The same set ``RateLimitRule.per`` accepts.
        CheckConstraint(
            "per IN ('deployment', 'key', 'user', 'model')",
            name="ck_rate_limit_rules_per",
        ),
    )

    name: Mapped[str] = mapped_column(String, primary_key=True)
    per: Mapped[str] = mapped_column(String)
    # The instance:model names a per-model rule limits; null for every other rule.
    models: Mapped[list[str] | None] = mapped_column(JSON, default=None)
    # The API key ids the rule is narrowed to; null for a rule that covers every key.
    keys: Mapped[list[str] | None] = mapped_column(JSON, default=None)
    rpm: Mapped[int | None] = mapped_column(default=None)
    tpm: Mapped[int | None] = mapped_column(default=None)
    max_concurrent: Mapped[int | None] = mapped_column(default=None)
    lease_sec: Mapped[float] = mapped_column(default=900.0, server_default="900")
    created_at: Mapped[datetime] = mapped_column(
        UtcDateTime(), default=lambda: datetime.now(UTC), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        UtcDateTime(),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
        server_default=func.now(),
    )
