"""ORM tables for pricing: the deployment price list, organization overrides, and upstream snapshots."""

import uuid
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import JSON, CheckConstraint, DateTime, ForeignKey, Index, String, Text, Uuid
from sqlalchemy.orm import Mapped, mapped_column

from gateway.models.base import Base, UtcDateTime
from gateway.models.money import UsdRate

# The vocabulary of ``ModelPricing.unit`` and ``OrganizationModelPricing.unit``.
# Every per-unit reader (``services/pricing_service`` helpers, the catalog) keys
# on these spellings, and the request schemas validate against them.
PRICING_UNITS: tuple[str, ...] = ("tokens", "requests", "images")

# The vocabulary of ``origin`` on the same two tables.
PRICING_ORIGINS: tuple[str, ...] = ("config", "api", "migration")


class PricingSnapshot(Base):
    """An approved, source-tagged upstream pricing catalog."""

    __tablename__ = "pricing_snapshots"

    source: Mapped[str] = mapped_column(primary_key=True)
    snapshot: Mapped[str] = mapped_column(Text)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )


class PricingSnapshotHistory(Base):
    """One accepted upstream pricing snapshot, kept after a later one replaces it.

    ``pricing_snapshots`` is the current state; this is the record. Written on
    every accept, never updated. ``accepted_by`` says whether an operator
    confirmed it or the scheduled refresh applied it on its own.
    """

    __tablename__ = "pricing_snapshot_history"
    __table_args__ = (Index("ix_pricing_snapshot_history_source_accepted_at", "source", "accepted_at"),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    source: Mapped[str] = mapped_column(String(64))
    accepted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    accepted_by: Mapped[str] = mapped_column(String(32))
    model_count: Mapped[int] = mapped_column()
    snapshot: Mapped[str] = mapped_column(Text)


class ModelPricing(Base):
    """Model pricing configuration."""

    __tablename__ = "model_pricing"

    model_key: Mapped[str] = mapped_column(primary_key=True)
    effective_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        primary_key=True,
        default=lambda: datetime.now(UTC),
    )
    input_price_per_million: Mapped[Decimal] = mapped_column(UsdRate())
    output_price_per_million: Mapped[Decimal] = mapped_column(UsdRate())
    # Nullable: providers without prompt caching (or models without a
    # discounted cache rate) leave these unset. When set, the cost
    # calculation prices cache_read_tokens / cache_write_tokens at these
    # per-million-token rates, following the provider inclusion convention
    # (see log_usage in _pipeline.py).
    cache_read_price_per_million: Mapped[Decimal | None] = mapped_column(UsdRate(), nullable=True)
    cache_write_price_per_million: Mapped[Decimal | None] = mapped_column(UsdRate(), nullable=True)
    cache_write_1h_price_per_million: Mapped[Decimal | None] = mapped_column(UsdRate(), nullable=True)
    # Ordered threshold rules. Each rule applies its supplied rates to the
    # entire request once ``total_input_tokens`` reaches ``min_input_tokens``.
    pricing_tiers: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    # What ``input_price_per_million`` is a rate per: ``tokens`` for a model,
    # ``requests`` for a gateway-run tool or a moderation call, ``images`` for
    # image generation. The rate columns are shared by all three and a reader
    # cannot tell which from the number, so the row says (``PRICING_UNITS``).
    unit: Mapped[str] = mapped_column(String(16), default="tokens", server_default="tokens")
    # Which path wrote the row: ``config`` (the file's ``pricing:`` block),
    # ``api`` (``POST /v1/pricing``), or ``migration``. NULL on a row written
    # before origins were recorded, which is a real answer and not a default.
    origin: Mapped[str | None] = mapped_column(String(16), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert model to dictionary."""
        return {
            "model_key": self.model_key,
            "effective_at": self.effective_at.isoformat() if self.effective_at else None,
            "input_price_per_million": self.input_price_per_million,
            "output_price_per_million": self.output_price_per_million,
            "cache_read_price_per_million": self.cache_read_price_per_million,
            "cache_write_price_per_million": self.cache_write_price_per_million,
            "cache_write_1h_price_per_million": self.cache_write_1h_price_per_million,
            "pricing_tiers": self.pricing_tiers,
            "unit": self.unit,
            "origin": self.origin,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class OrganizationModelPricing(Base):
    """One organization's rate for a model, sitting above the deployment price list.

    ``model_pricing`` carries no tenancy column: it is one price list for the
    whole deployment. This table is the layer above it, so an organization can
    price the models it uses at its own negotiated rates while every other
    organization, and every model it has not overridden, keeps resolving exactly
    as before. Resolution order is override, then deployment row, then the
    genai-prices dataset (`services.pricing_service.find_model_pricing`).

    **Keyed on ``model_key``, not a split provider and model.** The platform's
    equivalent table (`otari-ai` ``organization_model_pricing``) carries
    ``provider`` and ``model`` as separate columns. Here the whole pricing chain
    keys on one ``provider:model`` string, and that string is not always a
    provider and a model: a pricing key names a provider *instance*
    (``home_lab:llama-3``, over ``provider_type: openai``) and sometimes no model
    at all (``otari:web_search``). Splitting it would make an override
    unmatchable for exactly the keys an operator is most likely to have priced by
    hand, so the override keys the same way the row it overrides does.

    **An interval, where ``model_pricing`` carries a version series.** A price in
    ``model_pricing`` is ``(model_key, effective_at)`` and a later row silently
    shadows an earlier one, which is the right shape for a catalog an operator
    re-imports. An override is a commitment for a period, so it carries both ends
    and overlapping periods for one model are refused rather than shadowed
    (`services.organization_pricing_service`). ``effective_to`` NULL means open
    ended.

    **The overlap rule is enforced in the service, not by the database.** The
    natural constraint is a PostgreSQL ``EXCLUDE`` over a ``tstzrange``, and the
    platform has one. SQLite has no exclusion constraint and no range type, and
    it is what the OSS edition ships by default, so a database-side rule would
    hold on one engine and be a comment on the other. The unique index below is
    what both engines can enforce: it stops the exact-duplicate start, which is
    the collision two concurrent writers actually produce, while a partial
    overlap between two simultaneous inserts remains a narrow race the service's
    check can lose. Single-writer configuration traffic, and a wrong rate is
    visible and correctable rather than silent.

    Rates are the exact ``UsdRate`` type ``ModelPricing`` uses, and the two
    tables carry it together (#661, one migration over both). An override
    resolves *into* a transient ``ModelPricing``, so one implementation of the
    cost math prices both, which it could not if they disagreed about the type
    of money.
    """

    __tablename__ = "organization_model_pricing"
    __table_args__ = (
        # One index, doing both jobs, because they want the same columns in the
        # same order. As a constraint it refuses two rows for one key that begin
        # at the same instant, which is the part of the overlap rule either
        # engine can hold (see the class docstring for why the rest is in the
        # service). As an index it serves the resolution lookup: the two equality
        # columns lead, so the request path gets a prefix scan, and
        # ``effective_from`` trails so picking the newest applicable period is
        # index-ordered rather than a sort.
        Index(
            "uq_organization_model_pricing_period_start",
            "organization_id",
            "model_key",
            "effective_from",
            unique=True,
        ),
        # An inverted period would resolve for no instant at all, so it is a
        # storage error rather than a pricing decision. Equal ends are refused
        # too: a zero-width period is the same silent nothing.
        CheckConstraint(
            "effective_to IS NULL OR effective_to > effective_from",
            name="ck_organization_model_pricing_period_ordered",
        ),
        # Negative money prices a request as a credit. The service rejects it
        # with a message; these are the backstop for a writer that is not the
        # service, and they are spelled out per column because a single check
        # over all five would not say which rate was wrong.
        CheckConstraint(
            "input_price_per_million >= 0",
            name="ck_organization_model_pricing_input_non_negative",
        ),
        CheckConstraint(
            "output_price_per_million >= 0",
            name="ck_organization_model_pricing_output_non_negative",
        ),
        CheckConstraint(
            "cache_read_price_per_million IS NULL OR cache_read_price_per_million >= 0",
            name="ck_organization_model_pricing_cache_read_non_negative",
        ),
        CheckConstraint(
            "cache_write_price_per_million IS NULL OR cache_write_price_per_million >= 0",
            name="ck_organization_model_pricing_cache_write_non_negative",
        ),
        CheckConstraint(
            "cache_write_1h_price_per_million IS NULL OR cache_write_1h_price_per_million >= 0",
            name="ck_organization_model_pricing_cache_write_1h_non_negative",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    # CASCADE: an override means nothing without its organization, and usage rows
    # keep their own settled cost.
    # No ``index=True``: the composite lookup index below leads on this column,
    # so a plain one on it would be a second index serving queries the first
    # already answers, paid for on every write.
    organization_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("organization.id", ondelete="CASCADE"), nullable=False
    )
    model_key: Mapped[str] = mapped_column()
    input_price_per_million: Mapped[Decimal] = mapped_column(UsdRate())
    output_price_per_million: Mapped[Decimal] = mapped_column(UsdRate())
    # Nullable for the same reason ``ModelPricing``'s are: a provider without
    # prompt caching, or a model with no discounted cache rate, leaves them unset
    # and the cost calculation falls back the way it already does.
    cache_read_price_per_million: Mapped[Decimal | None] = mapped_column(UsdRate(), nullable=True)
    cache_write_price_per_million: Mapped[Decimal | None] = mapped_column(UsdRate(), nullable=True)
    cache_write_1h_price_per_million: Mapped[Decimal | None] = mapped_column(UsdRate(), nullable=True)
    # Same shape and same ``min_input_tokens`` key as ``ModelPricing``, so the
    # transient row an override resolves into needs no tier translation.
    pricing_tiers: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    # The same two columns ``ModelPricing`` carries, for the same reasons: an
    # override is read as a ``ModelPricing`` and has to say what it is per.
    unit: Mapped[str] = mapped_column(String(16), default="tokens", server_default="tokens")
    origin: Mapped[str | None] = mapped_column(String(16), nullable=True)
    # ``UtcDateTime``, not ``DateTime(timezone=True)``, and this is the one place
    # in this file where that distinction is load-bearing. The flag is a no-op on
    # SQLite, which is what ``core/config.py`` defaults ``database_url`` to, so a
    # plain column reads back naive there and this table's timestamps are the
    # ones that go out over the wire: ``OrganizationModelPricingPublic`` would
    # serialize them with no offset, a browser parses an offset-less date-time as
    # *local*, and the Edit dialog would then round-trip the period shifted by
    # the reader's UTC offset on every save. ``UtcDateTime.impl`` is
    # ``DateTime(timezone=True)``, so the DDL and the migration are unchanged; it
    # normalizes on the way in and stamps UTC on the way out.
    #
    # ``ModelPricing`` above keeps the plain column because nothing renders its
    # ``effective_at`` into an editable control; the transient row an override
    # resolves into is stamped in ``_override_as_model_pricing`` for the cost
    # path, which is a different fix for a different reader.
    effective_from: Mapped[datetime] = mapped_column(
        UtcDateTime(),
        default=lambda: datetime.now(UTC),
    )
    # NULL means open ended, which is the common case: an organization sets a
    # rate and it applies until something replaces it.
    effective_to: Mapped[datetime | None] = mapped_column(UtcDateTime(), default=None)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime(), default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(
        UtcDateTime(),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )
