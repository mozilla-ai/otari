"""ORM table for what a model server says about the models it serves."""

import uuid
from datetime import UTC, datetime

from sqlalchemy import JSON, CheckConstraint, DateTime, ForeignKey, Index, String, false, text
from sqlalchemy.orm import Mapped, mapped_column

from gateway.models.base import Base

# The inference engines whose model listing says how they serve a model.
SERVED_ENGINES = ("vllm", "sglang")


class ServedModel(Base):
    """One model as the engine serving it last described it, under the provider name it is listed by.

    The engine's word beats models.dev's, because it describes the model as this
    server runs it: a server short of memory serves a 500k model at 256k. Every
    successful listing of the provider replaces its rows, so a model the server
    stopped serving goes with it.

    A row reached through an organization's own provider key names that key and
    its organization, because two organizations can each list a different server
    under the same prefix. Every other row is the deployment's: a configured
    instance or a hosted provider, whose names are the deployment's own.

    ``settings_reported`` says whether the engine told how it was started.
    Without it a ``None`` parser means unknown rather than switched off.
    ``reasoning`` says the listing described the model's reasoning options.
    """

    __tablename__ = "served_models"
    __table_args__ = (
        CheckConstraint(
            "engine IN (" + ", ".join(f"'{engine}'" for engine in SERVED_ENGINES) + ")",
            name="ck_served_models_engine",
        ),
        CheckConstraint("(organization_id IS NULL) = (org_provider_key_id IS NULL)", name="ck_served_models_key_owner"),
        Index(
            "uq_served_models_deployment",
            "provider",
            "model",
            unique=True,
            postgresql_where=text("org_provider_key_id IS NULL"),
            sqlite_where=text("org_provider_key_id IS NULL"),
        ),
        Index(
            "uq_served_models_key",
            "org_provider_key_id",
            "model",
            unique=True,
            postgresql_where=text("org_provider_key_id IS NOT NULL"),
            sqlite_where=text("org_provider_key_id IS NOT NULL"),
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    provider: Mapped[str] = mapped_column(String(255))
    model: Mapped[str] = mapped_column(String(512))
    organization_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("organization.id", ondelete="CASCADE"), default=None, index=True
    )
    org_provider_key_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("org_provider_keys.id", ondelete="CASCADE"), default=None, index=True
    )
    engine: Mapped[str] = mapped_column(String(16))
    context_window: Mapped[int | None] = mapped_column(default=None)
    root: Mapped[str | None] = mapped_column(String(512), default=None)
    settings_reported: Mapped[bool] = mapped_column(default=False)
    reasoning_parser: Mapped[str | None] = mapped_column(String(64), default=None)
    tool_call_parser: Mapped[str | None] = mapped_column(String(64), default=None)
    input_modalities: Mapped[list[str] | None] = mapped_column(JSON, default=None)
    reasoning: Mapped[bool] = mapped_column(default=False, server_default=false())
    reported_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
