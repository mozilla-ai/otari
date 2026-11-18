"""ORM table for deployment settings the dashboard can change at runtime."""

from datetime import UTC, datetime

from sqlalchemy import DateTime
from sqlalchemy.orm import Mapped, mapped_column

from gateway.models.base import Base

# The ``RuntimeSetting.key`` naming the deployment's bootstrap operator. Deliberately not a
# SETTABLE_KEY, so runtime_settings_service ignores it as it ignores the master-key hash and the
# dashboard-session marker.
BOOTSTRAP_IDENTITY_KEY = "tenancy_bootstrap_user_id"


class RuntimeSetting(Base):
    """A persisted override for a runtime-toggleable config flag.

    A small key/value store for the handful of settings the dashboard can flip
    at runtime (model discovery, default pricing). When a key is present it wins
    over the config-file/env value and is applied on startup; when absent the
    config value stands. The value is stored as a string ("true"/"false") so the
    table can hold future non-boolean settings without a schema change.
    """

    __tablename__ = "runtime_settings"

    key: Mapped[str] = mapped_column(primary_key=True)
    value: Mapped[str] = mapped_column()
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )
