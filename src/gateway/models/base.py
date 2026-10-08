"""The declarative base, and the column types and mixins the table modules share."""

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import DateTime, func
from sqlalchemy.engine.interfaces import Dialect
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.types import TypeDecorator
from sqlmodel import Field, SQLModel


class Base(DeclarativeBase):
    """Base class for SQLAlchemy models.

    Shares ``SQLModel.metadata`` so the reconciled control plane's SQLModel
    tables (`gateway.models.tenancy`) and the gateway's own declarative tables
    land in one collection. That is what lets Alembic keep a single
    ``target_metadata``, and ``create_all``/``drop_all`` cover the whole schema,
    without either style having to know the other exists. The two classes keep
    separate declarative *registries*, so a same-named model on either side
    (``User``, during the strangle) resolves unambiguously.
    """

    metadata = SQLModel.metadata


class UtcDateTime(TypeDecorator[datetime]):
    """A timestamp that reads back UTC-aware on every engine.

    ``DateTime(timezone=True)`` alone is not enough, and the gap is the whole
    reason this exists. PostgreSQL honors it and hands back an aware value;
    SQLite has no timestamp type at all, so SQLAlchemy stores an ISO string and
    the flag is a no-op, and a value written as ``datetime.now(UTC)`` reads back
    with ``tzinfo=None``. A naive datetime then serializes with no offset, and a
    browser parses an offset-less timestamp as **local** time, so every tenancy
    timestamp in the dashboard would be wrong by the deployment's UTC offset on
    the engine the OSS edition ships by default.

    Both directions are handled: an aware value is normalized to UTC before it
    is stored, so a caller in another zone cannot write a wall-clock time that
    means something else, and a naive value read back is stamped UTC, because
    UTC is what everything here writes.

    The rendered DDL is exactly ``impl``'s, so this changes no migration and
    ``compare_metadata`` stays clean.
    """

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        if value.utcoffset() is None:
            # Refused rather than assumed. Reading a naive value back as UTC is
            # safe, because UTC is what everything here writes; writing one is
            # not, because the engines disagree about what it means. PostgreSQL
            # interprets it in the *session* time zone, so the same value lands
            # as a different instant depending on who connected, while SQLite
            # stores the wall clock as written. Silently picking one is how a
            # timestamp ends up hours off with nothing to show for it.
            msg = "A tenancy timestamp must be timezone-aware; got a naive datetime"
            raise ValueError(msg)
        return value.astimezone(UTC)

    def process_result_value(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        if value is not None and value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value


def _timestamp_field(*, default: Any = None, default_factory: Any = None, column_kwargs: dict[str, Any]) -> Any:
    """Build a timezone-aware timestamp field.

    The type arrives as ``sa_type`` rather than a ready-made ``sa_column``,
    because a ``Column`` instance declared on a mixin cannot be attached to more than one table.
    ``sa_type`` plus kwargs lets SQLModel build a fresh column per model.
    """
    if default_factory is not None:
        return Field(
            default_factory=default_factory,
            sa_type=UtcDateTime(),
            sa_column_kwargs=column_kwargs,
        )
    return Field(
        default=default,
        sa_type=UtcDateTime(),
        sa_column_kwargs=column_kwargs,
    )


class PrimaryKeyMixin(SQLModel):
    """A UUID primary key, rendered as CHAR(32) on SQLite and native on PostgreSQL."""

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)


class CreatedAtMixin(SQLModel):
    """Creation timestamp, defaulted in Python and in the database."""

    created_at: datetime = _timestamp_field(
        default_factory=lambda: datetime.now(UTC),
        column_kwargs={"server_default": func.now()},
    )


class UpdatedAtMixin(SQLModel):
    """Last-modification timestamp, stamped by the database on update.

    ``default=None`` and not merely a nullable annotation: without an explicit
    default the field is *required* on the pydantic side, which a table class
    hides (table models skip construction validation) and any schema inheriting
    this mixin would not.
    """

    updated_at: datetime | None = _timestamp_field(default=None, column_kwargs={"onupdate": func.now()})


def has_passed(moment: datetime | None) -> bool:
    """Whether a stored timestamp is now or earlier.

    SQLite hands a timestamp back naive; it was written as UTC, so it is read as UTC.
    """
    if moment is None:
        return False
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return moment <= datetime.now(UTC)
