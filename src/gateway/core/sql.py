"""Small SQLAlchemy expression helpers shared across layers.

Lives in ``core`` so both the API routes (read filters) and the services (bulk
mutation selections) can build the same condition from the same code: the two must
agree exactly, because a bulk delete re-derives its target set server-side from the
filters the operator was shown.
"""

from datetime import UTC, datetime
from typing import Any, Literal, cast

from sqlalchemy import ColumnElement, func
from sqlalchemy.ext.asyncio import AsyncSession

# How many values one repeatable entity filter (model / user / API key) may carry.
# Far above what a chart can distinguish (a stacked series folds past eight groups),
# so it never binds a real comparison; it is there to keep a caller from posting an
# unbounded IN list. Shared by the read endpoints and the bulk-mutation selection
# body for the same reason ``match_any`` is shared: the count an operator confirms
# is taken over the read filters and the mutation re-derives its target set from the
# body, so a value set one side rejects and the other accepts breaks that agreement.
MAX_FILTER_VALUES = 50


def match_any(column: Any, value: str | list[str]) -> ColumnElement[bool]:
    """Match a column against one value or any of several.

    A single value stays an equality test so it uses the column's index the way a
    one-value filter always did; several become an ``IN``. An empty list would match
    nothing, so callers skip the condition entirely rather than emitting ``IN ()``.
    """
    if isinstance(value, str):
        condition = column == value
    elif len(value) == 1:
        condition = column == value[0]
    else:
        condition = column.in_(value)
    return cast("ColumnElement[bool]", condition)


def utc_bound(value: datetime | None) -> datetime | None:
    """Pin an offset-less window bound to UTC.

    The date query params and selection bodies advertise ISO 8601, which parses to a
    naive value when the caller omits the offset. The driver then resolves it against
    the process's local timezone (asyncpg encodes ``timestamptz`` with
    ``astimezone``, which reads a naive datetime as local), so the same bound would
    select a different set of rows per deployment. Shared for the usual reason: a
    count an operator confirms and the delete that re-derives its target set from the
    same bound have to mean the same instant.
    """
    if value is None or value.tzinfo is not None:
        return value
    return value.replace(tzinfo=UTC)


# Time-series granularity. Declared here rather than beside one of its readers
# because the read endpoints, the telemetry storage port, and its adapters all
# have to agree on the grid; a chart built from two of them lines up only if
# they bucket the same way.
BucketGrain = Literal["hour", "day"]


def dialect_name(db: AsyncSession) -> str:
    """The name of the dialect this session's engine speaks."""
    bind = db.get_bind()
    return bind.dialect.name


def bucket_expr(dialect: str, bucket: BucketGrain, column: Any) -> Any:
    """A SQL expression that truncates ``column`` to the bucket start, in UTC.

    PostgreSQL ``date_trunc`` honors the session ``TimeZone``, so UTC is pinned
    with ``AT TIME ZONE 'UTC'`` (``func.timezone``) rather than trusting engine
    config; otherwise buckets would silently shift per deployment and break
    across DST. SQLite ``strftime`` already normalizes any stored offset to UTC.
    ``bucket`` is a validated ``Literal`` (never raw client text), so there is
    no injection surface.
    """
    if dialect == "sqlite":
        fmt = "%Y-%m-%dT%H:00:00Z" if bucket == "hour" else "%Y-%m-%dT00:00:00Z"
        return func.strftime(fmt, column)
    # PostgreSQL (and anything else that speaks date_trunc).
    return func.date_trunc(bucket, func.timezone("UTC", column))


def canonical_bucket(value: Any, bucket: BucketGrain) -> str:
    """Normalize a bucket key to canonical ISO-8601 UTC (``YYYY-MM-DDTHH:00:00Z``).

    SQLite already returns that string; PostgreSQL returns a (naive, UTC) datetime.
    """
    if isinstance(value, str):
        return value
    dt: datetime = value
    fmt = "%Y-%m-%dT%H:00:00Z" if bucket == "hour" else "%Y-%m-%dT00:00:00Z"
    return dt.strftime(fmt)
