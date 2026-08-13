"""Small SQLAlchemy expression helpers shared across layers.

Lives in ``core`` so both the API routes (read filters) and the services (bulk
mutation selections) can build the same condition from the same code: the two must
agree exactly, because a bulk delete re-derives its target set server-side from the
filters the operator was shown.
"""

from datetime import UTC, datetime
from typing import Any, cast

from sqlalchemy import ColumnElement

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
