"""Unit tests for UTC pinning of the usage window bounds.

The date query params and selection bodies advertise ISO 8601, so a caller that
omits the offset hands in a naive datetime. asyncpg encodes ``timestamptz`` with
``astimezone``, which reads a naive value as the process's local time, so the same
bound would select a different set of rows per deployment. `/v1/usage/summary`
routes through `_resolve_window` and was already pinned; the list, count, and bulk
mutation paths do not, so they pin the bound themselves.

The count an operator confirms and the delete that re-derives its target set from
the same body have to mean the same instant, so both sides are asserted here.
"""

from datetime import UTC, datetime, timedelta, timezone
from typing import Any

from gateway.api.routes.usage import _usage_filters
from gateway.services.usage_admin_service import UsageSelection, _selection_conditions

_NAIVE_START = datetime(2026, 8, 12, 8, 0)
_NAIVE_END = datetime(2026, 8, 13, 8, 0)


def _bound_values(conditions: list[Any]) -> list[datetime]:
    """The datetime each timestamp comparison binds, in the order they were built.

    A selection carries non-comparison conditions too (the imported-only guards),
    so anything that does not bind a datetime is skipped rather than unwrapped.
    """
    bounds = []
    for condition in conditions:
        value = getattr(getattr(condition, "right", None), "value", None)
        if isinstance(value, datetime):
            bounds.append(value)
    return bounds


def test_list_and_count_filters_pin_a_naive_bound_to_utc() -> None:
    conditions = _usage_filters(
        start_date=_NAIVE_START,
        end_date=_NAIVE_END,
        user_id=None,
        status=None,
        model=None,
        endpoint=None,
        scope=None,
    )
    assert _bound_values(conditions) == [
        _NAIVE_START.replace(tzinfo=UTC),
        _NAIVE_END.replace(tzinfo=UTC),
    ]


def test_list_and_count_filters_leave_an_offset_bound_alone() -> None:
    """A caller that sent an offset meant that instant, so it passes through."""
    offset = timezone(timedelta(hours=-4))
    aware_start = _NAIVE_START.replace(tzinfo=offset)
    conditions = _usage_filters(
        start_date=aware_start,
        end_date=None,
        user_id=None,
        status=None,
        model=None,
        endpoint=None,
        scope=None,
    )
    assert _bound_values(conditions) == [aware_start]


def test_bulk_selection_pins_the_same_bound_as_the_count() -> None:
    """The delete re-derives its target set server-side, so it has to agree."""
    selection = UsageSelection(by_filter=True, start_date=_NAIVE_START, end_date=_NAIVE_END)
    assert _bound_values(_selection_conditions(selection)) == [
        _NAIVE_START.replace(tzinfo=UTC),
        _NAIVE_END.replace(tzinfo=UTC),
    ]
