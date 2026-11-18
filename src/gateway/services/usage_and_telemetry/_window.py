"""The window, bucket grid and failure classes every usage summary shares, so two summaries agree."""

from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from typing import TypeVar, cast

from gateway.exceptions.usage_and_telemetry_exceptions import UsageTagFilterError
from gateway.schemas.usage_and_telemetry import Bucket, ErrorClass, UsageSeriesPoint

# A summary is range-bounded, unlike the raw list. Absent a start_date it looks
# back this far; a wider explicit window is clamped to the hard cap so a single
# request can never turn into an unbounded full-table scan on a growing log.
_DEFAULT_SUMMARY_LOOKBACK = timedelta(days=30)
_MAX_SUMMARY_SPAN = timedelta(days=366)

# Upper bound on zero-filled series points, so a pathological range/bucket combo
# (e.g. hourly over a year) cannot balloon the payload; beyond it a summary
# returns the sparse populated buckets instead.
MAX_SERIES_POINTS = 1000

# How many groups a grouped time series carries before the remainder folds into
# "other". Eight is the ceiling a stacked chart can keep legible (and the size of
# the dashboard's fixed categorical palette); the breakdown tables, not the chart,
# are the place to read a longer tail.
SERIES_TOP_N = 8

_PointT = TypeVar("_PointT")


def resolve_window(start_date: datetime | None, end_date: datetime | None) -> tuple[datetime, datetime]:
    """Clamp the requested window to a bounded, forward-ordered range.

    A summary must never scan an unbounded log: absent a start we look back
    ``_DEFAULT_SUMMARY_LOOKBACK``; a span wider than ``_MAX_SUMMARY_SPAN`` has its
    start pulled forward so the aggregates stay bounded by the timestamp index.

    An offset-less ISO datetime (which the query params advertise as valid) parses
    to a naive value; ``now(UTC)`` is aware. Comparing or subtracting the two would
    raise, so naive bounds are assumed UTC and made aware first.
    """
    if start_date is not None and start_date.tzinfo is None:
        start_date = start_date.replace(tzinfo=UTC)
    if end_date is not None and end_date.tzinfo is None:
        end_date = end_date.replace(tzinfo=UTC)
    end = end_date or datetime.now(UTC)
    start = start_date if start_date is not None else end - _DEFAULT_SUMMARY_LOOKBACK
    if start > end:
        start = end
    if end - start > _MAX_SUMMARY_SPAN:
        start = end - _MAX_SUMMARY_SPAN
    return start, end


def _empty_usage_point(bucket_start: str) -> UsageSeriesPoint:
    return UsageSeriesPoint(bucket_start=bucket_start, cost=0.0, tokens=0, requests=0)


def dense_series(
    start: datetime,
    end: datetime,
    bucket: Bucket,
    populated: Mapping[str, _PointT],
    empty: Callable[[str], _PointT] | None = None,
) -> list[_PointT]:
    """Fill every bucket in ``[floor(start), end)`` so the chart's x-axis is linear
    in time. ``GROUP BY`` omits empty buckets, so without this a sparse range (say
    usage on day 1 and day 20 of a month) would render as two adjacent bars and
    misread the trend. Falls back to the sparse buckets past ``MAX_SERIES_POINTS``.
    An empty window (no rows at all) returns an empty series, not a wall of zeros.

    ``empty`` builds the zero point for a gap; it is what lets another series type
    (the agent-telemetry summary's) share this fill rather than restate it.
    """
    if not populated:
        return []
    make_empty = cast("Callable[[str], _PointT]", empty or _empty_usage_point)
    step = timedelta(hours=1) if bucket == "hour" else timedelta(days=1)
    if bucket == "hour":
        cursor = start.replace(minute=0, second=0, microsecond=0)
        fmt = "%Y-%m-%dT%H:00:00Z"
    else:
        cursor = start.replace(hour=0, minute=0, second=0, microsecond=0)
        fmt = "%Y-%m-%dT00:00:00Z"
    points: list[_PointT] = []
    while cursor < end:
        if len(points) >= MAX_SERIES_POINTS:
            return [populated[key] for key in sorted(populated)]
        key = cursor.strftime(fmt)
        points.append(populated.get(key) or make_empty(key))
        cursor += step
    return points


def parse_tag_filters(tag: list[str] | None) -> dict[str, list[str]]:
    """Parse ``key:value`` tag filters into the values each key may hold."""
    by_key: dict[str, list[str]] = {}
    for item in tag or []:
        key, sep, value = item.partition(":")
        if not sep or not key:
            raise UsageTagFilterError
        by_key.setdefault(key, []).append(value)
    return by_key


def error_class_for(status_code: int | None) -> ErrorClass:
    """Coarse display bucket for a failure's HTTP status code.

    Two kinds of code reach this column: the status a provider returned, and the
    status the gateway itself refused with, since #465 records those rejections
    too and each row carries the code it returned (403 for a blocked or
    over-budget user, a user/key mismatch, or a model outside a key's allow-list,
    402 for missing pricing, 400 for a selector that no longer resolves).

    So ``auth`` currently covers both a provider rejecting the gateway's
    credentials and the gateway rejecting the caller, and a **budget denial files
    as ``auth``**, because ``reserve_budget`` refuses with 403 and the code is all
    this function sees. Splitting budget and permission refusals into their own
    class needs a discriminator the row does not reliably carry (``provider`` is
    NULL only on the gates that refuse before the selector resolves, not on the
    budget gate), and these names are dashboard-visible, so that stays a
    deliberate follow-up rather than something guessed at here.
    """
    if status_code is None:
        return "unknown"
    if status_code == 402:
        return "pricing"
    if status_code == 429:
        return "rate_limit"
    if status_code in (401, 403, 407):
        return "auth"
    if 500 <= status_code <= 599:
        return "provider_error"
    if 400 <= status_code <= 499:
        return "client_error"
    return "unknown"
