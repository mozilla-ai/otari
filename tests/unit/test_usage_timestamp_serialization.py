"""Unit tests for usage timestamp serialization (UTC-aware ISO on the wire).

`usage_logs.timestamp` is timezone-aware, but SQLite returns it naive, so a plain
`isoformat()` drops the offset and a browser reads a recent UTC time in its own zone
(showing "0s ago"). `_utc_iso` normalizes a naive value to UTC; an already-aware
value (Postgres) must pass through unchanged, so there is no cross-engine drift.
"""

from datetime import UTC, datetime, timedelta, timezone

from gateway.api.routes.usage import _utc_iso


def test_naive_is_treated_as_utc() -> None:
    # What SQLite hands back: aware column, but no tzinfo on read.
    assert _utc_iso(datetime(2026, 7, 23, 12, 0, 0)) == "2026-07-23T12:00:00+00:00"


def test_aware_utc_is_unchanged() -> None:
    # What Postgres hands back: already UTC-aware. Output must equal plain isoformat,
    # so the fix is a no-op on Postgres deployments.
    value = datetime(2026, 7, 23, 12, 0, 0, tzinfo=UTC)
    assert _utc_iso(value) == value.isoformat() == "2026-07-23T12:00:00+00:00"


def test_aware_nonutc_offset_is_preserved() -> None:
    # A non-UTC aware value keeps its offset (still an unambiguous instant a browser
    # parses correctly); the helper only fills in a missing offset.
    value = datetime(2026, 7, 23, 8, 0, 0, tzinfo=timezone(timedelta(hours=-4)))
    assert _utc_iso(value) == "2026-07-23T08:00:00-04:00"
