"""Unit tests for the ``_as_utc`` helper in ``gateway.api.deps``.

Regression coverage for the SQLite-naive-datetime bug surfaced during
manual QA on PR #58: ``DateTime(timezone=True)`` columns come back naive
from SQLite, so any code that subtracts them from ``datetime.now(UTC)``
crashed with ``TypeError: can't subtract offset-naive and offset-aware
datetimes``. The helper normalises the read side; these tests pin the
contract.
"""

from datetime import UTC, datetime, timedelta, timezone

from gateway.api.deps import _as_utc


def test_none_passes_through() -> None:
    assert _as_utc(None) is None


def test_naive_datetime_is_promoted_to_utc() -> None:
    naive = datetime(2026, 5, 19, 12, 0, 0)
    out = _as_utc(naive)
    assert out is not None
    assert out.tzinfo is UTC


def test_aware_utc_datetime_unchanged() -> None:
    aware = datetime(2026, 5, 19, 12, 0, 0, tzinfo=UTC)
    out = _as_utc(aware)
    assert out is aware


def test_aware_non_utc_datetime_unchanged() -> None:
    # Important: we don't convert non-UTC aware datetimes — the gateway
    # only ever writes UTC, so a non-UTC aware value would be a deliberate
    # caller decision. The helper just guarantees `tzinfo is not None`.
    plus_two = timezone(timedelta(hours=2))
    aware = datetime(2026, 5, 19, 12, 0, 0, tzinfo=plus_two)
    out = _as_utc(aware)
    assert out is aware


def test_subtracting_normalised_naive_does_not_raise() -> None:
    # The original bug: subtracting a naive value from `datetime.now(UTC)`
    # raises TypeError. After normalisation it must succeed.
    naive_last_used = datetime(2026, 5, 19, 12, 0, 0)
    normalised = _as_utc(naive_last_used)
    assert normalised is not None
    delta = datetime.now(UTC) - normalised
    assert delta.total_seconds() >= 0
