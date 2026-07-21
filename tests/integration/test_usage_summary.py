"""Integration tests for the aggregated usage summary + CSV export endpoints.

Runs against whatever backend the suite is configured for (SQLite by default,
PostgreSQL when TEST_DATABASE_URL / testcontainers provides one), so the same
assertions pin the cross-dialect bucketing and reconciliation contract.
"""

from __future__ import annotations

import csv
import io
import uuid
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from gateway.models.entities import UsageLog, User

SUMMARY_PATH = "/v1/usage/summary"
CSV_PATH = "/v1/usage/summary.csv"


def _ensure_user(db: Session, user_id: str) -> None:
    if db.query(User).filter(User.user_id == user_id).first() is None:
        db.add(User(user_id=user_id, alias=user_id, spend=0.0, blocked=False))
        db.flush()


def _make_log(
    db: Session,
    *,
    user_id: str | None = "u1",
    timestamp: datetime,
    api_key_id: str | None = None,
    model: str = "gpt-4",
    total_tokens: int | None = 15,
    cost: float | None = 0.01,
    status: str = "success",
    latency_ms: int | None = None,
) -> None:
    if user_id is not None:
        _ensure_user(db, user_id)
    db.add(
        UsageLog(
            id=str(uuid.uuid4()),
            user_id=user_id,
            api_key_id=api_key_id,
            timestamp=timestamp,
            model=model,
            provider="openai",
            endpoint="/v1/chat/completions",
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=total_tokens,
            cost=cost,
            status=status,
            error_message="boom" if status == "error" else None,
            latency_ms=latency_ms,
        )
    )


def test_summary_requires_master_key(client: TestClient) -> None:
    assert client.get(SUMMARY_PATH).status_code == 401
    assert client.get(CSV_PATH).status_code == 401


def test_summary_empty_range_is_all_zero(client: TestClient, master_key_header: dict[str, str]) -> None:
    resp = client.get(SUMMARY_PATH, headers=master_key_header, params={"user_id": "nobody"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["totals"] == {
        "cost": 0.0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "request_count": 0,
        "error_count": 0,
        "avg_latency_ms": None,
    }
    assert body["by_model"] == []
    assert body["by_user"] == []
    assert body["by_api_key"] == []
    assert body["series"] == []


def test_summary_totals_and_error_count(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    now = datetime.now(UTC) - timedelta(hours=1)
    _make_log(db_session, user_id="tot", timestamp=now, cost=0.10, total_tokens=100, latency_ms=100)
    _make_log(db_session, user_id="tot", timestamp=now, cost=0.20, total_tokens=200, latency_ms=200)
    _make_log(db_session, user_id="tot", timestamp=now, cost=0.30, total_tokens=300, status="error", latency_ms=None)
    db_session.commit()

    body = client.get(SUMMARY_PATH, headers=master_key_header, params={"user_id": "tot"}).json()
    totals = body["totals"]
    assert totals["cost"] == pytest.approx(0.60)
    assert totals["total_tokens"] == 600
    assert totals["request_count"] == 3
    assert totals["error_count"] == 1
    # avg over the two non-null latencies; the null row is excluded by AVG.
    assert totals["avg_latency_ms"] == pytest.approx(150.0)


def test_summary_null_cost_and_tokens_coalesce_to_zero(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    now = datetime.now(UTC) - timedelta(hours=1)
    _make_log(db_session, user_id="nul", timestamp=now, cost=None, total_tokens=None, status="error")
    db_session.commit()

    totals = client.get(SUMMARY_PATH, headers=master_key_header, params={"user_id": "nul"}).json()["totals"]
    assert totals["cost"] == 0.0
    assert totals["total_tokens"] == 0
    assert totals["request_count"] == 1


def test_summary_breakdowns_reconcile_with_totals(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    now = datetime.now(UTC) - timedelta(hours=2)
    for model, cost in (("gpt-4", 0.10), ("gpt-4", 0.10), ("claude", 0.05)):
        _make_log(db_session, user_id="rec", timestamp=now, model=model, cost=cost, total_tokens=10)
    db_session.commit()

    body = client.get(SUMMARY_PATH, headers=master_key_header, params={"user_id": "rec"}).json()
    grand = body["totals"]["cost"]
    for dimension in ("by_model", "by_user", "by_api_key"):
        assert sum(r["cost"] for r in body[dimension]) == pytest.approx(grand), dimension
        assert sum(r["requests"] for r in body[dimension]) == body["totals"]["request_count"], dimension
    # by_model is ordered by spend desc: gpt-4 (0.20) before claude (0.05).
    assert [r["key"] for r in body["by_model"]] == ["gpt-4", "claude"]


def test_summary_top_n_fold_reconciles(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    now = datetime.now(UTC) - timedelta(hours=1)
    # 105 distinct models, one request each; the endpoint caps at 100 + an "other".
    for idx in range(105):
        _make_log(db_session, user_id="fold", timestamp=now, model=f"m{idx:03d}", cost=0.01, total_tokens=1)
    db_session.commit()

    body = client.get(SUMMARY_PATH, headers=master_key_header, params={"user_id": "fold"}).json()
    by_model = body["by_model"]
    assert len(by_model) == 101  # 100 named + 1 folded
    other = by_model[-1]
    assert other["key"] is None
    assert other["requests"] == 5
    assert sum(r["requests"] for r in by_model) == body["totals"]["request_count"] == 105
    assert sum(r["cost"] for r in by_model) == pytest.approx(body["totals"]["cost"])


def test_summary_series_day_buckets_are_canonical_utc(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    day1 = datetime(2025, 1, 1, 6, 0, tzinfo=UTC)
    day1_late = datetime(2025, 1, 1, 23, 30, tzinfo=UTC)
    day2 = datetime(2025, 1, 2, 1, 0, tzinfo=UTC)
    for ts in (day1, day1_late, day2):
        _make_log(db_session, user_id="ser", timestamp=ts, cost=0.01, total_tokens=1)
    db_session.commit()

    body = client.get(
        SUMMARY_PATH,
        headers=master_key_header,
        # Bracket the fixed dates with an explicit window so the max-span clamp
        # (which pulls a >366d start forward) can't exclude these rows.
        params={
            "user_id": "ser",
            "bucket": "day",
            "start_date": "2024-12-31T00:00:00Z",
            "end_date": "2025-01-03T00:00:00Z",
        },
    ).json()
    series = {p["bucket_start"]: p["requests"] for p in body["series"]}
    # Both same-UTC-day rows collapse into one bucket keyed at UTC midnight.
    assert series == {"2025-01-01T00:00:00Z": 2, "2025-01-02T00:00:00Z": 1}


def test_summary_series_hour_buckets(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    base = datetime(2025, 3, 1, 10, 15, tzinfo=UTC)
    _make_log(db_session, user_id="hr", timestamp=base, cost=0.01, total_tokens=1)
    _make_log(db_session, user_id="hr", timestamp=base + timedelta(minutes=30), cost=0.01, total_tokens=1)
    _make_log(db_session, user_id="hr", timestamp=base + timedelta(hours=1), cost=0.01, total_tokens=1)
    db_session.commit()

    body = client.get(
        SUMMARY_PATH,
        headers=master_key_header,
        params={
            "user_id": "hr",
            "bucket": "hour",
            "start_date": "2025-03-01T00:00:00Z",
            "end_date": "2025-03-02T00:00:00Z",
        },
    ).json()
    series = {p["bucket_start"]: p["requests"] for p in body["series"]}
    assert series == {"2025-03-01T10:00:00Z": 2, "2025-03-01T11:00:00Z": 1}


def test_summary_defaults_to_last_30_days(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    recent = datetime.now(UTC) - timedelta(days=2)
    ancient = datetime.now(UTC) - timedelta(days=90)
    _make_log(db_session, user_id="win", timestamp=recent, cost=0.05, total_tokens=1)
    _make_log(db_session, user_id="win", timestamp=ancient, cost=0.99, total_tokens=1)
    db_session.commit()

    # No start_date -> only the within-30d row counts.
    body = client.get(SUMMARY_PATH, headers=master_key_header, params={"user_id": "win"}).json()
    assert body["totals"]["request_count"] == 1
    assert body["totals"]["cost"] == pytest.approx(0.05)


def test_summary_rejects_unknown_bucket(client: TestClient, master_key_header: dict[str, str]) -> None:
    resp = client.get(SUMMARY_PATH, headers=master_key_header, params={"bucket": "week"})
    assert resp.status_code == 422


def test_csv_export_shape_and_reconciliation(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    now = datetime.now(UTC) - timedelta(hours=1)
    _make_log(db_session, user_id="csv", timestamp=now, model="gpt-4", cost=0.10, total_tokens=10)
    _make_log(db_session, user_id="csv", timestamp=now, model="claude", cost=0.20, total_tokens=20)
    db_session.commit()

    resp = client.get(CSV_PATH, headers=master_key_header, params={"user_id": "csv"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/csv")
    assert "attachment" in resp.headers["content-disposition"]

    rows = list(csv.reader(io.StringIO(resp.text)))
    assert rows[0] == ["dimension", "key", "cost", "tokens", "requests"]
    model_rows = [r for r in rows[1:] if r[0] == "model"]
    assert {r[1] for r in model_rows} == {"gpt-4", "claude"}
    assert sum(float(r[2]) for r in model_rows) == pytest.approx(0.30)


def test_csv_export_guards_formula_injection(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    now = datetime.now(UTC) - timedelta(hours=1)
    # A model name crafted to run as a formula if opened in a spreadsheet.
    _make_log(db_session, user_id="inj", timestamp=now, model="=cmd|'/c calc'!A1", cost=0.01, total_tokens=1)
    db_session.commit()

    resp = client.get(CSV_PATH, headers=master_key_header, params={"user_id": "inj"})
    rows = list(csv.reader(io.StringIO(resp.text)))
    injected = [r for r in rows if r[0] == "model" and "cmd" in r[1]][0]
    # The dangerous leading '=' is neutralized with a leading quote.
    assert injected[1].startswith("'=")
