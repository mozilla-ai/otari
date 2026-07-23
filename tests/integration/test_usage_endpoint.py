"""Integration tests for the bulk usage endpoint."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from gateway.models.entities import UsageLog, User

USAGE_PATH = "/v1/usage"


def _ensure_user(db: Session, user_id: str) -> None:
    if db.query(User).filter(User.user_id == user_id).first() is None:
        db.add(User(user_id=user_id, alias=user_id, spend=0.0, blocked=False))
        db.flush()


def _make_log(
    db: Session,
    *,
    user_id: str,
    timestamp: datetime,
    api_key_id: str | None = None,
    model: str = "gpt-4",
    provider: str | None = "openai",
    endpoint: str = "/v1/chat/completions",
    prompt_tokens: int | None = 10,
    completion_tokens: int | None = 5,
    total_tokens: int | None = 15,
    cost: float | None = 0.01,
    status: str = "success",
    error_message: str | None = None,
    latency_ms: int | None = None,
    log_id: str | None = None,
) -> UsageLog:
    _ensure_user(db, user_id)
    log = UsageLog(
        id=log_id or str(uuid.uuid4()),
        user_id=user_id,
        api_key_id=api_key_id,
        timestamp=timestamp,
        model=model,
        provider=provider,
        endpoint=endpoint,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cost=cost,
        status=status,
        error_message=error_message,
        latency_ms=latency_ms,
    )
    db.add(log)
    return log


def test_list_usage_filters_by_api_key(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    k1 = client.post("/v1/keys", json={"key_name": "k1"}, headers=master_key_header).json()["id"]
    k2 = client.post("/v1/keys", json={"key_name": "k2"}, headers=master_key_header).json()["id"]
    ts = datetime(2026, 7, 1, 9, 0, tzinfo=UTC)
    _make_log(db_session, user_id="u", timestamp=ts, api_key_id=k1, log_id="log-k1")
    _make_log(db_session, user_id="u", timestamp=ts, api_key_id=k2, log_id="log-k2")
    db_session.commit()

    listed = client.get(USAGE_PATH, params={"api_key_id": k1}, headers=master_key_header)
    assert [r["id"] for r in listed.json()] == ["log-k1"]
    count = client.get("/v1/usage/count", params={"api_key_id": k1}, headers=master_key_header)
    assert count.json()["total"] == 1


def test_list_usage_requires_master_key(client: TestClient) -> None:
    response = client.get(USAGE_PATH)
    assert response.status_code == 401


def test_list_usage_returns_empty_when_no_logs(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    response = client.get(USAGE_PATH, headers=master_key_header)
    assert response.status_code == 200
    assert response.json() == []


def test_list_usage_orders_newest_first(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session: Session,
) -> None:
    now = datetime(2025, 1, 1, 12, 0, tzinfo=UTC)
    older = now - timedelta(hours=1)
    _make_log(db_session, user_id="order-user", timestamp=older)
    _make_log(db_session, user_id="order-user", timestamp=now)
    db_session.commit()

    response = client.get(USAGE_PATH, headers=master_key_header)
    assert response.status_code == 200
    data = response.json()
    assert [entry["timestamp"] for entry in data] == [now.isoformat(), older.isoformat()]


def test_list_usage_filter_by_start_date(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session: Session,
) -> None:
    earlier = datetime(2025, 2, 1, 10, 0, tzinfo=UTC)
    later = datetime(2025, 2, 1, 12, 0, tzinfo=UTC)
    _make_log(db_session, user_id="start-user", timestamp=earlier)
    _make_log(db_session, user_id="start-user", timestamp=later)
    db_session.commit()

    response = client.get(
        USAGE_PATH,
        headers=master_key_header,
        params={"start_date": later.isoformat()},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["timestamp"] == later.isoformat()


def test_list_usage_filter_by_end_date(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session: Session,
) -> None:
    earlier = datetime(2025, 3, 1, 8, 0, tzinfo=UTC)
    later = datetime(2025, 3, 1, 9, 0, tzinfo=UTC)
    _make_log(db_session, user_id="end-user", timestamp=earlier)
    _make_log(db_session, user_id="end-user", timestamp=later)
    db_session.commit()

    response = client.get(
        USAGE_PATH,
        headers=master_key_header,
        params={"end_date": later.isoformat()},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["timestamp"] == earlier.isoformat()


def test_list_usage_filter_by_time_range(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session: Session,
) -> None:
    base = datetime(2025, 4, 1, 0, 0, tzinfo=UTC)
    before = base - timedelta(hours=1)
    inside = base + timedelta(minutes=30)
    after = base + timedelta(hours=2)
    _make_log(db_session, user_id="range-user", timestamp=before)
    _make_log(db_session, user_id="range-user", timestamp=base)
    _make_log(db_session, user_id="range-user", timestamp=inside)
    _make_log(db_session, user_id="range-user", timestamp=after)
    db_session.commit()

    response = client.get(
        USAGE_PATH,
        headers=master_key_header,
        params={
            "start_date": base.isoformat(),
            "end_date": (base + timedelta(hours=2)).isoformat(),
        },
    )
    assert response.status_code == 200
    data = response.json()
    timestamps = [entry["timestamp"] for entry in data]
    assert timestamps == [inside.isoformat(), base.isoformat()]


def test_list_usage_filter_by_user_id(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session: Session,
) -> None:
    timestamp = datetime(2025, 5, 1, 15, 0, tzinfo=UTC)
    _make_log(db_session, user_id="filter-a", timestamp=timestamp)
    _make_log(db_session, user_id="filter-b", timestamp=timestamp)
    db_session.commit()

    response = client.get(
        USAGE_PATH,
        headers=master_key_header,
        params={"user_id": "filter-b"},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["user_id"] == "filter-b"


def test_list_usage_pagination(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session: Session,
) -> None:
    base = datetime(2025, 6, 1, 12, 0, tzinfo=UTC)
    timestamps = []
    for idx in range(5):
        ts = base + timedelta(minutes=idx)
        timestamps.append(ts)
        _make_log(db_session, user_id="pager", timestamp=ts)
    db_session.commit()

    response = client.get(
        USAGE_PATH,
        headers=master_key_header,
        params={"skip": 1, "limit": 2},
    )
    assert response.status_code == 200
    data = response.json()
    expected_order = [ts.isoformat() for ts in sorted(timestamps, reverse=True)]
    assert [entry["timestamp"] for entry in data] == expected_order[1:3]


def test_list_usage_response_shape(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session: Session,
) -> None:
    timestamp = datetime(2025, 7, 1, 9, 30, tzinfo=UTC)
    log = _make_log(
        db_session,
        user_id="shape-user",
        timestamp=timestamp,
        api_key_id=None,
        model="gpt-4o",
        provider="openai",
        endpoint="/custom",
        prompt_tokens=42,
        completion_tokens=7,
        total_tokens=49,
        cost=1.23,
        status="error",
        error_message="capacity",
        latency_ms=842,
        log_id="shape-log-id",
    )
    db_session.commit()

    response = client.get(USAGE_PATH, headers=master_key_header)
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0] == {
        "id": log.id,
        "user_id": "shape-user",
        "api_key_id": None,
        "timestamp": timestamp.isoformat(),
        "model": "gpt-4o",
        "provider": "openai",
        "endpoint": "/custom",
        "prompt_tokens": 42,
        "completion_tokens": 7,
        "total_tokens": 49,
        "cache_read_tokens": None,
        "cache_write_tokens": None,
        "cache_write_1h_tokens": None,
        "billing_meters": None,
        "pricing_breakdown": None,
        "cost": 1.23,
        "status": "error",
        "error_message": "capacity",
        "latency_ms": 842,
        "source": "gateway",
        "source_label": None,
        "counts_toward_budget": True,
    }


def test_list_usage_limit_max_enforced(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    response = client.get(
        USAGE_PATH,
        headers=master_key_header,
        params={"limit": 1001},
    )
    assert response.status_code == 422


def test_list_usage_skip_negative_rejected(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    response = client.get(
        USAGE_PATH,
        headers=master_key_header,
        params={"skip": -1},
    )
    assert response.status_code == 422


def test_list_usage_filter_by_epoch_seconds(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session: Session,
) -> None:
    older = datetime(2025, 8, 1, 0, 0, tzinfo=UTC)
    newer = datetime(2025, 8, 2, 0, 0, tzinfo=UTC)
    midpoint_epoch = int((older + timedelta(hours=12)).timestamp())
    _make_log(db_session, user_id="epoch-user", timestamp=older)
    _make_log(db_session, user_id="epoch-user", timestamp=newer)
    db_session.commit()

    response = client.get(
        USAGE_PATH,
        headers=master_key_header,
        params={"start_date": midpoint_epoch},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["timestamp"] == newer.isoformat()


def test_list_usage_filter_by_status(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session: Session,
) -> None:
    timestamp = datetime(2025, 9, 1, 12, 0, tzinfo=UTC)
    _make_log(db_session, user_id="status-user", timestamp=timestamp, status="success")
    _make_log(db_session, user_id="status-user", timestamp=timestamp, status="error", error_message="boom")
    db_session.commit()

    response = client.get(USAGE_PATH, headers=master_key_header, params={"status": "error"})
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["status"] == "error"


def test_list_usage_filter_by_model(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session: Session,
) -> None:
    timestamp = datetime(2025, 9, 2, 12, 0, tzinfo=UTC)
    _make_log(db_session, user_id="model-user", timestamp=timestamp, model="gpt-4o")
    _make_log(db_session, user_id="model-user", timestamp=timestamp, model="claude-sonnet-5")
    db_session.commit()

    response = client.get(USAGE_PATH, headers=master_key_header, params={"model": "claude-sonnet-5"})
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["model"] == "claude-sonnet-5"


def test_list_usage_filter_by_endpoint(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session: Session,
) -> None:
    timestamp = datetime(2025, 9, 3, 12, 0, tzinfo=UTC)
    _make_log(db_session, user_id="ep-user", timestamp=timestamp, endpoint="/v1/chat/completions")
    _make_log(db_session, user_id="ep-user", timestamp=timestamp, endpoint="/v1/embeddings")
    db_session.commit()

    response = client.get(USAGE_PATH, headers=master_key_header, params={"endpoint": "/v1/embeddings"})
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["endpoint"] == "/v1/embeddings"


def test_list_usage_filters_combine_with_and(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session: Session,
) -> None:
    timestamp = datetime(2025, 9, 4, 12, 0, tzinfo=UTC)
    # Only this row matches both status=error AND model=gpt-4o.
    _make_log(db_session, user_id="combo", timestamp=timestamp, model="gpt-4o", status="error", error_message="x")
    _make_log(db_session, user_id="combo", timestamp=timestamp, model="gpt-4o", status="success")
    _make_log(db_session, user_id="combo", timestamp=timestamp, model="claude-sonnet-5", status="error")
    db_session.commit()

    response = client.get(
        USAGE_PATH,
        headers=master_key_header,
        params={"status": "error", "model": "gpt-4o"},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["status"] == "error"
    assert data[0]["model"] == "gpt-4o"


def test_list_usage_still_returns_bare_list(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session: Session,
) -> None:
    """Contract guard: /v1/usage must stay a bare JSON array, not an envelope.

    External billing/analytics consumers depend on the top-level array; the
    paginated UI's total count is served by /v1/usage/count instead.
    """
    _make_log(db_session, user_id="contract", timestamp=datetime(2025, 9, 5, 12, 0, tzinfo=UTC))
    db_session.commit()

    response = client.get(USAGE_PATH, headers=master_key_header)
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_count_usage_requires_master_key(client: TestClient) -> None:
    response = client.get(f"{USAGE_PATH}/count")
    assert response.status_code == 401


def test_count_usage_matches_filtered_list(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session: Session,
) -> None:
    timestamp = datetime(2025, 9, 6, 12, 0, tzinfo=UTC)
    for _ in range(3):
        _make_log(db_session, user_id="count-user", timestamp=timestamp, status="error", error_message="e")
    _make_log(db_session, user_id="count-user", timestamp=timestamp, status="success")
    db_session.commit()

    count_resp = client.get(f"{USAGE_PATH}/count", headers=master_key_header, params={"status": "error"})
    assert count_resp.status_code == 200
    assert count_resp.json() == {"total": 3}

    # The count must match the number of rows the list returns for the same filter.
    list_resp = client.get(
        USAGE_PATH,
        headers=master_key_header,
        params={"status": "error", "limit": 1000},
    )
    assert len([row for row in list_resp.json() if row["user_id"] == "count-user"]) == 3


def test_count_usage_empty_is_zero(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    response = client.get(f"{USAGE_PATH}/count", headers=master_key_header, params={"user_id": "nobody"})
    assert response.status_code == 200
    assert response.json() == {"total": 0}
