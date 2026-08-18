"""Integration tests for the bulk usage endpoint."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from conftest import seed_workspace_id
from gateway.models.entities import APIKey, UsageLog, User

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
    status_code: int | None = None,
    latency_ms: int | None = None,
    log_id: str | None = None,
    policy_name: str | None = None,
    attempt_position: int | None = None,
    attempt_count: int | None = None,
    request_group_id: str | None = None,
) -> UsageLog:
    _ensure_user(db, user_id)
    log = UsageLog(
        id=log_id or str(uuid.uuid4()),
        workspace_id=seed_workspace_id(db),
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
        status_code=status_code,
        latency_ms=latency_ms,
        policy_name=policy_name,
        attempt_position=attempt_position,
        attempt_count=attempt_count,
        request_group_id=request_group_id,
    )
    db.add(log)
    return log


def test_list_usage_filters_by_request_group(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    """One group's rows are the whole plan behind one routed request."""
    ts = datetime(2026, 7, 1, 9, 0, tzinfo=UTC)
    _make_log(
        db_session,
        user_id="u",
        timestamp=ts,
        log_id="absorbed-1",
        status="absorbed",
        cost=None,
        policy_name="fast",
        attempt_position=1,
        attempt_count=2,
        request_group_id="grp-1",
    )
    _make_log(
        db_session,
        user_id="u",
        timestamp=ts + timedelta(milliseconds=300),
        log_id="served-1",
        policy_name="fast",
        attempt_position=2,
        attempt_count=2,
        request_group_id="grp-1",
    )
    _make_log(db_session, user_id="u", timestamp=ts, log_id="other-group", request_group_id="grp-2")
    db_session.commit()

    listed = client.get(USAGE_PATH, params={"request_group_id": "grp-1"}, headers=master_key_header)
    assert sorted(row["id"] for row in listed.json()) == ["absorbed-1", "served-1"]
    count = client.get("/v1/usage/count", params={"request_group_id": "grp-1"}, headers=master_key_header)
    assert count.json()["total"] == 2


def test_list_usage_filters_by_several_request_groups(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    """The dashboard resolves a whole page of rows in one batched lookup."""
    ts = datetime(2026, 7, 1, 9, 0, tzinfo=UTC)
    for group in ("grp-1", "grp-2", "grp-3"):
        _make_log(db_session, user_id="u", timestamp=ts, log_id=f"row-{group}", request_group_id=group)
    db_session.commit()

    listed = client.get(
        USAGE_PATH,
        params=[("request_group_id", "grp-1"), ("request_group_id", "grp-3")],
        headers=master_key_header,
    )
    assert sorted(row["id"] for row in listed.json()) == ["row-grp-1", "row-grp-3"]


def test_list_usage_request_group_batch_is_capped(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    """An unbounded IN list is rejected rather than executed."""
    response = client.get(
        USAGE_PATH,
        params=[("request_group_id", f"grp-{index}") for index in range(1001)],
        headers=master_key_header,
    )
    assert response.status_code == 422


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
        status_code=503,
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
        # Resolved from the joined user row, so a client can label a page of rows
        # without holding the users table. The helper sets alias == user_id.
        "user_alias": "shape-user",
        "api_key_id": None,
        # Null because this row has no key at all: the fall-back-to-the-id case.
        "api_key_name": None,
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
        "status_code": 503,
        "latency_ms": 842,
        "source": "gateway",
        "source_label": None,
        "counts_toward_budget": True,
        # Routing attribution: null for a request that named a plain model, which
        # is what this row is.
        "policy_name": None,
        "selection_reason": None,
        "attempt_position": None,
        "attempt_count": None,
        "request_group_id": None,
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


# ---------------------------------------------------------------------------
# Row labels: naming a page of rows must not cost the whole users/keys table
# ---------------------------------------------------------------------------


def test_list_usage_labels_rows_from_the_joined_entities(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session: Session,
) -> None:
    """A row carries its owner's alias and its key's name.

    Without these the dashboard had to page the entire users and api_keys tables
    on every visit to Usage and Activity just to turn ids into names, which grows
    with the deployment rather than with the page.
    """
    db_session.add(User(user_id="labeled-user", alias="Ada Lovelace", spend=0.0, blocked=False))
    db_session.flush()
    key = APIKey(
        workspace_id=seed_workspace_id(db_session),
        id=str(uuid.uuid4()),
        key_hash=f"hash-{uuid.uuid4()}",
        key_prefix="sk-test",
        key_name="CI pipeline",
        user_id="labeled-user",
    )
    db_session.add(key)
    db_session.flush()
    _make_log(
        db_session,
        user_id="labeled-user",
        timestamp=datetime(2025, 7, 2, 10, 0, tzinfo=UTC),
        api_key_id=key.id,
    )
    db_session.commit()

    response = client.get(USAGE_PATH, headers=master_key_header, params={"user_id": "labeled-user"})

    assert response.status_code == 200
    row = response.json()[0]
    assert row["user_alias"] == "Ada Lovelace"
    assert row["api_key_name"] == "CI pipeline"


def test_list_usage_keeps_a_row_whose_entities_are_gone(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session: Session,
) -> None:
    """The joins are outer: a row with no owner still comes back, unlabelled.

    Both foreign keys are ON DELETE SET NULL, so historical usage outlives the
    user and key it was billed to. An inner join would silently drop exactly the
    rows an operator most wants to see.
    """
    _make_log(
        db_session,
        user_id="soon-deleted",
        timestamp=datetime(2025, 7, 3, 10, 0, tzinfo=UTC),
        api_key_id=None,
    )
    db_session.commit()
    db_session.query(UsageLog).filter(UsageLog.user_id == "soon-deleted").update({"user_id": None})
    db_session.commit()

    response = client.get(USAGE_PATH, headers=master_key_header)

    assert response.status_code == 200
    orphans = [row for row in response.json() if row["user_id"] is None]
    assert len(orphans) == 1
    assert orphans[0]["user_alias"] is None
    assert orphans[0]["api_key_name"] is None


def test_summary_breakdowns_carry_labels_for_opaque_keys(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session: Session,
) -> None:
    """by_user and by_api_key name themselves, so a filter needs no table dump.

    This is what lets the dashboard's user and key pickers be built from the
    breakdown (top N by spend, in-window) the way the model picker already is.
    """
    db_session.add(User(user_id="summary-user", alias="Grace Hopper", spend=0.0, blocked=False))
    db_session.flush()
    key = APIKey(
        workspace_id=seed_workspace_id(db_session),
        id=str(uuid.uuid4()),
        key_hash=f"hash-{uuid.uuid4()}",
        key_prefix="sk-test",
        key_name="Nightly batch",
        user_id="summary-user",
    )
    db_session.add(key)
    db_session.flush()
    _make_log(
        db_session,
        user_id="summary-user",
        timestamp=datetime.now(UTC) - timedelta(hours=1),
        api_key_id=key.id,
    )
    db_session.commit()

    response = client.get(
        f"{USAGE_PATH}/summary",
        headers=master_key_header,
        params={"user_id": "summary-user", "dimensions": ["user", "api_key", "model"]},
    )

    assert response.status_code == 200
    body = response.json()
    user_row = next(r for r in body["by_user"] if r["key"] == "summary-user")
    assert user_row["label"] == "Grace Hopper"
    key_row = next(r for r in body["by_api_key"] if r["key"] == key.id)
    assert key_row["label"] == "Nightly batch"
    # A dimension whose key already reads as its own name carries no label.
    assert all(row["label"] is None for row in body["by_model"])
