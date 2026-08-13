"""Integration tests for the agent_telemetry purge endpoint and user-deletion cleanup."""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from gateway.models.entities import AgentTelemetry, APIKey, User

DELETE_PATH = "/v1/agent-telemetry"

_TS = datetime(2026, 7, 1, 12, 0, tzinfo=UTC)


def _ensure_user(db: Session, user_id: str) -> None:
    if db.query(User).filter(User.user_id == user_id).first() is None:
        db.add(User(user_id=user_id, alias=user_id, spend=0.0, blocked=False))
        db.flush()


def _ensure_api_key(db: Session, api_key_id: str, user_id: str) -> None:
    if db.query(APIKey).filter(APIKey.id == api_key_id).first() is None:
        db.add(APIKey(id=api_key_id, key_hash=f"hash-{api_key_id}", user_id=user_id))
        db.flush()


def _make_row(
    db: Session,
    *,
    row_id: str,
    user_id: str = "u",
    api_key_id: str | None = None,
    name: str = "tool_result",
    source: str = "claude_code",
    timestamp: datetime = _TS,
) -> AgentTelemetry:
    _ensure_user(db, user_id)
    if api_key_id is not None:
        _ensure_api_key(db, api_key_id, user_id)
    row = AgentTelemetry(
        id=row_id,
        api_key_id=api_key_id,
        user_id=user_id,
        timestamp=timestamp,
        name=name,
        tool_name="Bash",
        success=True,
        source=source,
        dedup_key=row_id,
    )
    db.add(row)
    return row


def _make_metric_row(
    db: Session,
    *,
    row_id: str,
    user_id: str = "u",
    name: str = "claude_code.commit.count",
    timestamp: datetime = _TS,
) -> AgentTelemetry:
    _ensure_user(db, user_id)
    row = AgentTelemetry(
        id=row_id,
        user_id=user_id,
        timestamp=timestamp,
        name=name,
        source="claude_code",
        dedup_key=row_id,
        kind="metric",
        value=3.0,
        temporality="cumulative",
        series_start=timestamp,
        series_key=f"series-{row_id}",
    )
    db.add(row)
    return row


def _get(db: Session, row_id: str) -> AgentTelemetry | None:
    return db.query(AgentTelemetry).filter(AgentTelemetry.id == row_id).first()


def test_purge_by_ids(client: TestClient, master_key_header: dict[str, str], db_session: Session) -> None:
    _make_row(db_session, row_id="t-1", user_id="alice")
    _make_row(db_session, row_id="t-2", user_id="alice")
    db_session.commit()

    resp = client.request("DELETE", DELETE_PATH, json={"ids": ["t-1"]}, headers=master_key_header)
    assert resp.status_code == 200
    assert resp.json() == {"deleted": 1}

    db_session.expire_all()
    assert _get(db_session, "t-1") is None
    assert _get(db_session, "t-2") is not None


def test_purge_by_filter_user_id_name_and_date_range(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    _make_row(db_session, row_id="alice-1", user_id="alice", name="tool_result")
    _make_row(db_session, row_id="alice-2", user_id="alice", name="user_prompt")
    _make_row(db_session, row_id="bob-1", user_id="bob", name="tool_result")
    db_session.commit()

    resp = client.request(
        "DELETE",
        DELETE_PATH,
        json={
            "by_filter": True,
            "user_id": "alice",
            "name": "tool_result",
            "start_date": "2026-01-01T00:00:00Z",
            "end_date": "2027-01-01T00:00:00Z",
        },
        headers=master_key_header,
    )
    assert resp.status_code == 200
    assert resp.json() == {"deleted": 1}

    db_session.expire_all()
    assert _get(db_session, "alice-1") is None
    assert _get(db_session, "alice-2") is not None
    assert _get(db_session, "bob-1") is not None


def test_purge_by_filter_api_key_id(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    _make_row(db_session, row_id="key-a-row", user_id="alice", api_key_id="key-a")
    _make_row(db_session, row_id="key-b-row", user_id="alice", api_key_id="key-b")
    db_session.commit()

    resp = client.request(
        "DELETE", DELETE_PATH, json={"by_filter": True, "api_key_id": "key-a"}, headers=master_key_header
    )
    assert resp.status_code == 200
    assert resp.json() == {"deleted": 1}

    db_session.expire_all()
    assert _get(db_session, "key-a-row") is None
    assert _get(db_session, "key-b-row") is not None


def test_purge_matching_zero_rows_is_not_an_error(client: TestClient, master_key_header: dict[str, str]) -> None:
    resp = client.request(
        "DELETE", DELETE_PATH, json={"by_filter": True, "user_id": "nobody-matches"}, headers=master_key_header
    )
    assert resp.status_code == 200
    assert resp.json() == {"deleted": 0}


def test_purge_requires_exactly_one_of_ids_or_by_filter(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    neither = client.request("DELETE", DELETE_PATH, json={}, headers=master_key_header)
    assert neither.status_code == 422

    both = client.request(
        "DELETE", DELETE_PATH, json={"ids": ["x"], "by_filter": True}, headers=master_key_header
    )
    assert both.status_code == 422


def test_purge_requires_master_key(client: TestClient) -> None:
    resp = client.request("DELETE", DELETE_PATH, json={"ids": ["x"]})
    assert resp.status_code == 401


def test_delete_user_removes_only_that_users_agent_telemetry_rows(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    _make_row(db_session, row_id="alice-row", user_id="alice")
    _make_row(db_session, row_id="bob-row", user_id="bob")
    db_session.commit()

    resp = client.delete("/v1/users/alice", headers=master_key_header)
    assert resp.status_code == 204

    db_session.expire_all()
    assert _get(db_session, "alice-row") is None
    assert _get(db_session, "bob-row") is not None


def test_purge_by_filter_removes_metric_rows_alongside_behavioral_ones(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    """One deletion mechanism covers both row kinds: the filter has no `kind`."""
    _make_row(db_session, row_id="alice-behavioral", user_id="alice")
    _make_metric_row(db_session, row_id="alice-metric", user_id="alice")
    _make_row(db_session, row_id="bob-behavioral", user_id="bob")
    db_session.commit()

    resp = client.request(
        "DELETE", DELETE_PATH, json={"by_filter": True, "user_id": "alice"}, headers=master_key_header
    )
    assert resp.status_code == 200
    assert resp.json() == {"deleted": 2}

    db_session.expire_all()
    assert _get(db_session, "alice-behavioral") is None
    assert _get(db_session, "alice-metric") is None
    assert _get(db_session, "bob-behavioral") is not None

    count = client.get("/v1/agent-telemetry/count", params={"user_id": "alice"}, headers=master_key_header)
    assert count.status_code == 200
    assert count.json()["total"] == 0


def test_delete_user_removes_that_users_metric_rows_too(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    _make_metric_row(db_session, row_id="alice-metric", user_id="alice")
    _make_row(db_session, row_id="alice-behavioral", user_id="alice")
    _make_metric_row(db_session, row_id="bob-metric", user_id="bob")
    db_session.commit()

    resp = client.delete("/v1/users/alice", headers=master_key_header)
    assert resp.status_code == 204

    db_session.expire_all()
    assert _get(db_session, "alice-metric") is None
    assert _get(db_session, "alice-behavioral") is None
    assert _get(db_session, "bob-metric") is not None
