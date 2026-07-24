"""Integration tests for the operator usage mutations: bulk delete and set-price.

Both endpoints must touch only imported rows (``counts_toward_budget = False``);
enforced gateway rows and ``users.spend`` are never affected.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from gateway.models.entities import UsageLog, User

DELETE_PATH = "/v1/usage"
SET_PRICE_PATH = "/v1/usage/set-price"
COUNT_PATH = "/v1/usage/count"

_TS = datetime(2026, 7, 1, 12, 0, tzinfo=UTC)


def _ensure_user(db: Session, user_id: str) -> None:
    if db.query(User).filter(User.user_id == user_id).first() is None:
        db.add(User(user_id=user_id, alias=user_id, spend=0.0, blocked=False))
        db.flush()


def _make_log(
    db: Session,
    *,
    log_id: str,
    counts_toward_budget: bool,
    user_id: str = "u",
    source: str = "claude_code",
    model: str = "openai/gpt-4",
    prompt_tokens: int | None = 1000,
    completion_tokens: int | None = 500,
    cache_read_tokens: int | None = None,
    cache_write_tokens: int | None = None,
    cost: float | None = None,
    status: str = "success",
    timestamp: datetime = _TS,
) -> UsageLog:
    _ensure_user(db, user_id)
    log = UsageLog(
        id=log_id,
        user_id=user_id,
        timestamp=timestamp,
        model=model,
        provider="openai",
        endpoint="external" if not counts_toward_budget else "/v1/chat/completions",
        source=source,
        # Imported rows carry a unique source_event_id (idempotency); gateway rows leave it NULL.
        source_event_id=log_id if not counts_toward_budget else None,
        counts_toward_budget=counts_toward_budget,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=(prompt_tokens or 0) + (completion_tokens or 0),
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
        cost=cost,
        status=status,
    )
    db.add(log)
    return log


def _get(db: Session, log_id: str) -> UsageLog | None:
    return db.query(UsageLog).filter(UsageLog.id == log_id).first()


# --------------------------------------------------------------------------- delete


def test_delete_by_ids_removes_only_imported(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    _make_log(db_session, log_id="imp-1", counts_toward_budget=False)
    _make_log(db_session, log_id="imp-2", counts_toward_budget=False)
    _make_log(db_session, log_id="gw-1", counts_toward_budget=True)
    db_session.commit()

    # The request names an imported row *and* an enforced gateway row; only the
    # imported one may be removed.
    resp = client.request(
        "DELETE", DELETE_PATH, json={"ids": ["imp-1", "gw-1"]}, headers=master_key_header
    )
    assert resp.status_code == 200
    assert resp.json() == {"deleted": 1}

    db_session.expire_all()
    assert _get(db_session, "imp-1") is None
    assert _get(db_session, "imp-2") is not None
    assert _get(db_session, "gw-1") is not None


def test_delete_by_filter_source(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    _make_log(db_session, log_id="cc-1", counts_toward_budget=False, source="claude_code")
    _make_log(db_session, log_id="cc-2", counts_toward_budget=False, source="claude_code")
    _make_log(db_session, log_id="other-1", counts_toward_budget=False, source="codex")
    db_session.commit()

    resp = client.request(
        "DELETE", DELETE_PATH, json={"by_filter": True, "source": "claude_code"}, headers=master_key_header
    )
    assert resp.status_code == 200
    assert resp.json() == {"deleted": 2}

    db_session.expire_all()
    assert _get(db_session, "cc-1") is None
    assert _get(db_session, "cc-2") is None
    assert _get(db_session, "other-1") is not None


def test_delete_by_filter_unpriced_only(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    _make_log(db_session, log_id="unpriced", counts_toward_budget=False, cost=None)
    _make_log(db_session, log_id="priced", counts_toward_budget=False, cost=0.02)
    db_session.commit()

    resp = client.request(
        "DELETE", DELETE_PATH, json={"by_filter": True, "priced": False}, headers=master_key_header
    )
    assert resp.status_code == 200
    assert resp.json() == {"deleted": 1}

    db_session.expire_all()
    assert _get(db_session, "unpriced") is None
    assert _get(db_session, "priced") is not None


def test_delete_by_filter_never_touches_gateway_rows(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    # An unfiltered by_filter delete targets every imported row, but must still
    # leave enforced gateway rows in place.
    _make_log(db_session, log_id="imp-1", counts_toward_budget=False)
    _make_log(db_session, log_id="gw-1", counts_toward_budget=True)
    db_session.commit()

    resp = client.request("DELETE", DELETE_PATH, json={"by_filter": True}, headers=master_key_header)
    assert resp.status_code == 200
    assert resp.json() == {"deleted": 1}

    db_session.expire_all()
    assert _get(db_session, "imp-1") is None
    assert _get(db_session, "gw-1") is not None


def test_delete_by_filter_api_key(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    _make_log(db_session, log_id="k1-a", counts_toward_budget=False)
    _make_log(db_session, log_id="k1-b", counts_toward_budget=False)
    db_session.query(UsageLog).filter(UsageLog.id.in_(["k1-a", "k1-b"])).update(
        {UsageLog.api_key_id: None}, synchronize_session=False
    )
    db_session.commit()

    # No key set on these rows; filtering to a specific key matches nothing.
    resp = client.request(
        "DELETE", DELETE_PATH, json={"by_filter": True, "api_key_id": "does-not-exist"}, headers=master_key_header
    )
    assert resp.status_code == 200
    assert resp.json() == {"deleted": 0}


def test_delete_requires_master_key(client: TestClient) -> None:
    resp = client.request("DELETE", DELETE_PATH, json={"ids": ["x"]})
    assert resp.status_code == 401


def test_delete_empty_selection_is_rejected(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    # Neither ids nor by_filter: a 422, so an empty body can never match every row.
    resp = client.request("DELETE", DELETE_PATH, json={}, headers=master_key_header)
    assert resp.status_code == 422


def test_delete_both_modes_is_rejected(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    resp = client.request(
        "DELETE", DELETE_PATH, json={"ids": ["x"], "by_filter": True}, headers=master_key_header
    )
    assert resp.status_code == 422


# ------------------------------------------------------------------------- set-price


def test_set_price_by_ids_recomputes_cost(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    _make_log(
        db_session,
        log_id="imp-1",
        counts_toward_budget=False,
        prompt_tokens=1000,
        completion_tokens=500,
        cost=None,
    )
    db_session.commit()

    resp = client.post(
        SET_PRICE_PATH,
        json={
            "ids": ["imp-1"],
            "input_price_per_million": 3.0,
            "output_price_per_million": 15.0,
        },
        headers=master_key_header,
    )
    assert resp.status_code == 200
    assert resp.json() == {"matched": 1, "updated": 1, "unchanged": 0}

    db_session.expire_all()
    row = _get(db_session, "imp-1")
    assert row is not None
    # 1000 input @ $3/1M + 500 output @ $15/1M = 0.003 + 0.0075
    assert row.cost == pytest.approx(0.0105)
    assert row.billing_meters is not None
    assert row.billing_meters["fresh_input_tokens"] == 1000
    assert row.billing_meters["completion_tokens"] == 500
    assert row.pricing_breakdown is not None
    meters = {line["meter"]: line for line in row.pricing_breakdown}
    assert meters["input"]["units"] == 1000
    assert meters["output"]["units"] == 500


def test_set_price_with_cache_rates(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    _make_log(
        db_session,
        log_id="imp-cache",
        counts_toward_budget=False,
        prompt_tokens=1000,
        completion_tokens=200,
        cache_read_tokens=500,
        cost=None,
    )
    db_session.commit()

    resp = client.post(
        SET_PRICE_PATH,
        json={
            "ids": ["imp-cache"],
            "input_price_per_million": 3.0,
            "output_price_per_million": 15.0,
            "cache_read_price_per_million": 0.3,
        },
        headers=master_key_header,
    )
    assert resp.status_code == 200
    db_session.expire_all()
    row = _get(db_session, "imp-cache")
    assert row is not None
    # Additive shape: total input = 1000 + 500 cache read. Fresh input = 1000.
    # 1000 * 3/1M + 200 * 15/1M + 500 * 0.3/1M = 0.003 + 0.003 + 0.00015
    assert row.cost == pytest.approx(0.00615)


def test_set_price_only_touches_imported(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    _make_log(db_session, log_id="gw-1", counts_toward_budget=True, cost=0.99)
    db_session.commit()

    resp = client.post(
        SET_PRICE_PATH,
        json={
            "ids": ["gw-1"],
            "input_price_per_million": 3.0,
            "output_price_per_million": 15.0,
        },
        headers=master_key_header,
    )
    assert resp.status_code == 200
    assert resp.json() == {"matched": 0, "updated": 0, "unchanged": 0}

    db_session.expire_all()
    row = _get(db_session, "gw-1")
    assert row is not None
    assert row.cost == 0.99  # untouched


def test_set_price_reports_unchanged_on_second_run(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    _make_log(db_session, log_id="imp-1", counts_toward_budget=False, cost=None)
    db_session.commit()
    body = {
        "ids": ["imp-1"],
        "input_price_per_million": 3.0,
        "output_price_per_million": 15.0,
    }
    first = client.post(SET_PRICE_PATH, json=body, headers=master_key_header)
    assert first.json()["updated"] == 1

    second = client.post(SET_PRICE_PATH, json=body, headers=master_key_header)
    assert second.json() == {"matched": 1, "updated": 0, "unchanged": 1}


def test_set_price_by_filter_model(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    _make_log(db_session, log_id="a", counts_toward_budget=False, model="openai/gpt-4", cost=None)
    _make_log(db_session, log_id="b", counts_toward_budget=False, model="anthropic/claude", cost=None)
    db_session.commit()

    resp = client.post(
        SET_PRICE_PATH,
        json={
            "by_filter": True,
            "model": "openai/gpt-4",
            "input_price_per_million": 1.0,
            "output_price_per_million": 1.0,
        },
        headers=master_key_header,
    )
    assert resp.status_code == 200
    assert resp.json()["matched"] == 1

    db_session.expire_all()
    assert _get(db_session, "a").cost is not None  # type: ignore[union-attr]
    assert _get(db_session, "b").cost is None  # type: ignore[union-attr]


def test_set_price_requires_master_key(client: TestClient) -> None:
    resp = client.post(
        SET_PRICE_PATH,
        json={"ids": ["x"], "input_price_per_million": 1.0, "output_price_per_million": 1.0},
    )
    assert resp.status_code == 401


def test_set_price_rejects_negative_rate(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    resp = client.post(
        SET_PRICE_PATH,
        json={"ids": ["x"], "input_price_per_million": -1.0, "output_price_per_million": 1.0},
        headers=master_key_header,
    )
    assert resp.status_code == 422


# --------------------------------------------------------- count for select-all affordance


def test_count_scopes_to_imported_rows(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    _make_log(db_session, log_id="imp-1", counts_toward_budget=False)
    _make_log(db_session, log_id="imp-2", counts_toward_budget=False)
    _make_log(db_session, log_id="gw-1", counts_toward_budget=True)
    db_session.commit()

    resp = client.get(COUNT_PATH, params={"counts_toward_budget": "false"}, headers=master_key_header)
    assert resp.status_code == 200
    assert resp.json()["total"] == 2
