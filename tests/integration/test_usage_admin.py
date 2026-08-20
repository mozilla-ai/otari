"""Integration tests for the operator usage mutations: bulk delete and set-price.

Both endpoints must touch only imported rows (``counts_toward_budget = False``);
enforced gateway rows and ``users.spend`` are never affected.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from conftest import seed_workspace_id
from gateway.core.sql import MAX_FILTER_VALUES
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
    source_label: str | None = None,
    provider: str = "openai",
    model: str = "openai/gpt-4",
    prompt_tokens: int | None = 1000,
    completion_tokens: int | None = 500,
    cache_read_tokens: int | None = None,
    cache_write_tokens: int | None = None,
    billing_meters: dict[str, int] | None = None,
    cost: float | None = None,
    status: str = "success",
    timestamp: datetime = _TS,
) -> UsageLog:
    _ensure_user(db, user_id)
    log = UsageLog(
        id=log_id,
        workspace_id=seed_workspace_id(db),
        user_id=user_id,
        timestamp=timestamp,
        model=model,
        provider=provider,
        endpoint="external" if not counts_toward_budget else "/v1/chat/completions",
        source=source,
        source_label=source_label,
        # Imported rows carry a unique source_event_id (idempotency); gateway rows leave it NULL.
        source_event_id=log_id if not counts_toward_budget else None,
        counts_toward_budget=counts_toward_budget,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=(prompt_tokens or 0) + (completion_tokens or 0),
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
        billing_meters=billing_meters,
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


def test_delete_by_filter_scopes_to_one_session(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    # The Usage-page session breakdown drills into Activity scoped to one
    # source_label. "Select all matching" is counted under that scope, so the
    # delete must honor it: without it, deleting the 1 row the operator was shown
    # would take every other session's rows in the window with it.
    _make_log(db_session, log_id="sess-a", counts_toward_budget=False, source_label="task-42")
    _make_log(db_session, log_id="sess-b", counts_toward_budget=False, source_label="task-43")
    _make_log(db_session, log_id="sess-none", counts_toward_budget=False)
    db_session.commit()

    resp = client.request(
        "DELETE", DELETE_PATH, json={"by_filter": True, "source_label": "task-42"}, headers=master_key_header
    )
    assert resp.status_code == 200
    assert resp.json() == {"deleted": 1}

    db_session.expire_all()
    assert _get(db_session, "sess-a") is None
    assert _get(db_session, "sess-b") is not None
    assert _get(db_session, "sess-none") is not None


def test_delete_by_filter_scopes_to_one_provider(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    _make_log(db_session, log_id="p-anthropic", counts_toward_budget=False, provider="anthropic")
    _make_log(db_session, log_id="p-openai", counts_toward_budget=False, provider="openai")
    db_session.commit()

    resp = client.request(
        "DELETE", DELETE_PATH, json={"by_filter": True, "provider": "anthropic"}, headers=master_key_header
    )
    assert resp.status_code == 200
    assert resp.json() == {"deleted": 1}

    db_session.expire_all()
    assert _get(db_session, "p-anthropic") is None
    assert _get(db_session, "p-openai") is not None


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


def test_ops_skip_budget_exempt_gateway_rows(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    # A gateway row from a budget-exempt key is counts_toward_budget=False but is NOT
    # imported (source="gateway"); delete and set-price must never touch it.
    _make_log(db_session, log_id="imp", counts_toward_budget=False, source="claude_code", cost=None)
    _make_log(db_session, log_id="gw-exempt", counts_toward_budget=False, source="gateway", cost=0.5)
    db_session.commit()

    # Delete-all-imported by filter removes only the imported row.
    resp = client.request("DELETE", DELETE_PATH, json={"by_filter": True}, headers=master_key_header)
    assert resp.json() == {"deleted": 1}
    db_session.expire_all()
    assert _get(db_session, "imp") is None
    assert _get(db_session, "gw-exempt") is not None

    # Set-price by filter must not reprice the surviving gateway-exempt row.
    resp = client.post(
        SET_PRICE_PATH,
        json={"by_filter": True, "input_price_per_million": 3.0, "output_price_per_million": 15.0},
        headers=master_key_header,
    )
    assert resp.json() == {"matched": 0, "updated": 0, "unchanged": 0}
    db_session.expire_all()
    assert _get(db_session, "gw-exempt").cost == Decimal("0.5")  # type: ignore[union-attr]


def test_delete_by_filter_scopes_to_the_named_models_only(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    """A multi-value filter deletes exactly its values, never the rest.

    This is the load-bearing case for repeatable filters on a destructive op: the
    operator confirms a count taken over the same filter set, so a body that widened
    a dimension (or ignored the extra values) would delete rows the table never
    showed. Three models, two named: the third must survive untouched.
    """
    _make_log(db_session, log_id="m-gpt", counts_toward_budget=False, model="openai/gpt-4")
    _make_log(db_session, log_id="m-claude", counts_toward_budget=False, model="anthropic/claude")
    _make_log(db_session, log_id="m-gemini", counts_toward_budget=False, model="google/gemini")
    db_session.commit()

    # The count the operator would have been shown agrees with what the delete removes.
    count = client.get(
        COUNT_PATH,
        headers=master_key_header,
        params={"model": ["openai/gpt-4", "anthropic/claude"], "counts_toward_budget": False},
    ).json()
    assert count["total"] == 2

    resp = client.request(
        "DELETE",
        DELETE_PATH,
        json={"by_filter": True, "model": ["openai/gpt-4", "anthropic/claude"]},
        headers=master_key_header,
    )
    assert resp.status_code == 200
    assert resp.json() == {"deleted": 2}

    db_session.expire_all()
    assert _get(db_session, "m-gpt") is None
    assert _get(db_session, "m-claude") is None
    assert _get(db_session, "m-gemini") is not None


def test_delete_by_filter_scopes_to_the_named_users_only(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    _make_log(db_session, log_id="u-alice", counts_toward_budget=False, user_id="alice")
    _make_log(db_session, log_id="u-bob", counts_toward_budget=False, user_id="bob")
    _make_log(db_session, log_id="u-carol", counts_toward_budget=False, user_id="carol")
    db_session.commit()

    resp = client.request(
        "DELETE", DELETE_PATH, json={"by_filter": True, "user_id": ["alice", "bob"]}, headers=master_key_header
    )
    assert resp.json() == {"deleted": 2}

    db_session.expire_all()
    assert _get(db_session, "u-carol") is not None


def test_set_price_by_filter_prices_the_named_models_only(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    _make_log(db_session, log_id="p-gpt", counts_toward_budget=False, model="openai/gpt-4", cost=None)
    _make_log(db_session, log_id="p-claude", counts_toward_budget=False, model="anthropic/claude", cost=None)
    _make_log(db_session, log_id="p-gemini", counts_toward_budget=False, model="google/gemini", cost=None)
    db_session.commit()

    resp = client.post(
        SET_PRICE_PATH,
        json={
            "by_filter": True,
            "model": ["openai/gpt-4", "anthropic/claude"],
            "input_price_per_million": 1.0,
            "output_price_per_million": 1.0,
        },
        headers=master_key_header,
    )
    assert resp.status_code == 200
    assert resp.json()["matched"] == 2

    db_session.expire_all()
    assert _get(db_session, "p-gpt").cost is not None  # type: ignore[union-attr]
    assert _get(db_session, "p-claude").cost is not None  # type: ignore[union-attr]
    # The model left out of the filter keeps its unpriced state.
    assert _get(db_session, "p-gemini").cost is None  # type: ignore[union-attr]


def test_by_filter_rejects_more_values_than_the_read_endpoints_accept(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    """The destructive body stops where /v1/usage/count stops.

    The count an operator confirms comes from the read endpoints, which 422 past
    MAX_FILTER_VALUES. A body that accepted more would delete over a filter set no
    count could ever have been shown for, on an unbounded IN list.
    """
    too_many = [f"m{index}" for index in range(MAX_FILTER_VALUES + 1)]
    assert client.get(COUNT_PATH, headers=master_key_header, params={"model": too_many}).status_code == 422

    resp = client.request(
        "DELETE", DELETE_PATH, json={"by_filter": True, "model": too_many}, headers=master_key_header
    )
    assert resp.status_code == 422

    at_cap = too_many[:MAX_FILTER_VALUES]
    assert (
        client.request(
            "DELETE", DELETE_PATH, json={"by_filter": True, "model": at_cap}, headers=master_key_header
        ).status_code
        == 200
    )
    # A single long value is not a list of 51: the bound is on the value count, so a
    # provider-qualified model name well past 50 characters still filters.
    long_name = f"openai/{'x' * 80}"
    assert (
        client.request(
            "DELETE", DELETE_PATH, json={"by_filter": True, "model": long_name}, headers=master_key_header
        ).status_code
        == 200
    )


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
    assert row.cost == Decimal("0.0105")
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
    assert row.cost == Decimal("0.00615")


def test_set_price_recovers_the_inclusive_shape_from_the_stored_meters(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    """An OpenAI-shaped import must not have its cached tokens added to the prompt again.

    The convention is not a column, so repricing recovers it from the meters
    settlement wrote: ``total_input_tokens`` equal to ``prompt_tokens`` means the
    cached slice was already inside the prompt. Assuming the additive shape here
    would bill 1500 input tokens for a row that reported 1000.
    """
    _make_log(
        db_session,
        log_id="imp-inclusive",
        counts_toward_budget=False,
        source="codex",
        prompt_tokens=1000,
        completion_tokens=200,
        cache_read_tokens=500,
        billing_meters={"total_input_tokens": 1000, "fresh_input_tokens": 500},
        cost=None,
    )
    db_session.commit()

    resp = client.post(
        SET_PRICE_PATH,
        json={
            "ids": ["imp-inclusive"],
            "input_price_per_million": 3.0,
            "output_price_per_million": 15.0,
            "cache_read_price_per_million": 0.3,
        },
        headers=master_key_header,
    )
    assert resp.status_code == 200
    db_session.expire_all()
    row = _get(db_session, "imp-inclusive")
    assert row is not None
    # Inclusive shape: total input stays 1000, of which 500 are the cached slice.
    # 500 * 3/1M + 200 * 15/1M + 500 * 0.3/1M = 0.0015 + 0.003 + 0.00015
    assert row.cost == Decimal("0.00465")
    assert row.billing_meters is not None
    assert row.billing_meters["total_input_tokens"] == 1000
    assert row.billing_meters["fresh_input_tokens"] == 500


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
    assert row.cost == Decimal("0.99")  # untouched


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
