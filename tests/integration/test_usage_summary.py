"""Integration tests for the aggregated usage summary + CSV export endpoints.

Runs against the PostgreSQL the suite is configured for (``TEST_DATABASE_URL``, or
a testcontainer). The bucketing expressions are dialect-aware (see
``_bucket_expr``), and these assertions pin the bucketing and reconciliation
contract they produce.
"""

from __future__ import annotations

import csv
import io
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from conftest import seed_workspace_id
from gateway.core.sql import MAX_FILTER_VALUES
from gateway.models.entities import APIKey, UsageLog, User

SUMMARY_PATH = "/v1/usage/summary"
CSV_PATH = "/v1/usage/summary.csv"


def _ensure_user(db: Session, user_id: str) -> None:
    if db.query(User).filter(User.user_id == user_id).first() is None:
        db.add(User(user_id=user_id, alias=user_id, spend=0.0, blocked=False))
        db.flush()


def _ensure_api_key(db: Session, key_id: str, user_id: str | None) -> None:
    # usage_logs.api_key_id is a real FK, so a row attributed to a key needs one.
    if db.query(APIKey).filter(APIKey.id == key_id).first() is None:
        db.add(
            APIKey(
                id=key_id,
                key_hash=f"hash-{key_id}",
                key_name=key_id,
                user_id=user_id,
                workspace_id=seed_workspace_id(db),
            )
        )
        db.flush()


def _make_log(
    db: Session,
    *,
    user_id: str | None = "u1",
    timestamp: datetime,
    api_key_id: str | None = None,
    model: str = "gpt-4",
    provider: str | None = "openai",
    endpoint: str = "/v1/chat/completions",
    source: str = "gateway",
    source_label: str | None = None,
    prompt_tokens: int | None = 10,
    completion_tokens: int | None = 5,
    total_tokens: int | None = 15,
    cache_read_tokens: int | None = None,
    cache_write_tokens: int | None = None,
    # Token meters are flat ints; the gateway-run tool meters are nested under the
    # reserved ``tools`` key, so the value type is not int-only.
    billing_meters: dict[str, Any] | None = None,
    cost: float | None = 0.01,
    status: str = "success",
    latency_ms: int | None = None,
) -> None:
    if user_id is not None:
        _ensure_user(db, user_id)
    if api_key_id is not None:
        _ensure_api_key(db, api_key_id, user_id)
    db.add(
        UsageLog(
            id=str(uuid.uuid4()),
            workspace_id=seed_workspace_id(db),
            user_id=user_id,
            api_key_id=api_key_id,
            timestamp=timestamp,
            model=model,
            provider=provider,
            endpoint=endpoint,
            source=source,
            source_label=source_label,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
            billing_meters=billing_meters,
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
        "cache_write_1h_tokens": 0,
        "request_count": 0,
        "error_count": 0,
        "avg_latency_ms": None,
        "unpriced_requests": 0,
        "billed_input_tokens": 0,
        "billed_output_tokens": 0,
    }
    assert body["by_model"] == []
    assert body["by_user"] == []
    assert body["by_api_key"] == []
    assert body["by_source"] == []
    assert body["by_source_label"] == []
    assert body["by_endpoint"] == []
    assert body["by_provider"] == []
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


def test_summary_unpriced_count_excludes_gateway_rejections(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    """``unpriced_requests`` is a pricing-gap signal, so only served rows count.

    The dashboard renders it as "N unpriced" next to the cost, meaning traffic is
    being metered without a price. A gateway-side rejection row also carries
    ``cost=NULL`` (nothing was spent), so counting error rows would make a budget
    or allow-list incident read as a pricing misconfiguration instead.
    """
    now = datetime.now(UTC) - timedelta(hours=1)
    _make_log(db_session, user_id="gap", timestamp=now, cost=None)
    _make_log(db_session, user_id="gap", timestamp=now, cost=None, status="error")
    _make_log(db_session, user_id="gap", timestamp=now, cost=0.02)
    db_session.commit()

    totals = client.get(SUMMARY_PATH, headers=master_key_header, params={"user_id": "gap"}).json()["totals"]
    assert totals["request_count"] == 3
    assert totals["error_count"] == 1
    assert totals["unpriced_requests"] == 1


def test_summary_breakdowns_reconcile_with_totals(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    now = datetime.now(UTC) - timedelta(hours=2)
    for model, cost in (("gpt-4", 0.10), ("gpt-4", 0.10), ("claude", 0.05)):
        _make_log(db_session, user_id="rec", timestamp=now, model=model, cost=cost, total_tokens=10)
    db_session.commit()

    body = client.get(SUMMARY_PATH, headers=master_key_header, params={"user_id": "rec"}).json()
    grand = body["totals"]["cost"]
    for dimension in (
        "by_model",
        "by_user",
        "by_api_key",
        "by_source",
        "by_source_label",
        "by_endpoint",
        "by_provider",
    ):
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
    assert other["is_other"] is True
    assert other["requests"] == 5
    # Named rows are never marked as the fold, even if their key were null.
    assert all(row["is_other"] is False for row in by_model[:-1])
    assert sum(r["requests"] for r in by_model) == body["totals"]["request_count"] == 105
    assert sum(r["cost"] for r in by_model) == pytest.approx(body["totals"]["cost"])


def test_summary_breaks_down_by_session_endpoint_and_provider(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    # Two imported agent sessions plus one gateway request: the three dimensions
    # stored on every row but previously absent from the aggregates.
    now = datetime.now(UTC) - timedelta(hours=1)
    _make_log(
        db_session,
        user_id="dim",
        timestamp=now,
        source="claude_code",
        source_label="session-a",
        endpoint="external",
        provider="anthropic",
        cost=0.50,
        prompt_tokens=30,
        completion_tokens=20,
        total_tokens=50,
    )
    _make_log(
        db_session,
        user_id="dim",
        timestamp=now,
        source="claude_code",
        source_label="session-b",
        endpoint="external",
        provider="anthropic",
        cost=0.20,
        total_tokens=20,
    )
    _make_log(
        db_session,
        user_id="dim",
        timestamp=now,
        endpoint="/v1/chat/completions",
        provider="openai",
        cost=0.10,
        total_tokens=10,
    )
    db_session.commit()

    body = client.get(SUMMARY_PATH, headers=master_key_header, params={"user_id": "dim"}).json()

    # Sessions, spend-ranked. Gateway rows carry no label, so they group under a
    # real null key that is *not* the synthesized fold.
    sessions = {row["key"]: row for row in body["by_source_label"]}
    assert [row["key"] for row in body["by_source_label"]] == ["session-a", "session-b", None]
    assert sessions["session-a"]["cost"] == pytest.approx(0.50)
    assert sessions["session-a"]["tokens"] == 50
    assert sessions[None]["requests"] == 1
    assert sessions[None]["is_other"] is False

    endpoints = {row["key"]: row["requests"] for row in body["by_endpoint"]}
    assert endpoints == {"external": 2, "/v1/chat/completions": 1}

    providers = {row["key"]: row["cost"] for row in body["by_provider"]}
    assert providers["anthropic"] == pytest.approx(0.70)
    assert providers["openai"] == pytest.approx(0.10)


def test_summary_session_breakdown_has_a_deeper_top_n(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    # Session cardinality dwarfs model/user cardinality, so that dimension keeps a
    # deeper cap (250) than the others (100): 150 sessions all stay named here,
    # while the same number of models would already be folding.
    now = datetime.now(UTC) - timedelta(hours=1)
    for idx in range(150):
        _make_log(
            db_session,
            user_id="deep",
            timestamp=now,
            model=f"m{idx:03d}",
            source="claude_code",
            source_label=f"s{idx:03d}",
            cost=0.01,
            total_tokens=1,
        )
    db_session.commit()

    body = client.get(SUMMARY_PATH, headers=master_key_header, params={"user_id": "deep"}).json()
    assert len(body["by_source_label"]) == 150
    assert all(row["is_other"] is False for row in body["by_source_label"])
    # The shallower cap still applies to the other dimensions.
    assert len(body["by_model"]) == 101
    assert body["by_model"][-1]["is_other"] is True


_BREAKDOWN_FIELDS = (
    "by_model",
    "by_user",
    "by_api_key",
    "by_source",
    "by_source_label",
    "by_endpoint",
    "by_provider",
)


def test_summary_computes_every_breakdown_by_default(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    # Omitting `dimensions` is the pre-selector behavior and must stay that way:
    # existing consumers never learned to ask for a breakdown.
    _make_log(db_session, user_id="dims", timestamp=datetime.now(UTC) - timedelta(hours=1), api_key_id=None)
    db_session.commit()

    body = client.get(SUMMARY_PATH, headers=master_key_header, params={"user_id": "dims"}).json()
    assert all(body[field] for field in _BREAKDOWN_FIELDS)


def test_summary_dimensions_selects_only_the_requested_breakdowns(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    # Each breakdown is its own GROUP BY pass, so a caller that reads one table
    # asks for one dimension. The excluded fields stay present but empty, so a
    # narrowed request cannot change the response schema.
    _make_log(db_session, user_id="pick", timestamp=datetime.now(UTC) - timedelta(hours=1))
    db_session.commit()

    resp = client.get(
        SUMMARY_PATH, headers=master_key_header, params={"user_id": "pick", "dimensions": ["model", "provider"]}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert [row["key"] for row in body["by_model"]] == ["gpt-4"]
    assert [row["key"] for row in body["by_provider"]] == ["openai"]
    assert body["by_user"] == []
    assert body["by_source_label"] == []
    assert body["by_endpoint"] == []
    # Totals and the series are never gated by the selector: they are what the
    # narrowed callers (tiles, timeline context) are asking for.
    assert body["totals"]["request_count"] == 1
    assert len(body["series"]) >= 1


def test_summary_dimensions_none_skips_every_breakdown(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    # "none" is how a totals/series-only caller expresses an empty selection, since
    # a repeated query param has no empty-list form.
    _make_log(db_session, user_id="bare", timestamp=datetime.now(UTC) - timedelta(hours=1), cost=0.25)
    db_session.commit()

    body = client.get(
        SUMMARY_PATH, headers=master_key_header, params={"user_id": "bare", "dimensions": "none"}
    ).json()
    assert all(body[field] == [] for field in _BREAKDOWN_FIELDS)
    assert body["totals"]["cost"] == pytest.approx(0.25)
    assert body["series"]


def test_summary_rejects_unknown_dimension(client: TestClient, master_key_header: dict[str, str]) -> None:
    resp = client.get(SUMMARY_PATH, headers=master_key_header, params={"dimensions": "cost_center"})
    assert resp.status_code == 422


def test_summary_filters_by_session_endpoint_and_provider(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    now = datetime.now(UTC) - timedelta(hours=1)
    _make_log(
        db_session,
        user_id="scope",
        timestamp=now,
        source="claude_code",
        source_label="wanted",
        endpoint="external",
        provider="anthropic",
        cost=0.50,
    )
    _make_log(
        db_session,
        user_id="scope",
        timestamp=now,
        source="claude_code",
        source_label="other",
        endpoint="external",
        provider="anthropic",
        cost=0.30,
    )
    _make_log(db_session, user_id="scope", timestamp=now, endpoint="/v1/chat/completions", provider="openai", cost=0.20)
    db_session.commit()

    scoped = client.get(
        SUMMARY_PATH, headers=master_key_header, params={"user_id": "scope", "source_label": "wanted"}
    ).json()
    assert scoped["totals"]["request_count"] == 1
    assert scoped["totals"]["cost"] == pytest.approx(0.50)

    by_endpoint = client.get(
        SUMMARY_PATH, headers=master_key_header, params={"user_id": "scope", "endpoint": "external"}
    ).json()
    assert by_endpoint["totals"]["request_count"] == 2

    by_provider = client.get(
        SUMMARY_PATH, headers=master_key_header, params={"user_id": "scope", "provider": "openai"}
    ).json()
    assert by_provider["totals"]["request_count"] == 1


def test_summary_filters_by_several_models_users_and_keys(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    """The three entity filters are repeatable: several values match any of them.

    The analytics page compares a handful of models / users / keys in one chart, so
    a single-value filter would force one request per value and make the tiles
    disagree with the comparison the operator asked for.
    """
    now = datetime.now(UTC) - timedelta(hours=1)
    _make_log(db_session, user_id="multi-a", timestamp=now, model="gpt-4", api_key_id="key-a", cost=0.10)
    _make_log(db_session, user_id="multi-b", timestamp=now, model="claude", api_key_id="key-b", cost=0.20)
    _make_log(db_session, user_id="multi-c", timestamp=now, model="gemini", api_key_id="key-c", cost=0.40)
    db_session.commit()

    everyone = ["multi-a", "multi-b", "multi-c"]

    two_users = client.get(SUMMARY_PATH, headers=master_key_header, params={"user_id": everyone[:2]}).json()
    assert two_users["totals"]["request_count"] == 2
    assert two_users["totals"]["cost"] == pytest.approx(0.30)
    assert {row["key"] for row in two_users["by_user"]} == {"multi-a", "multi-b"}

    two_models = client.get(
        SUMMARY_PATH, headers=master_key_header, params={"user_id": everyone, "model": ["gpt-4", "gemini"]}
    ).json()
    assert two_models["totals"]["request_count"] == 2
    assert two_models["totals"]["cost"] == pytest.approx(0.50)

    two_keys = client.get(
        SUMMARY_PATH, headers=master_key_header, params={"user_id": everyone, "api_key_id": ["key-a", "key-b"]}
    ).json()
    assert two_keys["totals"]["request_count"] == 2
    assert two_keys["totals"]["cost"] == pytest.approx(0.30)

    # A single value keeps working unchanged (the wire form every existing caller sends).
    one_user = client.get(SUMMARY_PATH, headers=master_key_header, params={"user_id": "multi-c"}).json()
    assert one_user["totals"]["request_count"] == 1
    assert one_user["totals"]["cost"] == pytest.approx(0.40)


def test_every_read_endpoint_caps_the_number_of_filter_values(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    """Every endpoint taking a repeatable filter bounds it, at the same value.

    The cap keeps a caller from posting an unbounded IN list, and it has to hold
    across the whole read surface: the bulk-mutation body carries the same ceiling,
    and that only means anything if the count an operator confirms is subject to it
    too. Sits far above any comparison a chart can render.
    """
    too_many = [f"m{index}" for index in range(MAX_FILTER_VALUES + 1)]
    at_cap = too_many[:MAX_FILTER_VALUES]
    for path, extra in (
        (SUMMARY_PATH, {}),
        (SERIES_PATH, {"group_by": "model"}),
        (CSV_PATH, {}),
        ("/v1/usage", {}),
        ("/v1/usage/count", {}),
    ):
        over = client.get(path, headers=master_key_header, params={**extra, "model": too_many})
        assert over.status_code == 422, path
        ok = client.get(path, headers=master_key_header, params={**extra, "model": at_cap})
        assert ok.status_code == 200, path

    # The bound counts values, not characters: a long provider-qualified model name
    # is a single value and must still filter.
    long_name = "openai:" + "a" * 80
    assert client.get(SUMMARY_PATH, headers=master_key_header, params={"model": long_name}).status_code == 200


def test_grouped_series_filters_by_several_models(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    # /series claims filter parity with /summary, so the stacked chart must scope to
    # the same value set the tiles beside it were computed over.
    ts = datetime(2025, 9, 1, 12, 0, tzinfo=UTC)
    _make_log(db_session, user_id="multiser", timestamp=ts, model="gpt-4", cost=0.10, total_tokens=15)
    _make_log(db_session, user_id="multiser", timestamp=ts, model="claude", cost=0.20, total_tokens=15)
    _make_log(db_session, user_id="multiser", timestamp=ts, model="gemini", cost=0.40, total_tokens=15)
    db_session.commit()

    body = client.get(
        SERIES_PATH,
        headers=master_key_header,
        params={
            "group_by": "model",
            "user_id": "multiser",
            "model": ["gpt-4", "claude"],
            "start_date": "2025-09-01T00:00:00Z",
            "end_date": "2025-09-02T00:00:00Z",
        },
    ).json()

    assert {g["key"] for g in body["groups"]} == {"gpt-4", "claude"}
    assert sum(p["cost"] for p in body["points"]) == pytest.approx(0.30)


def test_csv_export_filters_by_several_users(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    # The export takes the same window and filters as /summary, so a multi-value
    # comparison can be downloaded rather than re-filtered by hand.
    now = datetime.now(UTC) - timedelta(hours=1)
    _make_log(db_session, user_id="csv-a", timestamp=now, model="gpt-4", cost=0.10)
    _make_log(db_session, user_id="csv-b", timestamp=now, model="claude", cost=0.20)
    _make_log(db_session, user_id="csv-c", timestamp=now, model="gemini", cost=0.40)
    db_session.commit()

    resp = client.get(CSV_PATH, headers=master_key_header, params={"user_id": ["csv-a", "csv-b"]})
    assert resp.status_code == 200
    rows = list(csv.DictReader(io.StringIO(resp.text)))
    users = {row["key"] for row in rows if row["dimension"] == "user"}
    assert users == {"csv-a", "csv-b"}


def test_usage_list_and_count_filter_by_session_and_provider(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    # The drill-down from a session breakdown row: the raw log must be scopable to
    # that one session, and /count must agree so the paginator's total matches.
    now = datetime.now(UTC) - timedelta(hours=1)
    _make_log(
        db_session,
        user_id="drill",
        timestamp=now,
        source="claude_code",
        source_label="sess-1",
        provider="anthropic",
    )
    _make_log(
        db_session,
        user_id="drill",
        timestamp=now,
        source="claude_code",
        source_label="sess-2",
        provider="anthropic",
    )
    _make_log(db_session, user_id="drill", timestamp=now, provider="openai")
    db_session.commit()

    rows = client.get(
        "/v1/usage", headers=master_key_header, params={"user_id": "drill", "source_label": "sess-1"}
    ).json()
    assert [row["source_label"] for row in rows] == ["sess-1"]
    count = client.get(
        "/v1/usage/count", headers=master_key_header, params={"user_id": "drill", "source_label": "sess-1"}
    ).json()
    assert count["total"] == 1

    openai_rows = client.get(
        "/v1/usage", headers=master_key_header, params={"user_id": "drill", "provider": "openai"}
    ).json()
    assert len(openai_rows) == 1
    assert openai_rows[0]["provider"] == "openai"


def test_usage_list_and_count_filter_by_several_values(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    # The request log takes the same repeatable filters as the analytics endpoints, so
    # a drill-down carrying a multi-value comparison lands on exactly those rows, and
    # /count agrees with the list so the paginator total matches what is shown.
    now = datetime.now(UTC) - timedelta(hours=1)
    _make_log(db_session, user_id="listmulti", timestamp=now, model="gpt-4")
    _make_log(db_session, user_id="listmulti", timestamp=now, model="claude")
    _make_log(db_session, user_id="listmulti", timestamp=now, model="gemini")
    db_session.commit()

    params = {"user_id": "listmulti", "model": ["gpt-4", "claude"]}
    rows = client.get("/v1/usage", headers=master_key_header, params=params).json()
    assert sorted(row["model"] for row in rows) == ["claude", "gpt-4"]

    count = client.get("/v1/usage/count", headers=master_key_header, params=params).json()
    assert count["total"] == len(rows) == 2

    # Two users, one model: the other dimension still narrows as usual.
    _make_log(db_session, user_id="listmulti2", timestamp=now, model="gpt-4")
    db_session.commit()
    both = client.get(
        "/v1/usage/count",
        headers=master_key_header,
        params={"user_id": ["listmulti", "listmulti2"], "model": "gpt-4"},
    ).json()
    assert both["total"] == 2


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
    # Window [Dec 31, Jan 3) at day granularity => 3 dense buckets; both same-UTC-day
    # rows collapse into Jan 1, and the empty leading day is zero-filled.
    assert series == {
        "2024-12-31T00:00:00Z": 0,
        "2025-01-01T00:00:00Z": 2,
        "2025-01-02T00:00:00Z": 1,
    }


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
    # Window [Mar 1 00:00, Mar 2 00:00) at hour granularity => 24 dense buckets, the
    # two active hours populated and the rest zero-filled.
    assert len(series) == 24
    assert series["2025-03-01T10:00:00Z"] == 2
    assert series["2025-03-01T11:00:00Z"] == 1
    assert series["2025-03-01T09:00:00Z"] == 0


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


def test_summary_accepts_naive_start_date(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    # An offset-less ISO datetime (advertised as valid) parses naive; the endpoint
    # must treat it as UTC, not 500 on comparing it to the aware "now".
    now = datetime.now(UTC) - timedelta(hours=1)
    _make_log(db_session, user_id="naive", timestamp=now, cost=0.05, total_tokens=1)
    db_session.commit()

    resp = client.get(
        SUMMARY_PATH,
        headers=master_key_header,
        params={"user_id": "naive", "start_date": "2020-01-01T00:00:00"},
    )
    assert resp.status_code == 200
    assert resp.json()["totals"]["request_count"] == 1


def test_summary_series_zero_fills_empty_buckets(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    # Usage only on day 1 and day 4 of a 4-day window; the gap days must appear as
    # zero buckets so the chart's x-axis stays linear in time.
    _make_log(db_session, user_id="dense", timestamp=datetime(2025, 6, 1, 12, tzinfo=UTC), cost=0.01, total_tokens=1)
    _make_log(db_session, user_id="dense", timestamp=datetime(2025, 6, 4, 12, tzinfo=UTC), cost=0.02, total_tokens=1)
    db_session.commit()

    body = client.get(
        SUMMARY_PATH,
        headers=master_key_header,
        params={
            "user_id": "dense",
            "bucket": "day",
            "start_date": "2025-06-01T00:00:00Z",
            "end_date": "2025-06-05T00:00:00Z",
        },
    ).json()
    series = body["series"]
    assert [p["bucket_start"] for p in series] == [
        "2025-06-01T00:00:00Z",
        "2025-06-02T00:00:00Z",
        "2025-06-03T00:00:00Z",
        "2025-06-04T00:00:00Z",
    ]
    assert [p["requests"] for p in series] == [1, 0, 0, 1]


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


def test_csv_export_includes_session_endpoint_and_provider(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    now = datetime.now(UTC) - timedelta(hours=1)
    _make_log(
        db_session,
        user_id="csvdim",
        timestamp=now,
        source="claude_code",
        source_label="session-a",
        endpoint="external",
        provider="anthropic",
        cost=0.10,
        total_tokens=10,
    )
    db_session.commit()

    resp = client.get(CSV_PATH, headers=master_key_header, params={"user_id": "csvdim"})
    rows = list(csv.reader(io.StringIO(resp.text)))[1:]
    by_dimension = {r[0]: r[1] for r in rows}
    assert by_dimension["session"] == "session-a"
    assert by_dimension["endpoint"] == "external"
    assert by_dimension["provider"] == "anthropic"
    assert by_dimension["source"] == "claude_code"


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


# ---------------------------------------------------------------------------
# Billed token composition (series + totals + breakdowns) and /series grouping.
# ---------------------------------------------------------------------------

SERIES_PATH = "/v1/usage/series"


def test_series_composition_prefers_meters_and_falls_back(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    ts = datetime(2025, 3, 1, 10, 30, tzinfo=UTC)
    # Additive-convention row, normalized by its meters: prompt excludes the
    # cache buckets, so billed input (900) far exceeds prompt_tokens (100).
    _make_log(
        db_session,
        user_id="comp",
        timestamp=ts,
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
        cache_read_tokens=700,
        cache_write_tokens=100,
        billing_meters={
            "total_input_tokens": 900,
            "fresh_input_tokens": 100,
            "cache_read_tokens": 700,
            "cache_write_tokens": 100,
            "cache_write_1h_tokens": 0,
            "completion_tokens": 50,
        },
    )
    # Meterless row: falls back to the raw columns under the subset convention
    # (billed input = prompt_tokens as stored).
    _make_log(
        db_session,
        user_id="comp",
        timestamp=ts,
        prompt_tokens=200,
        completion_tokens=30,
        total_tokens=230,
        cache_read_tokens=120,
        status="error",
    )
    db_session.commit()

    body = client.get(
        SUMMARY_PATH,
        headers=master_key_header,
        params={"user_id": "comp", "start_date": "2025-03-01T00:00:00Z", "end_date": "2025-03-02T00:00:00Z"},
    ).json()

    assert body["totals"]["billed_input_tokens"] == 900 + 200
    [point] = [p for p in body["series"] if p["requests"]]
    assert point["input_tokens"] == 1100
    assert point["cache_read_tokens"] == 700 + 120
    assert point["cache_write_tokens"] == 100
    assert point["output_tokens"] == 80
    assert point["errors"] == 1
    # The raw provider total is untouched by the composition fields.
    assert point["tokens"] == 150 + 230

    # Breakdowns report the same billed quantity (input + output).
    [model_row] = body["by_model"]
    assert model_row["tokens"] == 1100 + 80


def test_breakdown_fold_reconciles_billed_output_with_divergent_meters(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    """The fold row must use billed output, not the raw completion column.

    A row whose completion meter diverges from its column (here: a NULL column
    with a meter present) would otherwise drift the residual, and the top
    groups' billed output could push the fold's token count negative.
    """
    ts = datetime(2025, 7, 1, 12, 0, tzinfo=UTC)
    # Two models so a fold exists when only the top one is kept... the summary
    # cap is 100, so instead diverge the meters and check exact reconciliation.
    _make_log(
        db_session,
        user_id="divg",
        timestamp=ts,
        model="metered",
        prompt_tokens=10,
        completion_tokens=None,
        billing_meters={
            "total_input_tokens": 10,
            "fresh_input_tokens": 10,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "cache_write_1h_tokens": 0,
            "completion_tokens": 40,
        },
    )
    _make_log(db_session, user_id="divg", timestamp=ts, model="plain")
    db_session.commit()

    body = client.get(
        SUMMARY_PATH,
        headers=master_key_header,
        params={"user_id": "divg", "start_date": "2025-07-01T00:00:00Z", "end_date": "2025-07-02T00:00:00Z"},
    ).json()
    totals = body["totals"]
    # Billed output prefers the meter (40) over the NULL column, plus the plain
    # row's raw 5.
    assert totals["billed_output_tokens"] == 45
    assert totals["completion_tokens"] == 5
    # Per-group billed sums reconcile with the billed totals exactly.
    by_model = {r["key"]: r for r in body["by_model"]}
    assert by_model["metered"]["tokens"] == 10 + 40
    assert by_model["plain"]["tokens"] == 10 + 5
    assert sum(r["tokens"] for r in body["by_model"]) == totals["billed_input_tokens"] + totals["billed_output_tokens"]


def test_grouped_series_requires_master_key_and_group_by(client: TestClient) -> None:
    assert client.get(SERIES_PATH).status_code == 401


def test_grouped_series_rejects_unknown_group_by(client: TestClient, master_key_header: dict[str, str]) -> None:
    assert client.get(SERIES_PATH, headers=master_key_header, params={"group_by": "provider"}).status_code == 422
    # group_by is required.
    assert client.get(SERIES_PATH, headers=master_key_header).status_code == 422


def test_grouped_series_by_model_reconciles(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    day1 = datetime(2025, 4, 1, 9, 0, tzinfo=UTC)
    day2 = datetime(2025, 4, 2, 9, 0, tzinfo=UTC)
    _make_log(db_session, user_id="grp", timestamp=day1, model="gpt-4", cost=0.30, total_tokens=15)
    _make_log(db_session, user_id="grp", timestamp=day1, model="claude", cost=0.10, total_tokens=15)
    _make_log(db_session, user_id="grp", timestamp=day2, model="gpt-4", cost=0.20, total_tokens=15)
    db_session.commit()

    body = client.get(
        SERIES_PATH,
        headers=master_key_header,
        params={
            "group_by": "model",
            "user_id": "grp",
            "start_date": "2025-04-01T00:00:00Z",
            "end_date": "2025-04-03T00:00:00Z",
        },
    ).json()

    assert body["group_by"] == "model"
    assert [g["key"] for g in body["groups"]] == ["gpt-4", "claude"]

    # Sparse points, canonical UTC buckets, sorted by bucket.
    buckets = [p["bucket_start"] for p in body["points"]]
    assert buckets == sorted(buckets)
    assert set(buckets) == {"2025-04-01T00:00:00Z", "2025-04-02T00:00:00Z"}

    day1_points = {p["key"]: p for p in body["points"] if p["bucket_start"] == "2025-04-01T00:00:00Z"}
    assert day1_points["gpt-4"]["cost"] == pytest.approx(0.30)
    assert day1_points["claude"]["cost"] == pytest.approx(0.10)
    # Billed tokens per (bucket, group): meterless rows fall back to prompt+output.
    assert day1_points["gpt-4"]["tokens"] == 15

    # The stack reconciles with the window totals.
    assert sum(p["cost"] for p in body["points"]) == pytest.approx(0.60)
    assert sum(p["requests"] for p in body["points"]) == 3


def test_grouped_series_folds_past_top_n_and_null_groups(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    ts = datetime(2025, 5, 1, 12, 0, tzinfo=UTC)
    # 10 models, descending spend, one request each: 8 named + 2 folded.
    for idx in range(10):
        _make_log(db_session, user_id="topn", timestamp=ts, model=f"m{idx}", cost=1.0 - idx * 0.05, total_tokens=15)
    db_session.commit()

    body = client.get(
        SERIES_PATH,
        headers=master_key_header,
        params={
            "group_by": "model",
            "user_id": "topn",
            "start_date": "2025-05-01T00:00:00Z",
            "end_date": "2025-05-02T00:00:00Z",
        },
    ).json()

    named = [g for g in body["groups"] if not g["is_other"]]
    fold = [g for g in body["groups"] if g["is_other"]]
    assert len(named) == 8
    assert len(fold) == 1
    assert fold[0]["requests"] == 2

    other_points = [p for p in body["points"] if p["is_other"]]
    assert sum(p["requests"] for p in other_points) == 2
    assert sum(p["cost"] for p in other_points) == pytest.approx(0.60 + 0.55)
    # Every named point carries its own key; the fold rows have key None.
    assert all(p["key"] is None for p in other_points)
    assert sum(p["requests"] for p in body["points"]) == 10


def test_grouped_series_null_group_key_survives_when_in_top_n(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    ts = datetime(2025, 6, 1, 12, 0, tzinfo=UTC)
    # api_key_id is NULL on both rows: the NULL group ranks in the top N and must
    # come back as a real key=None group (not the fold).
    _make_log(db_session, user_id="nullg", timestamp=ts, cost=0.10, total_tokens=15)
    _make_log(db_session, user_id="nullg", timestamp=ts, cost=0.20, total_tokens=15)
    db_session.commit()

    body = client.get(
        SERIES_PATH,
        headers=master_key_header,
        params={
            "group_by": "api_key_id",
            "user_id": "nullg",
            "start_date": "2025-06-01T00:00:00Z",
            "end_date": "2025-06-02T00:00:00Z",
        },
    ).json()

    assert [g["key"] for g in body["groups"]] == [None]
    assert body["groups"][0]["is_other"] is False
    [point] = body["points"]
    assert point["key"] is None
    assert point["is_other"] is False
    assert point["requests"] == 2
    assert point["cost"] == pytest.approx(0.30)


def test_grouped_series_null_top_group_and_fold_stay_separate(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    """The three-arm fold CASE: a NULL group ranked in the top N must not merge
    with the past-top-N fold, even though both are keyed NULL in SQL."""
    ts = datetime(2025, 7, 1, 12, 0, tzinfo=UTC)
    # The NULL user is the top spender; nine named users follow, so with top
    # N = 8 the window keeps NULL + seven named groups and folds the last two.
    _make_log(db_session, user_id=None, timestamp=ts, model="cf-model", cost=5.0, total_tokens=15)
    for idx in range(9):
        _make_log(
            db_session, user_id=f"cf-u{idx}", timestamp=ts, model="cf-model", cost=1.0 - idx * 0.05, total_tokens=15
        )
    db_session.commit()

    body = client.get(
        SERIES_PATH,
        headers=master_key_header,
        params={
            "group_by": "user_id",
            "model": "cf-model",
            "start_date": "2025-07-01T00:00:00Z",
            "end_date": "2025-07-02T00:00:00Z",
        },
    ).json()

    groups = body["groups"]
    assert len([g for g in groups if g["key"] is None and not g["is_other"]]) == 1
    assert len([g for g in groups if g["key"] is not None]) == 7
    [fold] = [g for g in groups if g["is_other"]]
    assert fold["requests"] == 2

    points = body["points"]
    [null_point] = [p for p in points if p["key"] is None and not p["is_other"]]
    assert null_point["requests"] == 1
    assert null_point["cost"] == pytest.approx(5.0)
    [fold_point] = [p for p in points if p["is_other"]]
    assert fold_point["requests"] == 2
    assert fold_point["cost"] == pytest.approx(0.65 + 0.60)
    assert sum(p["requests"] for p in points) == 10


def test_grouped_series_honors_provider_and_session_filters(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    """/series accepts the same filters as /summary (it claims parity).

    A filter it silently dropped would make the dashboard's stacked chart
    aggregate over a wider set than the tiles beside it.
    """
    ts = datetime(2025, 8, 1, 12, 0, tzinfo=UTC)
    _make_log(db_session, user_id="par", timestamp=ts, model="m1", provider="openai", source_label="s1", cost=0.10)
    _make_log(db_session, user_id="par", timestamp=ts, model="m1", provider="anthropic", source_label="s2", cost=0.20)
    db_session.commit()

    window = {"start_date": "2025-08-01T00:00:00Z", "end_date": "2025-08-02T00:00:00Z", "user_id": "par"}
    for params, expected_cost in (
        ({"provider": "openai"}, 0.10),
        ({"source_label": "s2"}, 0.20),
    ):
        body = client.get(
            SERIES_PATH, headers=master_key_header, params={"group_by": "model", **window, **params}
        ).json()
        assert sum(p["cost"] for p in body["points"]) == pytest.approx(expected_cost), params


def test_grouped_series_caps_hourly_buckets(client: TestClient, master_key_header: dict[str, str]) -> None:
    """An hourly grid over a too-wide window is rejected, not ballooned.

    /summary densifies then caps at _MAX_SERIES_POINTS; the grouped series is
    sparse per (bucket, group), so it bounds the bucket grid up front.
    """
    resp = client.get(
        SERIES_PATH,
        headers=master_key_header,
        params={
            "group_by": "model",
            "bucket": "hour",
            "start_date": "2025-01-01T00:00:00Z",
            "end_date": "2025-06-01T00:00:00Z",
        },
    )
    assert resp.status_code == 422
    assert "bucket=day" in resp.json()["detail"]
    # The same window is fine at daily granularity.
    ok = client.get(
        SERIES_PATH,
        headers=master_key_header,
        params={
            "group_by": "model",
            "bucket": "day",
            "start_date": "2025-01-01T00:00:00Z",
            "end_date": "2025-06-01T00:00:00Z",
        },
    )
    assert ok.status_code == 200


def test_tool_breakdown_counts_the_row_that_served_not_the_absorbed_attempt(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    """A failed-over request contributes one request and its full tool work.

    The production shape: a routing policy writes one ``absorbed`` row per attempt it
    recovered from plus the row that served, and only the serving row carries the tool
    ledger (``log_absorbed_attempt`` passes no tally, because it settles no
    reservation and a charge there would never reach ``users.spend``).

    So ``calls`` comes from the serving row while ``requests`` stays 1.

    Note what this does and does not prove. It does not exercise
    ``_request_count_expr`` in the aggregate: the query already restricts to rows
    carrying tool meters, and an absorbed row never has them, so it is excluded
    before the count is taken. That expression is defensive there, and the invariant
    it defends (absorbed rows carry no tool ledger) is asserted directly in
    ``test_routing_policies.test_tools_run_by_the_candidate_that_serves_...``. What
    this does prove is that a two-row request reports its tool work once, with the
    counts and cost coming off the row that served.
    """
    now = datetime.now(UTC)
    _make_log(
        db_session,
        user_id="tools",
        timestamp=now,
        status="absorbed",
        cost=None,
        billing_meters=None,
    )
    _make_log(
        db_session,
        user_id="tools",
        timestamp=now,
        status="success",
        cost=0.05,
        billing_meters={
            "total_input_tokens": 10,
            "completion_tokens": 5,
            "tools": {"web_search": {"billed": 3, "errors": 1, "unit_rate": 0.01}},
        },
    )
    db_session.commit()

    response = client.get(
        "/v1/usage/summary",
        params={"dimensions": "tool", "user_id": "tools"},
        headers=master_key_header,
    )
    assert response.status_code == 200, response.text
    by_tool = response.json()["by_tool"]
    assert len(by_tool) == 1
    row = by_tool[0]
    assert row["tool"] == "web_search"
    assert row["calls"] == 3
    assert row["errors"] == 1
    assert row["requests"] == 1
    assert row["cost"] == pytest.approx(0.03)


def test_tool_breakdown_is_empty_when_only_absorbed_rows_match(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    """Filtering to absorbed rows reports nothing rather than a contradictory row.

    Absorbed rows carry no tool meters, so the aggregate matches none of them and the
    entry is dropped instead of rendering as "N calls, 0 requests".
    """
    _make_log(
        db_session,
        user_id="absorbed-only",
        timestamp=datetime.now(UTC),
        status="absorbed",
        cost=None,
        billing_meters=None,
    )
    db_session.commit()

    response = client.get(
        "/v1/usage/summary",
        params={"dimensions": "tool", "status": "absorbed", "user_id": "absorbed-only"},
        headers=master_key_header,
    )
    assert response.status_code == 200, response.text
    assert response.json()["by_tool"] == []
