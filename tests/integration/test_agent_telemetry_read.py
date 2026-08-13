"""Integration tests for the agent_telemetry read endpoints.

Seeds `agent_telemetry` (metric and behavioral rows) and `usage_logs` directly,
so the assertions are about the aggregation, not about the OTLP receiver that
normally fills those tables.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from gateway.api.deps import reset_config
from gateway.core.config import GatewayConfig
from gateway.core.database import reset_db
from gateway.main import create_app
from gateway.models.entities import AgentTelemetry, APIKey, UsageLog, User

SUMMARY_PATH = "/v1/agent-telemetry/summary"
COUNT_PATH = "/v1/agent-telemetry/count"
SERIES_PATH = "/v1/agent-telemetry/series"

_NOW = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
_T0 = _NOW - timedelta(days=2)
_WINDOW = {"start_date": (_NOW - timedelta(days=5)).isoformat(), "end_date": (_NOW + timedelta(days=1)).isoformat()}


def _ensure_user(db: Session, user_id: str) -> None:
    if db.query(User).filter(User.user_id == user_id).first() is None:
        db.add(User(user_id=user_id, alias=user_id, spend=0.0, blocked=False))
        db.flush()


def _ensure_api_key(db: Session, api_key_id: str, user_id: str) -> None:
    if db.query(APIKey).filter(APIKey.id == api_key_id).first() is None:
        db.add(APIKey(id=api_key_id, key_hash=f"hash-{api_key_id}", user_id=user_id))
        db.flush()


def _metric_row(
    db: Session,
    *,
    row_id: str,
    name: str,
    value: float,
    user_id: str = "alice",
    api_key_id: str | None = None,
    temporality: str = "cumulative",
    series_key: str | None = None,
    series_start: datetime = _T0,
    timestamp: datetime = _T0,
    session_label: str = "s-1",
) -> None:
    _ensure_user(db, user_id)
    if api_key_id is not None:
        _ensure_api_key(db, api_key_id, user_id)
    db.add(
        AgentTelemetry(
            id=row_id,
            api_key_id=api_key_id,
            user_id=user_id,
            timestamp=timestamp,
            name=name,
            source="claude_code",
            session_label=session_label,
            dedup_key=row_id,
            kind="metric",
            value=value,
            temporality=temporality,
            series_start=series_start,
            series_key=series_key or f"series-{name}",
        )
    )


def _behavioral_row(
    db: Session,
    *,
    row_id: str,
    name: str,
    user_id: str = "alice",
    api_key_id: str | None = None,
    tool_name: str | None = None,
    decision: str | None = None,
    session_label: str = "s-1",
    timestamp: datetime = _T0,
) -> None:
    _ensure_user(db, user_id)
    if api_key_id is not None:
        _ensure_api_key(db, api_key_id, user_id)
    db.add(
        AgentTelemetry(
            id=row_id,
            api_key_id=api_key_id,
            user_id=user_id,
            timestamp=timestamp,
            name=name,
            tool_name=tool_name,
            decision=decision,
            source="claude_code",
            session_label=session_label,
            dedup_key=row_id,
        )
    )


def _usage_row(
    db: Session,
    *,
    row_id: str,
    cost: float,
    user_id: str = "alice",
    timestamp: datetime = _T0,
    session_label: str = "s-1",
) -> None:
    _ensure_user(db, user_id)
    db.add(
        UsageLog(
            id=row_id,
            user_id=user_id,
            timestamp=timestamp,
            model="claude-opus-4-8",
            provider="anthropic",
            endpoint="/v1/chat/completions",
            source="claude_code",
            source_event_id=row_id,
            source_label=session_label,
            counts_toward_budget=False,
            cost=cost,
            status="success",
        )
    )


def _seed(db: Session) -> None:
    # Outcome metrics: cumulative counters, one of them dimensioned into two series.
    _metric_row(db, row_id="m-commit-1", name="claude_code.commit.count", value=0.0)
    _metric_row(
        db,
        row_id="m-commit-2",
        name="claude_code.commit.count",
        value=4.0,
        timestamp=_T0 + timedelta(hours=1),
    )
    _metric_row(db, row_id="m-pr-1", name="claude_code.pull_request.count", value=2.0, temporality="delta")
    _metric_row(
        db,
        row_id="m-lines-added",
        name="claude_code.lines_of_code.count",
        value=120.0,
        temporality="delta",
        series_key="series-lines-added",
    )
    _metric_row(
        db,
        row_id="m-lines-removed",
        name="claude_code.lines_of_code.count",
        value=30.0,
        temporality="delta",
        series_key="series-lines-removed",
    )
    _metric_row(
        db, row_id="m-active", name="claude_code.active_time.total", value=7200.0, temporality="delta"
    )
    # Behavioral events (already stored since PR #548), read here for the first time.
    _behavioral_row(db, row_id="b-tool-1", name="tool_result", tool_name="Bash")
    _behavioral_row(db, row_id="b-tool-2", name="tool_result", tool_name="Bash")
    _behavioral_row(db, row_id="b-tool-3", name="tool_result", tool_name="Edit")
    _behavioral_row(db, row_id="b-dec-1", name="tool_decision", tool_name="Edit", decision="accept")
    _behavioral_row(db, row_id="b-dec-2", name="tool_decision", tool_name="Edit", decision="accept")
    _behavioral_row(db, row_id="b-dec-3", name="tool_decision", tool_name="Edit", decision="reject")
    _behavioral_row(db, row_id="b-turn-1", name="user_prompt")
    _behavioral_row(db, row_id="b-turn-2", name="user_prompt")
    _behavioral_row(db, row_id="b-turn-3", name="user_prompt", session_label="s-2")
    _behavioral_row(db, row_id="b-err-1", name="api_error")
    # Another user's rows, so every filter assertion has something to exclude.
    _behavioral_row(db, row_id="b-bob-1", name="user_prompt", user_id="bob", session_label="s-bob")
    _usage_row(db, row_id="u-1", cost=6.0)
    _usage_row(db, row_id="u-2", cost=2.0)
    _usage_row(db, row_id="u-bob", cost=99.0, user_id="bob")
    db.commit()


def test_summary_derives_cost_per_outcome(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    _seed(db_session)
    response = client.get(SUMMARY_PATH, params={**_WINDOW, "user_id": "alice"}, headers=master_key_header)

    assert response.status_code == 200, response.text
    body = response.json()
    outcomes = body["outcomes"]
    # Cumulative commits count their growth (0 -> 4), not the running total twice.
    assert outcomes["commits"] == 4.0
    assert outcomes["pull_requests"] == 2.0
    # Both dimensioned lines series sum into one "lines changed" measure.
    assert outcomes["lines_of_code"] == 150.0
    assert outcomes["active_time"] == 7200.0

    behavior = body["behavior"]
    assert behavior["tool_calls"] == 3
    assert {row["tool"]: row["calls"] for row in behavior["by_tool"]} == {"Bash": 2, "Edit": 1}
    assert behavior["tool_accepts"] == 2
    assert behavior["tool_rejects"] == 1
    assert behavior["turns"] == 3
    assert behavior["sessions"] == 2
    assert behavior["api_errors"] == 1

    measures = body["measures"]
    assert measures["cost_per_commit"] == pytest.approx(2.0)
    assert measures["cost_per_pull_request"] == pytest.approx(4.0)
    assert measures["cost_per_line"] == pytest.approx(8.0 / 150.0)
    assert measures["spend_per_active_hour"] == pytest.approx(4.0)
    # Acceptance comes from tool_decision only: code_edit_tool.decision is never stored.
    assert measures["edit_acceptance_rate"] == pytest.approx(2 / 3)
    assert measures["tool_acceptance_rate"] == pytest.approx(2 / 3)
    assert measures["turns_per_session"] == pytest.approx(1.5)
    assert measures["error_rate"] == pytest.approx(0.5)
    assert body["series"]


def test_summary_measures_are_null_without_a_denominator(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    _usage_row(db_session, row_id="u-lonely", cost=3.0)
    db_session.commit()
    response = client.get(SUMMARY_PATH, params={**_WINDOW, "user_id": "alice"}, headers=master_key_header)

    assert response.status_code == 200, response.text
    measures = response.json()["measures"]
    assert measures["cost_per_commit"] is None
    assert measures["edit_acceptance_rate"] is None
    assert measures["turns_per_session"] is None
    assert response.json()["outcomes"]["commits"] == 0.0


def test_summary_counter_reset_never_yields_a_negative_increment(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    """A new ``series_start`` is a fresh generation, diffed from its own base.

    The first generation began before the window, so its 40 is a level carried in
    and only the 6 that follow are counted; the second began at the reset, inside
    the window, so it is diffed from its own zero.
    """
    reset_at = _T0 + timedelta(hours=2)
    carried_in = _NOW - timedelta(days=10)
    _metric_row(db_session, row_id="r-1", name="claude_code.commit.count", value=40.0, series_start=carried_in)
    _metric_row(
        db_session,
        row_id="r-2",
        name="claude_code.commit.count",
        value=46.0,
        series_start=carried_in,
        timestamp=_T0 + timedelta(hours=1),
    )
    _metric_row(
        db_session,
        row_id="r-3",
        name="claude_code.commit.count",
        value=0.0,
        series_start=reset_at,
        timestamp=reset_at,
    )
    _metric_row(
        db_session,
        row_id="r-4",
        name="claude_code.commit.count",
        value=3.0,
        series_start=reset_at,
        timestamp=reset_at + timedelta(hours=1),
    )
    db_session.commit()

    response = client.get(SUMMARY_PATH, params={**_WINDOW, "user_id": "alice"}, headers=master_key_header)
    assert response.status_code == 200, response.text
    assert response.json()["outcomes"]["commits"] == 9.0


def test_summary_counts_the_first_reading_of_a_generation_that_began_in_window(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    """A counter is zero at its series start, so its first reading is growth in full.

    An OTel counter exports no data point until its first measurement, so a real
    session's first point already carries work: three commits arrive as readings of
    1 then 3, with no zero ahead of them. Diffing from the second reading on would
    report 2, and a session whose only export carries one commit would report 0.
    """
    _metric_row(
        db_session,
        row_id="m-fresh-1",
        name="claude_code.commit.count",
        value=1.0,
        series_key="series-fresh",
        series_start=_T0,
        timestamp=_T0 + timedelta(minutes=1),
    )
    _metric_row(
        db_session,
        row_id="m-fresh-2",
        name="claude_code.commit.count",
        value=3.0,
        series_key="series-fresh",
        series_start=_T0,
        timestamp=_T0 + timedelta(minutes=2),
    )
    db_session.commit()

    response = client.get(SUMMARY_PATH, params={**_WINDOW, "user_id": "alice"}, headers=master_key_header)

    assert response.status_code == 200, response.text
    assert response.json()["outcomes"]["commits"] == 3.0


def test_summary_reports_a_single_export_session(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    """One export is a whole short session's report, not an uncountable level."""
    _metric_row(
        db_session,
        row_id="m-only",
        name="claude_code.commit.count",
        value=1.0,
        series_key="series-only",
        series_start=_T0,
        timestamp=_T0 + timedelta(minutes=1),
    )
    db_session.commit()

    response = client.get(SUMMARY_PATH, params={**_WINDOW, "user_id": "alice"}, headers=master_key_header)

    assert response.status_code == 200, response.text
    assert response.json()["outcomes"]["commits"] == 1.0


def test_summary_scopes_cost_per_outcome_to_one_session(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    """US1's independent test: one session's outcomes against that session's spend.

    The filter has to reach both sides of the join, which name the session
    differently: `agent_telemetry.session_label` and `usage_logs.source_label`.
    """
    _seed(db_session)
    # A second session for the same user, with its own commits and its own spend.
    _metric_row(
        db_session,
        row_id="m-s2-commit",
        name="claude_code.commit.count",
        value=6.0,
        temporality="delta",
        series_key="series-commit-s2",
        session_label="s-2",
    )
    _usage_row(db_session, row_id="u-s2", cost=12.0, session_label="s-2")
    db_session.commit()

    scoped = client.get(SUMMARY_PATH, params={**_WINDOW, "session_label": "s-2"}, headers=master_key_header)

    assert scoped.status_code == 200, scoped.text
    body = scoped.json()
    assert body["outcomes"]["commits"] == 6.0
    assert body["usage"]["cost"] == pytest.approx(12.0)
    assert body["measures"]["cost_per_commit"] == pytest.approx(2.0)
    # s-1's outcomes, behavior, and spend stay out of the scoped answer; the one
    # behavioral row seeded into s-2 is a turn, so it is the only one counted.
    assert body["outcomes"]["pull_requests"] == 0.0
    assert body["behavior"]["tool_calls"] == 0
    assert body["behavior"]["turns"] == 1
    assert body["behavior"]["sessions"] == 1

    # Unfiltered, the same window still reports both sessions together.
    everything = client.get(SUMMARY_PATH, params={**_WINDOW, "user_id": "alice"}, headers=master_key_header)
    assert everything.status_code == 200, everything.text
    assert everything.json()["outcomes"]["commits"] == 10.0
    assert everything.json()["usage"]["cost"] == pytest.approx(20.0)


def test_count_returns_rows_matching_the_filter(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    _seed(db_session)
    everything = client.get(COUNT_PATH, params=_WINDOW, headers=master_key_header)
    assert everything.status_code == 200, everything.text
    assert everything.json()["total"] == 17

    alice = client.get(COUNT_PATH, params={**_WINDOW, "user_id": "alice"}, headers=master_key_header)
    assert alice.json()["total"] == 16

    by_name = client.get(
        COUNT_PATH, params={**_WINDOW, "user_id": "alice", "name": "user_prompt"}, headers=master_key_header
    )
    assert by_name.json()["total"] == 3


def test_series_returns_one_series_per_group(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    _seed(db_session)
    response = client.get(SERIES_PATH, params={**_WINDOW, "group_by": "user_id"}, headers=master_key_header)

    assert response.status_code == 200, response.text
    body = response.json()
    assert {group["key"] for group in body["groups"]} == {"alice", "bob"}
    assert {point["key"] for point in body["points"]} == {"alice", "bob"}
    assert sum(point["rows"] for point in body["points"] if point["key"] == "alice") == 16


def test_series_rejects_a_window_with_too_many_buckets(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    response = client.get(
        SERIES_PATH,
        params={
            "group_by": "user_id",
            "bucket": "hour",
            "start_date": (_NOW - timedelta(days=300)).isoformat(),
            "end_date": _NOW.isoformat(),
        },
        headers=master_key_header,
    )
    assert response.status_code == 422


@pytest.mark.parametrize("path", [SUMMARY_PATH, COUNT_PATH, SERIES_PATH])
def test_read_endpoints_require_the_master_key(client: TestClient, path: str) -> None:
    response = client.get(path, params={"group_by": "user_id"})
    assert response.status_code in (401, 403)


@pytest.mark.parametrize("path", [SUMMARY_PATH, COUNT_PATH, SERIES_PATH])
def test_read_endpoints_are_absent_in_hybrid_mode(monkeypatch: pytest.MonkeyPatch, path: str) -> None:
    monkeypatch.setenv("OTARI_AI_TOKEN", "gw_test_token")
    config = GatewayConfig(mode="hybrid", platform={"base_url": "http://localhost:8100/api/v1"})
    app = create_app(config)

    with TestClient(app) as hybrid_client:
        response = hybrid_client.get(path, params={"group_by": "user_id"})

    assert response.status_code == 404
    reset_config()
    reset_db()
