"""Integration coverage for the OTLP metrics receiver (POST /v1/metrics).

Claude Code ships its outcome counters (lines changed, commits, pull requests,
active time) on the metrics signal, separately from the logs signal that carries
usage and behavioral events. They are recorded content-free and non-billable.
"""

import gzip
import json
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

from fastapi.testclient import TestClient
from sqlalchemy import Engine, event
from sqlalchemy.orm import Session

from gateway.api.routes.otlp import _MAX_METRIC_DATA_POINTS
from gateway.models.entities import AgentTelemetry, UsageLog, User

from .otlp_helpers import gauge_metric, metrics_export, metrics_export_protobuf, number_point, sum_metric

_PATH = "/v1/metrics"
_START = 1784000000000000000
_POINT = 1784000060000000000
# The batch size SC-009 names, and the ceiling on the statements it may cost.
# The exact count is SQLAlchemy's own executemany batching; the bound only has to
# be far enough below the point count to rule out a row-at-a-time regression.
_BULK_POINTS = 1000
_MAX_BULK_INSERT_STATEMENTS = 10


@contextmanager
def _agent_telemetry_inserts() -> Iterator[list[str]]:
    """Record every INSERT issued against agent_telemetry while the block runs.

    The `client` fixture owns its async engine privately, so the listener goes on
    the `Engine` class rather than one instance; an `AsyncEngine` drives a sync
    `Engine` underneath, so the gateway's own statements arrive here too.
    """
    statements: list[str] = []

    def record(
        conn: Any, cursor: Any, statement: str, parameters: Any, context: Any, executemany: bool
    ) -> None:
        if statement.lstrip().upper().startswith("INSERT INTO AGENT_TELEMETRY"):
            statements.append(statement)

    event.listen(Engine, "before_cursor_execute", record)
    try:
        yield statements
    finally:
        event.remove(Engine, "before_cursor_execute", record)


def _exempt_key(client: TestClient, master_key_header: dict[str, str], user_id: str = "alice") -> dict[str, str]:
    client.post("/v1/users", json={"user_id": user_id}, headers=master_key_header)
    response = client.post(
        "/v1/keys",
        json={"key_name": f"metrics-import-{user_id}", "user_id": user_id, "exclude_from_budget": True},
        headers=master_key_header,
    )
    assert response.status_code == 200, response.text
    return {"Otari-Key": f"Bearer {response.json()['key']}"}


def _outcome_export() -> dict[str, Any]:
    return metrics_export(
        sum_metric(
            "claude_code.lines_of_code.count",
            number_point(_POINT, 12, start=_START, type="added", **{"session.id": "s-1"}),
            number_point(_POINT, 5, start=_START, type="removed", **{"session.id": "s-1"}),
        ),
        sum_metric("claude_code.commit.count", number_point(_POINT, 2, start=_START, **{"session.id": "s-1"})),
        sum_metric("claude_code.pull_request.count", number_point(_POINT, 1, start=_START, **{"session.id": "s-1"})),
        sum_metric("claude_code.active_time.total", number_point(_POINT, 930, start=_START, **{"session.id": "s-1"})),
    )


def test_metrics_export_records_outcome_counters(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    headers = _exempt_key(client, master_key_header)
    response = client.post(_PATH, json=_outcome_export(), headers=headers)

    assert response.status_code == 200, response.text
    rows = db_session.query(AgentTelemetry).order_by(AgentTelemetry.name, AgentTelemetry.value).all()
    assert [row.name for row in rows] == [
        "claude_code.active_time.total",
        "claude_code.commit.count",
        "claude_code.lines_of_code.count",
        "claude_code.lines_of_code.count",
        "claude_code.pull_request.count",
    ]
    assert all(row.kind == "metric" for row in rows)
    assert all(row.user_id == "alice" and row.api_key_id is not None for row in rows)
    assert all(row.source == "claude_code" and row.session_label == "s-1" for row in rows)
    assert all(row.temporality == "cumulative" and row.series_start is not None for row in rows)
    lines = [row for row in rows if row.name == "claude_code.lines_of_code.count"]
    # Dimensioned points stay two distinct series, never collapsed into one row (FR-007).
    assert sorted(row.value or 0.0 for row in lines) == [5.0, 12.0]
    assert len({row.series_key for row in lines}) == 2
    # Content-free: the ``type`` dimension is folded into the series hash, not stored.
    assert all("added" not in str(row.__dict__) for row in rows)
    assert db_session.query(UsageLog).count() == 0


def test_metrics_export_accepts_protobuf_and_gzip(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    headers = {
        **_exempt_key(client, master_key_header, "protobuf-user"),
        "Content-Type": "application/x-protobuf",
        "Content-Encoding": "gzip",
    }
    body = gzip.compress(
        metrics_export_protobuf(
            sum_metric("claude_code.commit.count", number_point(_POINT, 3, start=_START, **{"session.id": "s-pb"}))
        )
    )
    response = client.post(_PATH, content=body, headers=headers)

    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("application/x-protobuf")
    row = db_session.query(AgentTelemetry).filter(AgentTelemetry.session_label == "s-pb").one()
    assert row.name == "claude_code.commit.count" and row.value == 3.0


def test_metrics_export_records_gauge_points_as_deltas(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    """A gauge is not a running total, so each point is its own increment."""
    headers = _exempt_key(client, master_key_header, "gauge-user")
    response = client.post(
        _PATH,
        json=metrics_export(
            gauge_metric(
                "claude_code.active_time.total", number_point(_POINT, 45, start=_START, **{"session.id": "s-gauge"})
            )
        ),
        headers=headers,
    )

    assert response.status_code == 200, response.text
    row = db_session.query(AgentTelemetry).filter(AgentTelemetry.session_label == "s-gauge").one()
    assert row.temporality == "delta" and row.value == 45.0


def test_metrics_export_refuses_master_key(client: TestClient, master_key_header: dict[str, str]) -> None:
    response = client.post(_PATH, json=_outcome_export(), headers=master_key_header)
    assert response.status_code == 403


def test_metrics_export_skips_metrics_captured_elsewhere(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    """Already-billed and already-behavioral metrics are dropped, not stored (FR-004)."""
    headers = _exempt_key(client, master_key_header, "skipper")
    response = client.post(
        _PATH,
        json=metrics_export(
            sum_metric("claude_code.token.usage", number_point(_POINT, 4321, start=_START, type="input")),
            sum_metric("claude_code.cost.usage", number_point(_POINT, 12.5, start=_START)),
            sum_metric("claude_code.code_edit_tool.decision", number_point(_POINT, 1, start=_START, decision="accept")),
            sum_metric("claude_code.commit.count", number_point(_POINT, 1, start=_START, **{"session.id": "s-skip"})),
        ),
        headers=headers,
    )

    assert response.status_code == 200, response.text
    rows = db_session.query(AgentTelemetry).all()
    assert [row.name for row in rows] == ["claude_code.commit.count"]

    summary = client.get("/v1/usage/summary", headers=master_key_header)
    assert summary.status_code == 200, summary.text
    assert summary.json()["totals"]["cost"] == 0.0
    assert summary.json()["totals"]["request_count"] == 0


def test_metrics_export_accepts_unrecognized_names(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    headers = _exempt_key(client, master_key_header, "future-agent")
    response = client.post(
        _PATH,
        json=metrics_export(
            sum_metric("claude_code.session.count", number_point(_POINT, 1, start=_START)),
            sum_metric("claude_code.brand_new.metric", number_point(_POINT, 9, start=_START)),
            sum_metric("claude_code.commit.count", number_point(_POINT, 1, start=_START, **{"session.id": "s-new"})),
        ),
        headers=headers,
    )

    assert response.status_code == 200, response.text
    assert [row.name for row in db_session.query(AgentTelemetry).all()] == ["claude_code.commit.count"]


def test_metrics_export_is_idempotent_across_replays(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    """A retried export, and a cumulative series re-reported in full, add nothing."""
    headers = _exempt_key(client, master_key_header, "replayer")
    export = _outcome_export()
    assert client.post(_PATH, json=export, headers=headers).status_code == 200
    assert client.post(_PATH, json=export, headers=headers).status_code == 200
    assert db_session.query(AgentTelemetry).count() == 5

    # A later export of the same series at a new point time is a new row; the
    # already-reported point in it is still deduped.
    later = metrics_export(
        sum_metric(
            "claude_code.commit.count",
            number_point(_POINT, 2, start=_START, **{"session.id": "s-1"}),
            number_point(_POINT + 60_000_000_000, 3, start=_START, **{"session.id": "s-1"}),
        )
    )
    assert client.post(_PATH, json=later, headers=headers).status_code == 200
    assert db_session.query(AgentTelemetry).filter(AgentTelemetry.name == "claude_code.commit.count").count() == 2


def test_metrics_export_rejects_points_for_a_soft_deleted_user(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    headers = _exempt_key(client, master_key_header, "doomed-metrics")
    db_session.query(User).filter(User.user_id == "doomed-metrics").update({"deleted_at": datetime.now(UTC)})
    db_session.commit()

    response = client.post(
        _PATH,
        json=metrics_export(
            sum_metric("claude_code.commit.count", number_point(_POINT, 1, start=_START, **{"session.id": "s-gone"}))
        ),
        headers=headers,
    )

    assert response.status_code == 200, response.text
    assert int(response.json()["partialSuccess"]["rejectedDataPoints"]) == 1
    assert db_session.query(AgentTelemetry).count() == 0


def test_metrics_export_rejects_too_many_data_points(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    """The data-point cap is the metrics endpoint's own, not the logs event cap (R14)."""
    headers = _exempt_key(client, master_key_header, "flooder")
    points = [
        number_point(_POINT + index, 1, start=_START, **{"session.id": f"s-{index}"})
        for index in range(_MAX_METRIC_DATA_POINTS + 1)
    ]
    response = client.post(
        _PATH, json=metrics_export(sum_metric("claude_code.commit.count", *points)), headers=headers
    )

    assert response.status_code == 413, response.text
    assert db_session.query(AgentTelemetry).count() == 0


def test_metrics_export_rejects_oversized_and_bomb_bodies(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    key_headers = _exempt_key(client, master_key_header, "oversized")
    oversized = client.post(
        _PATH,
        content=b"x" * (8 * 1024 * 1024 + 1),
        headers={**key_headers, "Content-Type": "application/x-protobuf"},
    )
    assert oversized.status_code == 413

    bomb = client.post(
        _PATH,
        content=gzip.compress(b"\0" * (8 * 1024 * 1024 + 1)),
        headers={**key_headers, "Content-Type": "application/x-protobuf", "Content-Encoding": "gzip"},
    )
    assert bomb.status_code == 413


def test_metrics_export_rejects_malformed_bodies(client: TestClient, master_key_header: dict[str, str]) -> None:
    key_headers = _exempt_key(client, master_key_header, "malformed")
    garbage = client.post(
        _PATH, content=b"\xff\xfe\x01", headers={**key_headers, "Content-Type": "application/x-protobuf"}
    )
    assert garbage.status_code == 400

    not_json = client.post(
        _PATH, content=b"{not json", headers={**key_headers, "Content-Type": "application/json"}
    )
    assert not_json.status_code == 400

    unsupported = client.post(_PATH, content=b"hi", headers={**key_headers, "Content-Type": "text/plain"})
    assert unsupported.status_code == 415


def test_metrics_export_bulk_inserts_without_a_round_trip_per_point(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    """A collision-free export of SC-009's batch costs bounded inserts (FR-018).

    The row count alone would not catch the regression this guards against, since
    a row-at-a-time ingest stores exactly the same rows; the statement count is
    what separates the two.
    """
    headers = _exempt_key(client, master_key_header, "bulky")
    # One point per second: OTLP point times are nanoseconds, but a datetime holds
    # microseconds, so points spaced a nanosecond apart would be one instant and
    # dedup to a single row.
    points = [
        number_point(_POINT + index * 1_000_000_000, index, start=_START, **{"session.id": "s-bulk"})
        for index in range(_BULK_POINTS)
    ]
    with _agent_telemetry_inserts() as inserts:
        response = client.post(
            _PATH, json=metrics_export(sum_metric("claude_code.commit.count", *points)), headers=headers
        )

    assert response.status_code == 200, response.text
    stored = db_session.query(AgentTelemetry).filter(AgentTelemetry.session_label == "s-bulk").count()
    assert stored == _BULK_POINTS
    assert 0 < len(inserts) <= _MAX_BULK_INSERT_STATEMENTS


def test_metrics_export_records_a_point_carrying_no_session(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    """A point with no session.id is still attributed to the importing user.

    It simply cannot take part in a per-session join afterwards (spec Edge Case
    "Missing session or attribution fields").
    """
    headers = _exempt_key(client, master_key_header, "sessionless")
    response = client.post(
        _PATH,
        json=metrics_export(sum_metric("claude_code.commit.count", number_point(_POINT, 7, start=_START))),
        headers=headers,
    )

    assert response.status_code == 200, response.text
    row = db_session.query(AgentTelemetry).one()
    assert row.session_label is None
    assert row.user_id == "sessionless" and row.kind == "metric" and row.value == 7.0


# ---------------------------------------------------------------------------
# User Story 3: the existing capture toggle also governs outcome metrics.
# ---------------------------------------------------------------------------


def _key_with_capture_override(
    client: TestClient, master_key_header: dict[str, str], user_id: str, *, capture: bool | None
) -> tuple[dict[str, str], str]:
    client.post("/v1/users", json={"user_id": user_id}, headers=master_key_header)
    payload: dict[str, object] = {
        "key_name": f"metrics-import-{user_id}",
        "user_id": user_id,
        "exclude_from_budget": True,
    }
    if capture is not None:
        payload["capture_agent_telemetry"] = capture
    response = client.post("/v1/keys", json=payload, headers=master_key_header)
    assert response.status_code == 200, response.text
    body = response.json()
    return {"Otari-Key": f"Bearer {body['key']}"}, str(body["id"])


def test_capture_toggle_off_blocks_metric_rows_but_not_usage(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    headers, key_id = _key_with_capture_override(client, master_key_header, "quiet", capture=False)
    response = client.post(_PATH, json=_outcome_export(), headers=headers)

    assert response.status_code == 200, response.text
    assert json.loads(response.text).get("partialSuccess", {}) == {}
    assert db_session.query(AgentTelemetry).count() == 0
    assert db_session.query(UsageLog).count() == 0

    patch = client.patch(f"/v1/keys/{key_id}", json={"capture_agent_telemetry": None}, headers=master_key_header)
    assert patch.status_code == 200, patch.text
    assert patch.json()["capture_agent_telemetry"] is None

    response = client.post(_PATH, json=_outcome_export(), headers=headers)
    assert response.status_code == 200, response.text
    assert db_session.query(AgentTelemetry).count() == 5
