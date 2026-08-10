"""Integration coverage for content-free Claude Code behavioral events."""

from datetime import UTC, datetime

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from gateway.api.routes.otlp import _MAX_EVENTS_PER_EXPORT
from gateway.models.entities import AgentTelemetry, UsageLog, User

from .otlp_helpers import log_record, logs_export


def _exempt_key(client: TestClient, master_key_header: dict[str, str], user_id: str = "alice") -> dict[str, str]:
    client.post("/v1/users", json={"user_id": user_id}, headers=master_key_header)
    response = client.post(
        "/v1/keys",
        json={"key_name": f"logs-import-{user_id}", "user_id": user_id, "exclude_from_budget": True},
        headers=master_key_header,
    )
    assert response.status_code == 200
    return {"Otari-Key": f"Bearer {response.json()['key']}"}


def test_otlp_logs_record_only_allowlisted_behavior(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    headers = _exempt_key(client, master_key_header)
    response = client.post(
        "/v1/logs",
        json=logs_export(
            log_record(
                1784000000000000000,
                **{
                    "event.name": "tool_result",
                    "session.id": "s-1",
                    "tool.name": "Bash",
                    "success": True,
                    "input": "secret",
                },
            ),
            log_record(
                1784000001000000000,
                **{"event.name": "user_prompt", "session.id": "s-1", "prompt": "secret", "prompt_length": 19},
            ),
            log_record(1784000002000000000, **{"event.name": "unknown", "prompt": "secret"}),
        ),
        headers=headers,
    )

    assert response.status_code == 200, response.text
    rows = db_session.query(AgentTelemetry).order_by(AgentTelemetry.name).all()
    assert [row.name for row in rows] == ["tool_result", "user_prompt"]
    assert rows[0].tool_name == "Bash" and rows[0].success is True
    assert rows[1].prompt_length == 19
    assert all("secret" not in str(row.__dict__) for row in rows)
    assert db_session.query(UsageLog).count() == 0


def test_otlp_logs_read_string_encoded_tool_success(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    """Claude Code emits ``success`` as the string "true"/"false", not a boolValue."""
    headers = _exempt_key(client, master_key_header, "stringly")
    response = client.post(
        "/v1/logs",
        json=logs_export(
            log_record(
                1784000010000000000,
                **{
                    "event.name": "tool_result",
                    "session.id": "s-str",
                    "tool_name": "Bash",
                    "success": "true",
                    "duration_ms": 1234,
                },
            ),
            log_record(
                1784000011000000000,
                **{"event.name": "tool_result", "session.id": "s-str", "tool_name": "Read", "success": "false"},
            ),
        ),
        headers=headers,
    )

    assert response.status_code == 200, response.text
    rows = db_session.query(AgentTelemetry).filter(AgentTelemetry.session_label == "s-str").all()
    assert {(row.tool_name, row.success) for row in rows} == {("Bash", True), ("Read", False)}


def test_otlp_logs_reject_behavior_for_soft_deleted_user(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    """A live key whose user was soft-deleted must not keep accruing telemetry."""
    headers = _exempt_key(client, master_key_header, "doomed-behavioral")
    db_session.query(User).filter(User.user_id == "doomed-behavioral").update({"deleted_at": datetime.now(UTC)})
    db_session.commit()

    response = client.post(
        "/v1/logs",
        json=logs_export(
            log_record(1784000020000000000, **{"event.name": "user_prompt", "session.id": "s-gone", "prompt_length": 7})
        ),
        headers=headers,
    )

    assert response.status_code == 200, response.text
    assert int(response.json()["partialSuccess"]["rejectedLogRecords"]) == 1
    assert db_session.query(AgentTelemetry).filter(AgentTelemetry.session_label == "s-gone").count() == 0


def test_otlp_logs_export_limit_counts_usage_and_behavior_together(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    """The per-export row bound covers both event families, not each on its own.

    Usage and behavioral events are disjoint by ``event.name``, so an export can hold
    a full allowance of each and stay under a per-list check while persisting twice
    the intended number of rows.
    """
    headers = _exempt_key(client, master_key_header, "floody")
    half = _MAX_EVENTS_PER_EXPORT // 2
    usage = [
        log_record(
            1784000030000000000 + index,
            **{
                "event.name": "api_request",
                "model": "claude-opus-4-8",
                "input_tokens": 10,
                "output_tokens": 5,
                "request_id": f"req-flood-{index}",
            },
        )
        for index in range(half + 1)
    ]
    behavior = [
        log_record(
            1784000040000000000 + index,
            **{"event.name": "user_prompt", "session.id": "s-flood", "prompt_length": index},
        )
        for index in range(half)
    ]
    assert len(usage) <= _MAX_EVENTS_PER_EXPORT and len(behavior) <= _MAX_EVENTS_PER_EXPORT
    assert len(usage) + len(behavior) > _MAX_EVENTS_PER_EXPORT

    response = client.post("/v1/logs", json=logs_export(*usage, *behavior), headers=headers)

    assert response.status_code == 413, response.text
    assert db_session.query(AgentTelemetry).filter(AgentTelemetry.session_label == "s-flood").count() == 0
    assert db_session.query(UsageLog).count() == 0
