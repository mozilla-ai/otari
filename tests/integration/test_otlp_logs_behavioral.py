"""Integration coverage for content-free Claude Code behavioral events."""

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from gateway.models.entities import AgentTelemetry, UsageLog

from .otlp_helpers import log_record, logs_export


def _exempt_key(client: TestClient, master_key_header: dict[str, str]) -> dict[str, str]:
    client.post("/v1/users", json={"user_id": "alice"}, headers=master_key_header)
    response = client.post(
        "/v1/keys",
        json={"key_name": "logs-import", "user_id": "alice", "exclude_from_budget": True},
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
