from datetime import UTC, datetime

from gateway.services.agent_telemetry_service import event_dedup_key, map_behavioral_event


def test_behavioral_mapping_keeps_only_allowlisted_attributes() -> None:
    timestamp = datetime(2026, 8, 6, tzinfo=UTC)
    prompt = map_behavioral_event(
        "user_prompt",
        {"session.id": "session-1", "prompt": "do not persist", "prompt_length": 42},
        timestamp=timestamp,
        source="claude_code",
        user_id="alice",
    )
    assert prompt is not None
    assert prompt.prompt_length == 42
    assert prompt.tool_name is None
    assert prompt.decision is None
    assert prompt.status_code is None

    event = map_behavioral_event(
        "tool_result",
        {"session.id": "session-1", "tool.name": "Bash", "success": True, "input": "secret"},
        timestamp=timestamp,
        source="claude_code",
        user_id="alice",
    )
    assert event is not None
    assert event.tool_name == "Bash"
    assert event.success is True
    assert event_dedup_key(event, "alice") == event.dedup_key
