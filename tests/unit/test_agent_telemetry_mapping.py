from datetime import UTC, datetime

from gateway.models.entities import AgentTelemetry
from gateway.services.agent_telemetry_service import TelemetryRecord, event_dedup_key, map_behavioral_event

_METRIC_ONLY_FIELDS = ("kind", "value", "temporality", "series_start", "series_key")


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


def test_dedup_key_distinguishes_tool_use_id() -> None:
    timestamp = datetime(2026, 8, 6, tzinfo=UTC)
    attrs_a = {"tool.name": "Bash", "duration_ms": 120, "success": True, "tool_use_id": "tool-1"}
    attrs_b = {**attrs_a, "tool_use_id": "tool-2"}
    event_a = map_behavioral_event("tool_result", attrs_a, timestamp=timestamp, source="claude_code", user_id="alice")
    event_b = map_behavioral_event("tool_result", attrs_b, timestamp=timestamp, source="claude_code", user_id="alice")
    assert event_a is not None
    assert event_b is not None
    assert event_a.dedup_key != event_b.dedup_key


def test_dedup_key_distinguishes_event_sequence() -> None:
    timestamp = datetime(2026, 8, 6, tzinfo=UTC)
    attrs_a = {"tool.name": "Bash", "duration_ms": 120, "success": True, "event.sequence": 1}
    attrs_b = {**attrs_a, "event.sequence": 2}
    event_a = map_behavioral_event("tool_result", attrs_a, timestamp=timestamp, source="claude_code", user_id="alice")
    event_b = map_behavioral_event("tool_result", attrs_b, timestamp=timestamp, source="claude_code", user_id="alice")
    assert event_a is not None
    assert event_b is not None
    assert event_a.dedup_key != event_b.dedup_key


def test_dedup_key_falls_back_without_tool_use_id_or_sequence() -> None:
    """A source that supplies neither field still gets a key, and a retry of the
    exact same event (still missing both) still collides on it (FR-003)."""
    timestamp = datetime(2026, 8, 6, tzinfo=UTC)
    attrs = {"tool.name": "Bash", "duration_ms": 120, "success": True}
    event_a = map_behavioral_event("tool_result", attrs, timestamp=timestamp, source="claude_code", user_id="alice")
    event_b = map_behavioral_event(
        "tool_result", dict(attrs), timestamp=timestamp, source="claude_code", user_id="alice"
    )
    assert event_a is not None
    assert event_b is not None
    assert event_a.dedup_key
    assert event_a.dedup_key == event_b.dedup_key


def test_no_metric_only_fields_on_telemetry_record_or_agent_telemetry() -> None:
    """This feature's schema ships only what it populates (FR-007): the metric-only
    columns pre-shaped for the not-yet-built metrics receiver are never added."""
    for attribute in _METRIC_ONLY_FIELDS:
        assert not hasattr(TelemetryRecord, attribute)
        assert attribute not in TelemetryRecord.__dataclass_fields__
        assert not hasattr(AgentTelemetry, attribute)
