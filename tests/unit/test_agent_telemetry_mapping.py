from datetime import UTC, datetime

from gateway.services.agent_telemetry_service import (
    TelemetryRecord,
    event_dedup_key,
    map_behavioral_event,
    map_metric_point,
    metric_dedup_key,
    metric_series_key,
)

_POINT_TS = datetime(2026, 8, 12, 9, 0, tzinfo=UTC)
_SERIES_START = datetime(2026, 8, 12, 8, 0, tzinfo=UTC)


def _metric(
    name: str, value: float = 3.0, temporality: str = "cumulative", **attrs: object
) -> TelemetryRecord | None:
    return map_metric_point(
        name,
        value,
        temporality,
        _SERIES_START,
        attrs,
        timestamp=_POINT_TS,
        source="claude_code",
        user_id="alice",
    )


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


def test_metric_mapping_records_each_outcome_counter() -> None:
    """The four outcome counters map to content-free metric rows (FR-003)."""
    for name, value in (
        ("claude_code.commit.count", 2.0),
        ("claude_code.pull_request.count", 1.0),
        ("claude_code.active_time.total", 930.0),
    ):
        record = _metric(name, value, "cumulative", **{"session.id": "s-1"})
        assert record is not None, name
        assert record.kind == "metric"
        assert record.name == name
        assert record.value == value
        assert record.temporality == "cumulative"
        assert record.series_start == _SERIES_START
        assert record.timestamp == _POINT_TS
        assert record.series_key
        assert record.session_label == "s-1"
        assert record.dedup_key == metric_dedup_key(record, "alice")


def test_metric_mapping_keeps_dimensioned_lines_of_code_points_apart() -> None:
    """``type=added`` and ``type=removed`` are two series, never collapsed (FR-007/R5)."""
    added = _metric("claude_code.lines_of_code.count", 12.0, "delta", type="added")
    removed = _metric("claude_code.lines_of_code.count", 5.0, "delta", type="removed")
    assert added is not None and removed is not None
    assert added.kind == removed.kind == "metric"
    assert added.value == 12.0
    assert removed.value == 5.0
    assert added.temporality == removed.temporality == "delta"
    assert added.series_key != removed.series_key
    assert added.dedup_key != removed.dedup_key


def test_metric_mapping_drops_attributes_it_does_not_model() -> None:
    """Only the allow-listed metric columns are populated (FR-005)."""
    record = _metric("claude_code.commit.count", 1.0, "delta", **{"session.id": "s-1", "branch": "secret-branch"})
    assert record is not None
    assert "secret-branch" not in str(record.__dict__)
    assert record.tool_name is None
    assert record.decision is None
    assert record.prompt_length is None


def test_metric_mapping_skips_metrics_captured_elsewhere() -> None:
    """Already-billed or already-behavioral signals are recognized and skipped (FR-004)."""
    for name in (
        "claude_code.token.usage",
        "claude_code.cost.usage",
        "claude_code.code_edit_tool.decision",
    ):
        assert _metric(name) is None, name


def test_metric_mapping_skips_unrecognized_names() -> None:
    """An unknown name (and the deferred session counter) is skipped, not an error (FR-009)."""
    assert _metric("claude_code.session.count") is None
    assert _metric("claude_code.something.brand_new") is None


def test_metric_series_key_separates_dimensions_and_ignores_attribute_order() -> None:
    added = metric_series_key("claude_code.lines_of_code.count", {"type": "added"})
    removed = metric_series_key("claude_code.lines_of_code.count", {"type": "removed"})
    assert added != removed
    assert metric_series_key("m", {"a": "1", "b": "2"}) == metric_series_key("m", {"b": "2", "a": "1"})


def test_metric_dedup_key_is_stable_and_user_scoped() -> None:
    """Same point, same key; same point under another user, a different key (R6)."""
    record = _metric("claude_code.commit.count", 4.0)
    assert record is not None
    assert metric_dedup_key(record, "alice") == metric_dedup_key(record, "alice")
    assert metric_dedup_key(record, "alice") != metric_dedup_key(record, "bob")
