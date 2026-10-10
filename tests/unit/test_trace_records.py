"""The trace records' document contract.

The contract is what keeps a client's text out of a span, so it is asserted on the
record types themselves, before anything stores one.
"""

import uuid
from typing import Any

import pytest

from gateway.types.traces import (
    InvalidSpanError,
    SpanRecord,
    TraceScope,
    TraceWrite,
    is_identifier,
)


def _span(**overrides: Any) -> SpanRecord:
    fields: dict[str, Any] = {"span_id": "req-1", "kind": "step", "origin": "gateway", "name": "chat", "outcome": "ok"}
    return SpanRecord(**(fields | overrides))


def test_a_well_formed_span_is_accepted() -> None:
    span = _span(tool_name="getGatewayOverview", attributes={"otari.routing.attempt_position": 2})

    assert span.attributes == {"otari.routing.attempt_position": 2}


@pytest.mark.parametrize(
    ("field", "value"),
    [("kind", "thought"), ("origin", "inferred"), ("outcome", "maybe")],
)
def test_a_value_outside_a_closed_vocabulary_is_refused(field: str, value: str) -> None:
    with pytest.raises(InvalidSpanError, match=field):
        _span(**{field: value})


def test_an_attribute_off_the_allowlist_is_refused() -> None:
    """Verbatim attribute passthrough is not part of the shape, so an unknown key never reaches an adapter."""
    with pytest.raises(InvalidSpanError, match="allowlisted"):
        _span(attributes={"gen_ai.input.messages": "hello"})


def test_an_attribute_of_the_wrong_type_is_refused() -> None:
    with pytest.raises(InvalidSpanError, match="int"):
        _span(attributes={"otari.routing.attempt_position": "2"})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("name", "summarize: the quarterly report for Acme"),
        ("tool_name", "rm -rf /"),
        ("error_class", "The provider said: your key sk-123 is invalid"),
    ],
)
def test_prose_in_a_client_supplied_string_is_refused(field: str, value: str) -> None:
    """A name a client chose can carry text; only identifier-shaped strings are stored."""
    with pytest.raises(InvalidSpanError, match=field) as raised:
        _span(**{field: value})
    assert value not in str(raised.value), "the error must name the field, never echo its value"


@pytest.mark.parametrize("field", ["span_id", "parent_span_id", "otel_trace_id", "otel_span_id"])
def test_prose_in_an_id_is_refused(field: str) -> None:
    """An OTLP exporter chooses its own ids, so an id is held to the identifier shape too."""
    with pytest.raises(InvalidSpanError, match=field):
        _span(**{field: "my password is secret"})


def test_prose_in_a_trace_id_is_refused() -> None:
    with pytest.raises(InvalidSpanError, match="trace_id"):
        TraceWrite(
            workspace_id=uuid.uuid4(),
            trace_id="my password is secret",
            user_id="u",
            api_key_id=None,
            session_source="none",
            spans=(),
        )


def test_prose_in_an_attribute_string_is_refused() -> None:
    with pytest.raises(InvalidSpanError, match="identifier-shaped"):
        _span(attributes={"error.type": "connection reset by peer"})


def test_an_id_longer_than_the_tables_hold_is_refused() -> None:
    with pytest.raises(InvalidSpanError, match="span_id"):
        _span(span_id="x" * 65)


def test_a_trace_write_refuses_an_unknown_session_source() -> None:
    with pytest.raises(InvalidSpanError, match="session_source"):
        TraceWrite(
            workspace_id=uuid.uuid4(),
            trace_id="t-1",
            user_id="u",
            api_key_id=None,
            session_source="inferred",
            spans=(),
        )


@pytest.mark.parametrize(
    ("value", "expected"),
    [("claude-code", True), ("anthropic:claude-sonnet-4-5", True), ("two words", False), ("", False)],
)
def test_identifier_shape(value: str, expected: bool) -> None:
    assert is_identifier(value) is expected


def test_an_empty_workspace_scope_is_not_deployment_wide() -> None:
    """Every workspace is a deliberate choice; a caller with no workspaces sees none."""
    assert TraceScope.workspaces(frozenset()).deployment_wide is False
    assert TraceScope.deployment().deployment_wide is True


def test_a_span_keeps_the_attributes_it_was_checked_with() -> None:
    attributes: dict[str, str | int | float | bool] = {"http.response.status_code": 200}
    span = SpanRecord(span_id="s1", kind="step", origin="gateway", name="step", outcome="ok", attributes=attributes)

    attributes["prose"] = "not an identifier at all"

    assert dict(span.attributes) == {"http.response.status_code": 200}
    with pytest.raises(TypeError):
        span.attributes["prose"] = "x"  # type: ignore[index]


@pytest.mark.parametrize(
    "value",
    ["alice@example.com", "/Users/alice/notes.txt", "src/secret/notes.txt", "~/x", "sk-ant-api03-abc", "ghp_abc123"],
)
def test_a_name_shaped_like_personal_data_or_a_credential_is_refused(value: str) -> None:
    with pytest.raises(InvalidSpanError, match="tool_name"):
        _span(tool_name=value)


@pytest.mark.parametrize(
    "value", ["openai/gpt-4o", "mzai:ovhcloud:gpt-oss-120b", "claude-3-5-sonnet@20240620", "mcp__github__create_issue"]
)
def test_a_real_model_or_tool_name_is_kept(value: str) -> None:
    assert is_identifier(value)


def test_an_id_takes_no_slash_or_at_sign() -> None:
    with pytest.raises(InvalidSpanError, match="span_id"):
        _span(span_id="a/b@c")
