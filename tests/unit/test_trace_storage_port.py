"""The trace storage port's document contract, and how the composition root binds it.

The contract is what keeps a client's text out of a span whichever adapter stores
it, so it is asserted on the record types themselves, before any adapter runs.
"""

import uuid
from typing import Any

import pytest

from gateway.adapters.trace_storage_adapter import LocalTraceStorage, NullTraceStorage
from gateway.container import ContainerError, build_container
from gateway.core.config import GatewayConfig
from gateway.core.unit_of_work import UnitOfWork
from gateway.ports.trace_storage_port import (
    InvalidSpanError,
    SpanRecord,
    TraceScope,
    TraceStoragePort,
    TraceWrite,
    is_identifier,
)
from gateway.types.trace_tables import TraceTables

_HYBRID = GatewayConfig(mode="hybrid", platform={"base_url": "http://platform.test/api/v1"})


def _span(**overrides: Any) -> SpanRecord:
    fields: dict[str, Any] = {"span_id": "req-1", "kind": "step", "origin": "gateway", "name": "chat", "outcome": "ok"}
    return SpanRecord(**(fields | overrides))


def _no_tables(uow: UnitOfWork) -> TraceTables:
    raise AssertionError("the binding must not build repositories at resolve time")


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


def test_a_deployment_serving_the_control_plane_keeps_traces_in_its_database() -> None:
    container = build_container(config=GatewayConfig(), workspace_listener=None, trace_tables=_no_tables)

    store = container.resolve(TraceStoragePort, None)

    assert isinstance(store, LocalTraceStorage)
    assert container.resolve(TraceStoragePort, None) is store, "one store serves every request"


def test_the_database_store_refuses_to_build_without_its_repositories() -> None:
    container = build_container(config=GatewayConfig(), workspace_listener=None)

    with pytest.raises(ContainerError, match="traces repositories builder"):
        container.resolve(TraceStoragePort, None)


def test_a_data_plane_only_gateway_keeps_no_traces() -> None:
    """With no database and no peer to send them to yet, recording drops them."""
    container = build_container(config=_HYBRID, workspace_listener=None, trace_tables=_no_tables)

    assert isinstance(container.resolve(TraceStoragePort, None), NullTraceStorage)


def test_trace_storage_is_chosen_once_when_the_container_is_built() -> None:
    config = GatewayConfig()
    container = build_container(config=config, workspace_listener=None, trace_tables=_no_tables)
    config.mode = "hybrid"

    assert isinstance(container.resolve(TraceStoragePort, None), LocalTraceStorage)


def test_trace_storage_refuses_a_container_built_without_config() -> None:
    container = build_container(workspace_listener=None)

    with pytest.raises(ContainerError, match="TraceStoragePort"):
        container.resolve(TraceStoragePort, None)


def test_a_span_keeps_the_attributes_it_was_checked_with() -> None:
    attributes: dict[str, str | int | float | bool] = {"http.response.status_code": 200}
    span = SpanRecord(span_id="s1", kind="step", origin="gateway", name="step", outcome="ok", attributes=attributes)

    attributes["prose"] = "not an identifier at all"

    assert dict(span.attributes) == {"http.response.status_code": 200}
    with pytest.raises(TypeError):
        span.attributes["prose"] = "x"  # type: ignore[index]
