"""Tests for OpenTelemetry context propagation middleware."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from opentelemetry import propagate, trace
from opentelemetry.context import Context
from opentelemetry.sdk.trace import TracerProvider
from starlette.responses import StreamingResponse

from gateway.context_propagation import extract_trace_context
from gateway.core.config import GatewayConfig
from gateway.main import create_app


@pytest.fixture(autouse=True)
def _isolate_config_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in os.environ:
        if name.startswith("OTARI_"):
            monkeypatch.delenv(name)


def _test_config(tmp_path: Path, **overrides: Any) -> GatewayConfig:
    return GatewayConfig(
        _env_file=None,  # type: ignore[call-arg]  # BaseSettings accepts this; pyright infers __init__ from fields only.
        database_url=f"sqlite:///{tmp_path / 'trace-test.db'}",
        master_key="sk-test-master",
        mode="standalone",
        require_pricing=False,
        **overrides,
    )


@pytest.mark.parametrize(
    ("traceparent", "expected_trace_id"),
    [
        ("00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01", "4bf92f3577b34da6a3ce929d0e0e4736"),
        ("00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-00", "4bf92f3577b34da6a3ce929d0e0e4736"),
        ("00-ffffffffffffffffffffffffffffffff-aaaaaaaaaaaaaaaa-01", "ffffffffffffffffffffffffffffffff"),
    ],
)
def test_http_request_span_inherits_traceparent(
    traceparent: str,
    expected_trace_id: str,
    tmp_path: Path,
) -> None:
    app = create_app(_test_config(tmp_path, accept_incoming_trace_context=True))
    provider = TracerProvider()
    tracer = provider.get_tracer(__name__)

    @app.get("/test-trace-span")
    async def generate_trace_span() -> dict[str, str]:
        with tracer.start_as_current_span("request_span") as span:
            span_context = span.get_span_context()
            return {
                "trace_id": format(span_context.trace_id, "032x"),
                "vendor_tracestate": span_context.trace_state.get("vendor") or "",
            }

    with TestClient(app) as client:
        response = client.get(
            "/test-trace-span",
            headers={
                "traceparent": traceparent,
                "tracestate": "vendor=value",
            },
        )

    assert response.status_code == 200
    assert response.json()["trace_id"] == expected_trace_id
    assert response.json()["vendor_tracestate"] == "value"


@pytest.mark.parametrize(
    ("invalid_traceparent", "forbidden_trace_id"),
    [
        ("invalid", None),
        ("00", None),
        ("00-short-parts", None),
        ("00-00000000000000000000000000000000-00f067aa0ba902b7-01", "00000000000000000000000000000000"),
        ("00-4bf92f3577b34da6a3ce929d0e0e4736-0000000000000000-01", "4bf92f3577b34da6a3ce929d0e0e4736"),
    ],
)
def test_invalid_traceparent_creates_new_root_span(
    invalid_traceparent: str,
    forbidden_trace_id: str | None,
    tmp_path: Path,
) -> None:
    app = create_app(_test_config(tmp_path, accept_incoming_trace_context=True))
    provider = TracerProvider()
    tracer = provider.get_tracer(__name__)

    @app.get("/test-invalid-traceparent")
    async def generate_trace_span() -> dict[str, str]:
        with tracer.start_as_current_span("request_span") as span:
            return {"trace_id": format(span.get_span_context().trace_id, "032x")}

    with TestClient(app) as client:
        response = client.get(
            "/test-invalid-traceparent",
            headers={
                "traceparent": invalid_traceparent,
            },
        )

    assert response.status_code == 200
    trace_id = response.json()["trace_id"]
    assert trace_id != "00000000000000000000000000000000"
    if forbidden_trace_id is not None:
        assert trace_id != forbidden_trace_id


def test_missing_traceparent_creates_new_root_span(tmp_path: Path) -> None:
    app = create_app(_test_config(tmp_path))
    provider = TracerProvider()
    tracer = provider.get_tracer(__name__)

    @app.get("/test-missing-traceparent")
    async def generate_trace_span() -> dict[str, str]:
        with tracer.start_as_current_span("request_span") as span:
            return {"trace_id": format(span.get_span_context().trace_id, "032x")}

    with TestClient(app) as client:
        response = client.get("/test-missing-traceparent")

    assert response.status_code == 200
    assert response.json()["trace_id"] != "00000000000000000000000000000000"


def test_extract_trace_context_valid_format() -> None:
    context = extract_trace_context(
        {"traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"}
    )
    assert context is not None


def test_extract_trace_context_uses_configured_global_propagator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configured_propagator = Mock()
    configured_context = Context()
    configured_propagator.extract.return_value = configured_context
    monkeypatch.setattr(propagate, "get_global_textmap", lambda: configured_propagator)

    assert (
        extract_trace_context(
            {"traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"}
        )
        is configured_context
    )
    configured_propagator.extract.assert_called_once()


@pytest.mark.parametrize(
    "carrier",
    [
        {},
        {"traceparent": "invalid"},
        {"traceparent": "00-00000000000000000000000000000000-00f067aa0ba902b7-01"},
        {"traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-0000000000000000-01"},
    ],
)
def test_extract_trace_context_invalid_or_missing(carrier: dict[str, str]) -> None:
    assert isinstance(extract_trace_context(carrier), Context)


def test_streaming_response_span_inherits_traceparent(tmp_path: Path) -> None:
    app = create_app(_test_config(tmp_path, accept_incoming_trace_context=True))

    @app.get("/test-stream-trace-span")
    async def generate_stream_trace_span() -> StreamingResponse:
        async def stream() -> AsyncIterator[bytes]:
            trace_id = format(trace.get_current_span().get_span_context().trace_id, "032x")
            yield trace_id.encode("utf-8")

        return StreamingResponse(stream(), media_type="text/plain")

    incoming_trace_id = "4bf92f3577b34da6a3ce929d0e0e4736"
    with TestClient(app) as client:
        response = client.get(
            "/test-stream-trace-span",
            headers={"traceparent": f"00-{incoming_trace_id}-00f067aa0ba902b7-01"},
        )

    assert response.status_code == 200
    assert response.text == incoming_trace_id


def test_context_is_detached_after_request_completes(tmp_path: Path) -> None:
    app = create_app(_test_config(tmp_path, accept_incoming_trace_context=True))
    provider = TracerProvider()
    tracer = provider.get_tracer(__name__)

    @app.get("/test-context-isolation")
    async def generate_trace_span() -> dict[str, str]:
        with tracer.start_as_current_span("request_span") as span:
            return {"trace_id": format(span.get_span_context().trace_id, "032x")}

    incoming_trace_id = "4bf92f3577b34da6a3ce929d0e0e4736"
    with TestClient(app) as client:
        first_response = client.get(
            "/test-context-isolation",
            headers={"traceparent": f"00-{incoming_trace_id}-00f067aa0ba902b7-01"},
        )
        second_response = client.get("/test-context-isolation")

    assert first_response.status_code == 200
    assert first_response.json()["trace_id"] == incoming_trace_id

    assert second_response.status_code == 200
    assert second_response.json()["trace_id"] != incoming_trace_id


def test_trace_context_propagation_disabled_by_default(tmp_path: Path) -> None:
    app = create_app(_test_config(tmp_path))
    provider = TracerProvider()
    tracer = provider.get_tracer(__name__)

    @app.get("/test-default-disabled-trace-context")
    async def generate_trace_span() -> dict[str, str]:
        with tracer.start_as_current_span("request_span") as span:
            return {"trace_id": format(span.get_span_context().trace_id, "032x")}

    incoming_trace_id = "4bf92f3577b34da6a3ce929d0e0e4736"
    with TestClient(app) as client:
        response = client.get(
            "/test-default-disabled-trace-context",
            headers={"traceparent": f"00-{incoming_trace_id}-00f067aa0ba902b7-01"},
        )

    assert response.status_code == 200
    assert response.json()["trace_id"] != incoming_trace_id


def test_trace_context_propagation_can_be_disabled(tmp_path: Path) -> None:
    app = create_app(_test_config(tmp_path, accept_incoming_trace_context=False))
    provider = TracerProvider()
    tracer = provider.get_tracer(__name__)

    @app.get("/test-disabled-trace-context")
    async def generate_trace_span() -> dict[str, str]:
        with tracer.start_as_current_span("request_span") as span:
            return {"trace_id": format(span.get_span_context().trace_id, "032x")}

    incoming_trace_id = "4bf92f3577b34da6a3ce929d0e0e4736"
    with TestClient(app) as client:
        response = client.get(
            "/test-disabled-trace-context",
            headers={"traceparent": f"00-{incoming_trace_id}-00f067aa0ba902b7-01"},
        )

    assert response.status_code == 200
    assert response.json()["trace_id"] != incoming_trace_id
