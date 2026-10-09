"""Settings for agent traces: whether the gateway records them, how long they are kept, and the writer's limits."""

from typing import Annotated

from pydantic import BaseModel, Field

from gateway.core.settings_view import OMITTED, SettingsGroup, Shown


class TraceSettings(BaseModel):
    trace_capture_enabled: Annotated[bool, Shown(SettingsGroup.METERING)] = Field(
        default=True,
        description=(
            "Record an agent trace for every completion request: the request, each LLM call with its cost, "
            "routing attempts, guardrail checks, MCP connections and gateway-run tool calls. No prompt, output "
            "or tool content is stored. Requires restart."
        ),
    )
    trace_retention_days: Annotated[int, Shown(SettingsGroup.METERING)] = Field(
        default=30,
        gt=0,
        description="Delete a trace once it has had no activity for this many days.",
    )
    trace_session_max_age_days: Annotated[int, Shown(SettingsGroup.METERING)] = Field(
        default=90,
        gt=0,
        description=(
            "Delete a session's trace this many days after it started, even while it is still active, so a client "
            "that keeps reusing one session id gets a new trace rather than one that never ends."
        ),
    )
    trace_queue_max_spans: Annotated[int, OMITTED] = Field(
        default=10_000,
        gt=0,
        description=(
            "Spans the trace writer holds before it drops new requests' spans. A request's spans are dropped "
            "whole, never in part, and every drop is counted in gateway_trace_spans_dropped."
        ),
    )
    trace_flush_max_spans: Annotated[int, OMITTED] = Field(
        default=500, gt=0, description="Spans the trace writer sends to its store in one batch."
    )
    trace_flush_interval_s: Annotated[float, OMITTED] = Field(
        default=1.0, gt=0, description="Longest a span waits in the trace writer before a batch is sent."
    )
    trace_write_timeout_s: Annotated[float, OMITTED] = Field(
        default=5.0, gt=0, description="Longest one batch write may take; a batch that times out is dropped."
    )
    trace_shutdown_flush_s: Annotated[float, OMITTED] = Field(
        default=5.0, gt=0, description="Longest the trace writer spends flushing at shutdown before it drops the rest."
    )
    trace_max_spans_per_request: Annotated[int, OMITTED] = Field(
        default=256, gt=0, description="Spans one request may record; past it the rest are counted, not kept."
    )
