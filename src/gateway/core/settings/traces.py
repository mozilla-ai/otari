"""Settings for agent traces: whether the gateway records them, how long they are kept, and the writer's limits."""

from typing import Annotated, Literal

from pydantic import BaseModel, Field

from gateway.core.settings_view import OMITTED, SettingsGroup, Shown


class TraceSettings(BaseModel):
    trace_capture_enabled: Annotated[bool, Shown(SettingsGroup.METERING)] = Field(
        default=True,
        description=(
            "Record an agent trace for every completion request: the request, each LLM call with its cost, "
            "routing attempts, guardrail checks, MCP connections and gateway-run tool calls. No prompt, output "
            "or tool content is stored unless trace_content_enabled is on and a workspace keeps it. Requires restart."
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
    trace_content_enabled: Annotated[bool, OMITTED] = Field(
        default=False,
        description=(
            "Keep what an instrumented coding agent reports a span was about: the prompt that opened a turn, the "
            "command a tool ran or the file it touched, and what a guardrail gate checked. Stored encrypted under "
            "the key trace_content_key_backend names, and not kept at all without one; the agent has to send it too "
            "(Claude Code's OTEL_LOG_TOOL_DETAILS and OTEL_LOG_USER_PROMPTS, otari hook's OTARI_HOOK_REPORT_CONTENT). "
            "Off by default."
        ),
    )
    trace_platform_break_glass_enabled: Annotated[bool, OMITTED] = Field(
        default=False,
        description=(
            "Let the deployment's operator open any organization's session content without its owner's share, "
            "giving a reason and a reference (a legal request, a ticket). Every opening is recorded. An "
            "organization's owners can do the same within their organization whatever this says. Off by default."
        ),
    )
    trace_content_key_backend: Annotated[Literal["secret_key", "aws_kms", "aws_kms_only"], OMITTED] = Field(
        default="secret_key",
        description=(
            "What seals trace content: 'secret_key' wraps each data key with OTARI_SECRET_KEY, 'aws_kms' has AWS KMS "
            "make it under trace_content_kms_key_id. Content sealed either way stays readable while that key is "
            "configured. 'aws_kms_only' is 'aws_kms' with OTARI_SECRET_KEY left out: content it sealed reads back "
            "empty, so nothing opens without a KMS call. Requires restart."
        ),
    )
    trace_content_kms_key_id: Annotated[str | None, OMITTED] = Field(
        default=None,
        description=(
            "The AWS KMS key new trace content is sealed under: a key id, key ARN or alias. Required by 'aws_kms'; "
            "set with 'secret_key', it still opens what it sealed."
        ),
    )
    trace_content_kms_previous_key_ids: Annotated[str | None, OMITTED] = Field(
        default=None,
        description=(
            "Earlier AWS KMS keys whose trace content still opens, separated by commas or whitespace. List a "
            "replaced key here for as long as traces are kept."
        ),
    )
    trace_content_kms_region: Annotated[str | None, OMITTED] = Field(
        default=None, description="AWS region of the trace content KMS key; unset uses the AWS SDK's own default."
    )
    trace_content_kms_endpoint_url: Annotated[str | None, OMITTED] = Field(
        default=None, description="An AWS KMS endpoint other than the region's own, such as a VPC endpoint."
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
