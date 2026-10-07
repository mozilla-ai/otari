"""Settings for agent traces: whether the gateway records them, how long they are kept, and the writer's limits."""

from typing import Annotated, Literal, Self

from pydantic import BaseModel, Field, model_validator

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
    trace_content_capture_max: Annotated[Literal["off", "tool_io", "full"], Shown(SettingsGroup.METERING)] = Field(
        default="off",
        description=(
            "The most trace content any workspace may keep: off, tool arguments and results (tool_io), or also "
            "prompts and model output (full). Off by default, so an operator opts the deployment in. Only limits: "
            "content stays off in every workspace until one of its admins turns it on there. Requires restart."
        ),
    )
    trace_content_retention_days: Annotated[int, Shown(SettingsGroup.METERING)] = Field(
        default=7,
        gt=0,
        description="Delete captured trace content, and destroy the keys that sealed it, after this many days.",
    )
    trace_content_key_backend: Annotated[Literal["secret_box", "aws_kms"], Shown(SettingsGroup.METERING)] = Field(
        default="secret_box",
        description=(
            "Where the key that encrypts captured trace content lives: OTARI_SECRET_KEY (secret_box), or an AWS "
            "KMS key (aws_kms, which needs the kms extra and trace_content_kms_key_id). Requires restart."
        ),
    )
    trace_content_kms_key_id: Annotated[str | None, Shown(SettingsGroup.METERING)] = Field(
        default=None,
        description="The AWS KMS key id or ARN the aws_kms key backend generates and decrypts data keys with.",
    )

    @model_validator(mode="after")
    def _kms_backend_names_its_key(self) -> Self:
        if self.trace_content_key_backend == "aws_kms" and not self.trace_content_kms_key_id:
            msg = "trace_content_key_backend 'aws_kms' requires trace_content_kms_key_id"
            raise ValueError(msg)
        # Decrypt is pinned to the key, so it is named by its id or ARN: an alias an operator
        # later repoints would leave every stored session key unable to unwrap.
        if self.trace_content_kms_key_id and ":alias/" in f":{self.trace_content_kms_key_id}":
            msg = "trace_content_kms_key_id must name the key by id or ARN, not by alias"
            raise ValueError(msg)
        return self
