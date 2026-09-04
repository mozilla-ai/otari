"""OTLP receiver for LLM usage telemetry (traces, logs, and metrics).

Any application instrumented for GenAI telemetry can ship its usage to Otari over
OTLP and have it recorded as imported usage: priced at Otari's own rates, budget
exempt, idempotent, and content-free. Two signal endpoints, one mapping:

- ``POST /v1/traces`` accepts spans, the shape most GenAI instrumentation emits
  (OpenLLMetry, OpenLIT, native SDK OpenTelemetry).
- ``POST /v1/logs`` accepts log events, the shape Claude Code emits (its
  ``api_request`` event rides the logs signal, not traces).

A third endpoint, ``POST /v1/metrics``, receives the metrics signal. It carries no
usage: it records the content-free outcome counters (lines changed, commits, pull
requests, active time) that give recorded spend a denominator.

Attribute mapping prefers the OpenTelemetry GenAI semantic conventions
(``gen_ai.provider.name``, ``gen_ai.request.model``, ``gen_ai.usage.*``) with
``otari.*`` for session/client attribution, and special-cases two coding agents
that ride the logs signal with their own attribute names: Claude Code
(``api_request``) and Codex (``codex.api_request`` / ``codex.sse_event``). The
provenance ``source`` is the client (``otari.client_name``, or ``claude_code`` /
``codex`` for the special cases). Only the numeric usage is read; prompts,
responses, and user identity are never persisted.

Claude Code reports cache reads/writes additively (outside ``input_tokens``);
Codex and OpenAI-shaped emitters report cached tokens as a subset of
``input_tokens``. Each mapped event carries ``cache_tokens_in_prompt`` so the
pricing layer de-includes the OpenAI shape instead of double-charging cache reads.

Both protobuf and JSON OTLP payloads are accepted (optionally gzip-encoded), and a
standard OTLP export response is returned. Authentication requires a budget-exempt
API key: OTLP events carry no user attribution of their own, so usage binds to the
key's user. The master key is refused here (it has no user to bind to; use
``POST /v1/usage/external-events`` with an explicit ``user_id`` instead). Point the
exporter's endpoint at the Otari root; the exporter appends the signal path
(``/v1/traces`` or ``/v1/logs``) itself.
"""

import hashlib
import re
import zlib
from collections import defaultdict
from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from google.protobuf.json_format import MessageToJson, Parse, ParseError
from google.protobuf.message import DecodeError, Message
from opentelemetry.proto.collector.logs.v1.logs_service_pb2 import (
    ExportLogsServiceRequest,
    ExportLogsServiceResponse,
)
from opentelemetry.proto.collector.metrics.v1.metrics_service_pb2 import (
    ExportMetricsServiceRequest,
    ExportMetricsServiceResponse,
)
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceRequest,
    ExportTraceServiceResponse,
)
from opentelemetry.proto.metrics.v1.metrics_pb2 import AggregationTemporality
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import TelemetryStoragePortDep, get_config, get_db, verify_api_key_or_master_key
from gateway.core.config import GatewayConfig
from gateway.log_config import logger
from gateway.models.entities import APIKey
from gateway.ports.telemetry_storage_port import TelemetryRecord
from gateway.services.agent_telemetry_service import (
    CUMULATIVE,
    DELTA,
    map_behavioral_event,
    map_metric_point,
)
from gateway.services.agent_telemetry_service import ingest as ingest_telemetry
from gateway.services.external_usage_service import (
    MAX_EVENTS_PER_BATCH,
    ExternalEventsRequest,
    ExternalUsageEvent,
    ingest_external_events,
    reserved_source_reason,
)

router = APIRouter(tags=["otel"])

_JSON = "application/json"
_PROTOBUF = ("application/x-protobuf", "application/octet-stream")
_DEFAULT_SOURCE = "otel"

_OTLP_RESPONSE_DESCRIPTION = (
    "OTLP export response. A partial success still returns HTTP 200; the rejected count "
    "is present only when the gateway rejected part of the export."
)


class _OTLPModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True)


class OTLPTracePartialSuccess(_OTLPModel):
    rejected_spans: str | None = Field(
        default=None,
        alias="rejectedSpans",
        description="Rejected span count, encoded as an int64 string by OTLP JSON.",
    )
    error_message: str | None = Field(default=None, alias="errorMessage")


class OTLPTraceServiceResponse(_OTLPModel):
    partial_success: OTLPTracePartialSuccess | None = Field(default=None, alias="partialSuccess")


class OTLPLogsPartialSuccess(_OTLPModel):
    rejected_log_records: str | None = Field(
        default=None,
        alias="rejectedLogRecords",
        description="Rejected log record count, encoded as an int64 string by OTLP JSON.",
    )
    error_message: str | None = Field(default=None, alias="errorMessage")


class OTLPLogsServiceResponse(_OTLPModel):
    partial_success: OTLPLogsPartialSuccess | None = Field(default=None, alias="partialSuccess")


class OTLPMetricsPartialSuccess(_OTLPModel):
    rejected_data_points: str | None = Field(
        default=None,
        alias="rejectedDataPoints",
        description="Rejected data point count, encoded as an int64 string by OTLP JSON.",
    )
    error_message: str | None = Field(default=None, alias="errorMessage")


class OTLPMetricsServiceResponse(_OTLPModel):
    partial_success: OTLPMetricsPartialSuccess | None = Field(default=None, alias="partialSuccess")


def _otlp_response_docs(response_model: type[BaseModel]) -> dict[int | str, dict[str, Any]]:
    """Add the protobuf representation beside FastAPI's JSON response schema."""
    return {
        200: {
            "content": {
                "application/x-protobuf": {
                    "schema": {"$ref": f"#/components/schemas/{response_model.__name__}"}
                }
            }
        }
    }

# Request bounds. OTLP exporters batch to a few MiB per export; the cap keeps one
# request from expanding without bound in memory (gzip bombs decompress fully
# before parsing) or driving an unbounded number of row inserts. Both limits are
# well above anything a real exporter sends, and a 413 is terminal for OTLP
# clients (4xx responses are not retried), so oversized exports fail loudly
# instead of retry-looping.
_MAX_BODY_BYTES = 8 * 1024 * 1024
_MAX_EVENTS_PER_EXPORT = 10 * MAX_EVENTS_PER_BATCH
# The metrics endpoint's own bound, deliberately independent of the event cap
# above. A metric's attribute cardinality is caller-controlled: one metric name
# fans out into a data point per attribute combination, so the body-size cap
# alone does not bound how many rows an export asks for.
_MAX_METRIC_DATA_POINTS = 10 * MAX_EVENTS_PER_BATCH

# Codex rides the logs signal (like Claude Code) but with its own attribute names.
# Its per-request usage lands in one of two events depending on transport: the
# websocket/Responses path emits ``codex.sse_event`` (final usage on the
# ``response.completed`` kind), the HTTP ``/models`` path emits ``codex.api_request``
# with the ``gen_ai.usage.*`` names. Both carry a bare ``model`` (not
# ``gen_ai.request.model``), so the generic GenAI mapping would skip them.
_CODEX_USAGE_EVENTS = ("codex.api_request", "codex.sse_event")

# Providers that report cache reads/writes *additively* (outside ``input_tokens``),
# the Anthropic shape. Everything else is assumed OpenAI-shaped, where cached
# tokens are a subset of ``input_tokens`` and must be de-included before pricing.
_ADDITIVE_CACHE_PROVIDERS = {"anthropic"}


def _content_type(request: Request) -> str:
    return request.headers.get("content-type", "").split(";")[0].strip().lower()


def _decode_body(body: bytes, content_encoding: str | None) -> bytes:
    if len(body) > _MAX_BODY_BYTES:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, "OTLP payload too large")
    if content_encoding and "gzip" in content_encoding.lower():
        # Streamed with an output cap: gzip.decompress would expand a gzip bomb
        # fully in memory, and a corrupt body must be a client error (4xx), not a
        # 500 the exporter retries forever.
        decompressor = zlib.decompressobj(wbits=16 + zlib.MAX_WBITS)
        try:
            decoded = decompressor.decompress(body, _MAX_BODY_BYTES)
        except zlib.error as exc:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid gzip-encoded OTLP body") from exc
        if decompressor.unconsumed_tail:
            raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, "OTLP payload too large")
        if not decompressor.eof:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "Truncated gzip-encoded OTLP body")
        return decoded
    return body


def _parse(body: bytes, content_type: str, message: Message) -> Message:
    """Parse an OTLP request body (protobuf or JSON) into ``message``."""
    if content_type in _PROTOBUF:
        try:
            message.ParseFromString(body)
        except DecodeError as exc:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid OTLP protobuf payload") from exc
        return message
    if content_type == _JSON:
        try:
            Parse(body.decode("utf-8"), message, ignore_unknown_fields=True)
        except (ParseError, UnicodeDecodeError, ValueError) as exc:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid OTLP/JSON payload") from exc
        return message
    raise HTTPException(
        status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
        "Unsupported content-type; send OTLP as application/json or application/x-protobuf.",
    )


def _any_value(value: Any) -> Any:
    """Unwrap an OTLP AnyValue protobuf into a Python scalar (or None)."""
    kind = value.WhichOneof("value")
    if kind == "string_value":
        return value.string_value
    if kind == "int_value":
        return value.int_value
    if kind == "double_value":
        return value.double_value
    if kind == "bool_value":
        return value.bool_value
    return None


def _attributes(kv_list: Any) -> dict[str, Any]:
    return {kv.key: _any_value(kv.value) for kv in kv_list}


def _int(value: Any) -> int:
    try:
        return max(int(value), 0)
    except (TypeError, ValueError):
        return 0


def _nanos_to_dt(nanos: int) -> datetime | None:
    if not nanos:
        return None
    try:
        return datetime.fromtimestamp(int(nanos) / 1_000_000_000, tz=UTC)
    except (TypeError, ValueError, OSError):
        return None


def _resolve_timestamp(attrs: dict[str, Any], default: datetime | None) -> datetime | None:
    # Claude Code carries an explicit event.timestamp; otherwise use the span/record time.
    ts = attrs.get("event.timestamp")
    if isinstance(ts, str):
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except ValueError:
            pass
    return default


def _sanitize_source(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._:-]+", "-", name).strip("-")[:64]
    # "gateway" is reserved for usage Otari served itself and the `otari-ai:` prefix for
    # the provenance tags otari.ai writes; a client claiming either would masquerade as
    # those rows (and fail the ingest schema with a 500).
    if not slug or reserved_source_reason(slug) is not None:
        return _DEFAULT_SOURCE
    return slug


def _codex_event_id(attrs: dict[str, Any], explicit: Any) -> str:
    """A stable idempotency key for a Codex usage event.

    Codex's websocket/Responses event has no per-request id (only a
    ``conversation.id`` shared across turns), so derive a deterministic key from
    the conversation, the event timestamp, and the token counts. Replays hash to
    the same value; distinct turns differ by timestamp. When an explicit id is
    present (e.g. ``gen_ai.response.id`` on the HTTP path) it is used verbatim.
    """
    if explicit:
        return str(explicit)
    seed = "|".join(
        str(attrs.get(k, ""))
        for k in (
            "conversation.id",
            "event.timestamp",
            "input_token_count",
            "output_token_count",
            "gen_ai.usage.input_tokens",
            "gen_ai.usage.output_tokens",
        )
    )
    return "codex-" + hashlib.sha256(seed.encode()).hexdigest()[:32]


def _build_event(
    attrs: dict[str, Any],
    *,
    fallback_event_id: str | None,
    default_timestamp: datetime | None,
    duration_ms: float | None,
) -> tuple[str, ExternalUsageEvent] | None:
    """Map span/log attributes onto a (source, event), or None to skip.

    Special-cases Claude Code and Codex (which use their own attribute names on the
    logs signal), otherwise reads the GenAI semantic conventions. Records with no
    LLM usage (no model/provider/tokens) are skipped, so non-LLM spans and other
    log events are ignored.
    """
    provider: Any
    duration: Any
    # No OTLP shape carries the 5m/1h cache-write split, so every cache write stays in
    # the base (5m) bucket. Claude Code writes with both TTLs (measured on real session
    # transcripts, roughly a quarter to a third of creation tokens are 5m and the rest
    # 1h), so either default is wrong for the other share: 5m undercharges the 1h
    # tokens, 1h overcharges the 5m ones. Anthropic prices a 1h write at 2x base input
    # against 1.25x for a 5m one, so booking the base bucket keeps an imported estimate
    # from ever reading above true cost. Callers holding the real split use
    # POST /v1/usage/external-events, which takes cache_write_1h_tokens directly.
    # No branch below reassigns this today: it stays a variable rather than a literal at
    # the construction site as the seam for an emitter that does report the split.
    cache_write_1h = 0
    event_name = attrs.get("event.name")
    if event_name == "api_request":
        # Claude Code: cache reads/writes are additive (outside input_tokens).
        source = "claude_code"
        provider = "anthropic"
        model = attrs.get("model")
        event_id = attrs.get("request_id") or fallback_event_id
        input_tokens = _int(attrs.get("input_tokens"))
        output_tokens = _int(attrs.get("output_tokens"))
        cache_read = _int(attrs.get("cache_read_tokens"))
        cache_write = _int(attrs.get("cache_creation_tokens"))
        session = attrs.get("session.id")
        duration = attrs.get("duration_ms", duration_ms)
        cache_tokens_in_prompt = False
    elif event_name in _CODEX_USAGE_EVENTS:
        # Codex: bare model/provider_name, OpenAI-shaped token counts (cached is a
        # subset of input). Only the Responses ``response.completed`` event carries
        # final usage; other sse_event kinds (deltas, rate limits) are skipped.
        if event_name == "codex.sse_event" and attrs.get("event.kind") != "response.completed":
            return None
        source = "codex"
        provider = attrs.get("provider_name") or "openai"
        model = attrs.get("model")
        if event_name == "codex.sse_event":
            input_tokens = _int(attrs.get("input_token_count"))
            output_tokens = _int(attrs.get("output_token_count"))
            cache_read = _int(attrs.get("cached_token_count"))
            event_id = _codex_event_id(attrs, None)
            duration = duration_ms
        else:  # codex.api_request (HTTP /models path)
            input_tokens = _int(attrs.get("gen_ai.usage.input_tokens"))
            output_tokens = _int(attrs.get("gen_ai.usage.output_tokens"))
            cache_read = _int(attrs.get("gen_ai.usage.cache_read.input_tokens"))
            event_id = _codex_event_id(attrs, attrs.get("gen_ai.response.id"))
            duration = attrs.get("duration_ms", duration_ms)
        cache_write = 0
        session = attrs.get("conversation.id")
        cache_tokens_in_prompt = True
    else:
        provider = attrs.get("gen_ai.provider.name") or attrs.get("gen_ai.system")
        model = attrs.get("gen_ai.request.model") or attrs.get("gen_ai.response.model")
        event_id = attrs.get("gen_ai.response.id") or fallback_event_id
        input_tokens = _int(attrs.get("gen_ai.usage.input_tokens"))
        output_tokens = _int(attrs.get("gen_ai.usage.output_tokens"))
        cache_read = _int(
            attrs.get("gen_ai.usage.cache_read_tokens")
            or attrs.get("gen_ai.usage.cache_read.input_tokens")
            or attrs.get("gen_ai.usage.cached_tokens")
        )
        cache_write = _int(attrs.get("gen_ai.usage.cache_write_tokens"))
        client = attrs.get("otari.client_name")
        source = _sanitize_source(str(client)) if client else _DEFAULT_SOURCE
        session = attrs.get("otari.user_session_label") or attrs.get("otari.session_label")
        duration = duration_ms
        cache_tokens_in_prompt = str(provider).strip().lower() not in _ADDITIVE_CACHE_PROVIDERS

    timestamp = _resolve_timestamp(attrs, default_timestamp)
    if not (event_id and model and provider and timestamp):
        return None
    if not (input_tokens or output_tokens or cache_read or cache_write):
        return None  # not an LLM call worth recording
    try:
        event = ExternalUsageEvent(
            source_event_id=str(event_id),
            timestamp=timestamp,
            provider=str(provider),
            model=str(model),
            status="success",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_write,
            cache_write_1h_tokens=cache_write_1h,
            cache_tokens_in_prompt=cache_tokens_in_prompt,
            duration_ms=int(duration) if duration is not None else None,
            session_label=str(session) if session else None,
        )
    except Exception:  # noqa: BLE001 - a value that fails the event schema is skipped, not fatal
        return None
    return source, event


def _require_import_key(api_key: APIKey | None) -> APIKey:
    """Reject master-key OTLP exports before any parsing work.

    OTLP payloads carry no user attribution, so ingestion binds usage to the API
    key's user. The master key has no user: accepting it would 200 while every
    event is rejected downstream, and the exporter would silently drop the data.
    """
    if api_key is None:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            "OTLP export cannot authenticate with the master key: OTLP events carry no user to "
            "attribute usage to. Use a budget-exempt API key for the importing user, or "
            "POST /v1/usage/external-events with an explicit user_id.",
        )
    return api_key


def _capture_telemetry(api_key: APIKey, config: GatewayConfig) -> bool:
    """Whether content-free agent telemetry is captured for this key.

    One toggle governs one table: the per-key override wins, otherwise the
    deployment setting. Both the behavioral events on the logs signal and the
    outcome metrics on the metrics signal answer to it.
    """
    if api_key.capture_agent_telemetry is not None:
        return api_key.capture_agent_telemetry
    return config.capture_agent_telemetry


def _point_value(point: Any) -> Any:
    """Read an OTLP NumberDataPoint's value, whichever arm it arrived on.

    None when neither arm is set: proto3 would report the unset ``as_int`` as 0,
    and a spurious 0 inside a cumulative series reads as a reset the series start
    never announced, clamping one diff and re-counting the whole total on the next.
    """
    arm = point.WhichOneof("value")
    if arm is None:
        return None
    return point.as_double if arm == "as_double" else point.as_int


async def _ingest(
    pairs: list[tuple[str, ExternalUsageEvent]],
    *,
    api_key: APIKey,
    db: AsyncSession,
    config: GatewayConfig,
) -> int:
    """Group events by their derived source and ingest each group (chunked).

    Returns the number of rejected events, for the OTLP partial-success response.
    """
    if len(pairs) > _MAX_EVENTS_PER_EXPORT:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, "Too many usage events in one OTLP export")
    rejected = 0
    by_source: dict[str, list[ExternalUsageEvent]] = defaultdict(list)
    for source, event in pairs:
        by_source[source].append(event)
    for source, events in by_source.items():
        for start in range(0, len(events), MAX_EVENTS_PER_BATCH):
            result = await ingest_external_events(
                db,
                ExternalEventsRequest(source=source, events=events[start : start + MAX_EVENTS_PER_BATCH]),
                api_key=api_key,
                is_master_key=False,
                reject_user_mismatch=config.reject_user_mismatch,
            )
            rejected += result.rejected
            logger.info(
                "otlp ingest: source=%s accepted=%d duplicate=%d rejected=%d",
                source,
                result.accepted,
                result.duplicate,
                result.rejected,
            )
    return rejected


def _otlp_response(content_type: str, response: Message) -> Response:
    if content_type == _JSON:
        return Response(content=MessageToJson(response), media_type=_JSON)
    return Response(content=response.SerializeToString(), media_type="application/x-protobuf")


@router.post(
    "/v1/traces",
    response_model=OTLPTraceServiceResponse,
    response_description=_OTLP_RESPONSE_DESCRIPTION,
    responses=_otlp_response_docs(OTLPTraceServiceResponse),
)
async def receive_traces(
    request: Request,
    auth_result: Annotated[tuple[APIKey | None, bool], Depends(verify_api_key_or_master_key)],
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> Response:
    """Ingest LLM usage from OTLP spans (GenAI semantic conventions)."""
    api_key = _require_import_key(auth_result[0])
    content_type = _content_type(request)
    body = _decode_body(await request.body(), request.headers.get("content-encoding"))
    parsed = _parse(body, content_type, ExportTraceServiceRequest())
    assert isinstance(parsed, ExportTraceServiceRequest)

    pairs: list[tuple[str, ExternalUsageEvent]] = []
    for resource_spans in parsed.resource_spans:
        for scope_spans in resource_spans.scope_spans:
            for span in scope_spans.spans:
                duration = None
                if span.start_time_unix_nano and span.end_time_unix_nano:
                    duration = (span.end_time_unix_nano - span.start_time_unix_nano) / 1_000_000
                mapped = _build_event(
                    _attributes(span.attributes),
                    fallback_event_id=span.span_id.hex() or None,
                    default_timestamp=_nanos_to_dt(span.start_time_unix_nano),
                    duration_ms=duration,
                )
                if mapped is not None:
                    pairs.append(mapped)

    response = ExportTraceServiceResponse()
    if pairs:
        rejected = await _ingest(pairs, api_key=api_key, db=db, config=config)
        if rejected:
            response.partial_success.rejected_spans = rejected
            response.partial_success.error_message = f"{rejected} usage event(s) rejected (see gateway logs)"
    return _otlp_response(content_type, response)


@router.post(
    "/v1/logs",
    response_model=OTLPLogsServiceResponse,
    response_description=_OTLP_RESPONSE_DESCRIPTION,
    responses=_otlp_response_docs(OTLPLogsServiceResponse),
)
async def receive_logs(
    request: Request,
    auth_result: Annotated[tuple[APIKey | None, bool], Depends(verify_api_key_or_master_key)],
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    storage: TelemetryStoragePortDep,
) -> Response:
    """Ingest LLM usage from OTLP log events (Claude Code, Codex, or GenAI logs)."""
    api_key = _require_import_key(auth_result[0])
    content_type = _content_type(request)
    body = _decode_body(await request.body(), request.headers.get("content-encoding"))
    parsed = _parse(body, content_type, ExportLogsServiceRequest())
    assert isinstance(parsed, ExportLogsServiceRequest)

    capture_telemetry = _capture_telemetry(api_key, config)
    pairs: list[tuple[str, ExternalUsageEvent]] = []
    telemetry: list[TelemetryRecord] = []
    for resource_logs in parsed.resource_logs:
        for scope_logs in resource_logs.scope_logs:
            for record in scope_logs.log_records:
                attrs = _attributes(record.attributes)
                timestamp = _nanos_to_dt(record.time_unix_nano)
                mapped = _build_event(
                    attrs,
                    fallback_event_id=None,
                    default_timestamp=timestamp,
                    duration_ms=None,
                )
                if mapped is not None:
                    pairs.append(mapped)
                if timestamp is not None and capture_telemetry:
                    behavioral = map_behavioral_event(
                        str(attrs.get("event.name", "")),
                        attrs,
                        timestamp=timestamp,
                        source="claude_code",
                        user_id=api_key.user_id,
                    )
                    if behavioral is not None:
                        telemetry.append(behavioral)

    # Usage and behavioral events insert a row each and are disjoint by event name,
    # so the bound is on their total: checked per-list, one export could persist
    # twice _MAX_EVENTS_PER_EXPORT rows. The narrower check inside _ingest still
    # covers /v1/traces, which has no behavioral path.
    if len(pairs) + len(telemetry) > _MAX_EVENTS_PER_EXPORT:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, "Too many events in one OTLP export")

    response = ExportLogsServiceResponse()
    rejected = await _ingest(pairs, api_key=api_key, db=db, config=config) if pairs else 0
    if telemetry:
        rejected += (await ingest_telemetry(db, telemetry, api_key=api_key, storage=storage)).rejected
    if rejected:
        response.partial_success.rejected_log_records = rejected
        response.partial_success.error_message = f"{rejected} log record(s) rejected (see gateway logs)"
    return _otlp_response(content_type, response)


@router.post(
    "/v1/metrics",
    response_model=OTLPMetricsServiceResponse,
    response_description=_OTLP_RESPONSE_DESCRIPTION,
    responses=_otlp_response_docs(OTLPMetricsServiceResponse),
)
async def receive_metrics(
    request: Request,
    auth_result: Annotated[tuple[APIKey | None, bool], Depends(verify_api_key_or_master_key)],
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    storage: TelemetryStoragePortDep,
) -> Response:
    """Ingest content-free coding-agent outcome metrics from OTLP metric points.

    Records the outcome counters a coding agent reports on the metrics signal and
    that Otari has no other source for: lines of code changed, commits, pull
    requests, and active time. Points are stored exactly as reported, with their
    OTLP series identity, so a cumulative counter is turned into an increment at
    read time rather than re-counted on every export. Metrics that duplicate an
    already-recorded signal (token/cost usage, already billed; edit decisions,
    already captured as behavioral events) are skipped, as is any metric name this
    gateway does not know, so a newer agent version never breaks reception.

    Outcome metrics are never billable: they touch no budget and no spend. Capture
    answers to the same ``capture_agent_telemetry`` toggle as behavioral events;
    with it off, the export still succeeds and simply stores nothing.
    """
    api_key = _require_import_key(auth_result[0])
    content_type = _content_type(request)
    body = _decode_body(await request.body(), request.headers.get("content-encoding"))
    parsed = _parse(body, content_type, ExportMetricsServiceRequest())
    assert isinstance(parsed, ExportMetricsServiceRequest)

    capture_telemetry = _capture_telemetry(api_key, config)
    telemetry: list[TelemetryRecord] = []
    seen_points = 0
    for resource_metrics in parsed.resource_metrics:
        for scope_metrics in resource_metrics.scope_metrics:
            for metric in scope_metrics.metrics:
                data = metric.WhichOneof("data")
                if data == "sum":
                    points = metric.sum.data_points
                    # A gauge is a level, not a running total, so its points are
                    # increments; an unspecified temporality is read the same way,
                    # since summing it as cumulative would invent growth.
                    temporality = (
                        CUMULATIVE
                        if metric.sum.aggregation_temporality
                        == AggregationTemporality.AGGREGATION_TEMPORALITY_CUMULATIVE
                        else DELTA
                    )
                elif data == "gauge":
                    points = metric.gauge.data_points
                    temporality = DELTA
                else:
                    continue  # histograms and summaries carry no outcome counter
                seen_points += len(points)
                if seen_points > _MAX_METRIC_DATA_POINTS:
                    raise HTTPException(
                        status.HTTP_413_CONTENT_TOO_LARGE, "Too many metric data points in one OTLP export"
                    )
                if not capture_telemetry:
                    continue
                for point in points:
                    timestamp = _nanos_to_dt(point.time_unix_nano)
                    if timestamp is None:
                        continue  # a point with no time cannot be ordered or deduped
                    mapped = map_metric_point(
                        metric.name,
                        _point_value(point),
                        temporality,
                        _nanos_to_dt(point.start_time_unix_nano),
                        _attributes(point.attributes),
                        timestamp=timestamp,
                        source="claude_code",
                        user_id=api_key.user_id,
                    )
                    if mapped is not None:
                        telemetry.append(mapped)

    response = ExportMetricsServiceResponse()
    if telemetry:
        result = await ingest_telemetry(db, telemetry, api_key=api_key, storage=storage)
        logger.info(
            "otlp metrics ingest: accepted=%d duplicate=%d rejected=%d",
            result.accepted,
            result.duplicate,
            result.rejected,
        )
        if result.rejected:
            response.partial_success.rejected_data_points = result.rejected
            response.partial_success.error_message = (
                f"{result.rejected} metric data point(s) rejected (see gateway logs)"
            )
    return _otlp_response(content_type, response)
