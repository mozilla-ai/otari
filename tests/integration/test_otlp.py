"""Integration tests for the OTLP receiver (POST /v1/logs and /v1/traces).

Coding agents export OpenTelemetry to Otari; the receiver maps their usage onto the
external-usage ingestion path. Claude Code posts ``api_request`` log events, Codex
posts ``codex.sse_event`` / ``codex.api_request`` log events (both with OpenAI-shaped
token counts, verified against a real captured Codex export), and generically
instrumented apps post GenAI-convention spans. The Codex/GenAI attribute names and
token semantics here mirror payloads captured from the real clients.
"""

import gzip
import json
from datetime import UTC, datetime
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from gateway.models.entities import UsageLog, User

_PATH = "/v1/logs"


def _attr(key: str, value: Any) -> dict[str, Any]:
    if isinstance(value, bool):
        return {"key": key, "value": {"boolValue": value}}
    if isinstance(value, int):
        return {"key": key, "value": {"intValue": str(value)}}  # OTLP/JSON encodes int64 as string
    return {"key": key, "value": {"stringValue": str(value)}}


def _api_request_record(source_event_id: str = "req_otlp_1", **over: Any) -> dict[str, Any]:
    attrs = {
        "event.name": "api_request",
        "event.timestamp": "2026-07-22T12:00:00.000Z",
        "model": "claude-opus-4-8",
        "input_tokens": 1200,
        "output_tokens": 450,
        "cache_read_tokens": 8000,
        "cache_creation_tokens": 1024,
        "duration_ms": 1198,
        "request_id": source_event_id,
        "session.id": "2c92f0a0-773f-41f2-8703-1ab0a6a2065b",
        "user.email": "nathan@example.com",  # present but must never be persisted
        **over,
    }
    return {"timeUnixNano": "1784000000000000000", "body": {"stringValue": "api_request"}, "attributes": [
        _attr(k, v) for k, v in attrs.items()
    ]}


def _otlp(*records: dict[str, Any]) -> dict[str, Any]:
    return {"resourceLogs": [{"scopeLogs": [{"logRecords": list(records)}]}]}


def _exempt_key(client: TestClient, master_key_header: dict[str, str], user_id: str = "alice") -> dict[str, str]:
    client.post("/v1/users", json={"user_id": user_id}, headers=master_key_header)
    resp = client.post(
        "/v1/keys",
        json={"key_name": "cc-otlp", "user_id": user_id, "exclude_from_budget": True},
        headers=master_key_header,
    )
    assert resp.status_code == 200, resp.text
    return {"Otari-Key": f"Bearer {resp.json()['key']}"}


def _seed_pricing(client: TestClient, master_key_header: dict[str, str]) -> None:
    client.post(
        "/v1/pricing",
        json={
            "model_key": "anthropic:claude-opus-4-8",
            "input_price_per_million": 15.0,
            "output_price_per_million": 75.0,
            "cache_read_price_per_million": 1.5,
            "cache_write_price_per_million": 18.75,
            "effective_at": "2020-01-01T00:00:00Z",
        },
        headers=master_key_header,
    )


def test_otlp_api_request_is_ingested_and_priced(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    headers = _exempt_key(client, master_key_header)
    _seed_pricing(client, master_key_header)

    resp = client.post(_PATH, json=_otlp(_api_request_record()), headers=headers)
    assert resp.status_code == 200, resp.text

    row = db_session.query(UsageLog).filter(UsageLog.source_event_id == "req_otlp_1").one()
    assert row.source == "claude_code"
    assert row.user_id == "alice"
    assert row.api_key_id is not None
    assert row.counts_toward_budget is False
    assert row.model == "claude-opus-4-8"
    assert row.prompt_tokens == 1200 and row.completion_tokens == 450 and row.cache_read_tokens == 8000
    # Claude Code uses 1h caching, and the event has no split, so cache creation is
    # booked entirely as 1h (priced at the 1h rate, not the 5m base).
    assert row.cache_write_tokens == 1024 and row.cache_write_1h_tokens == 1024
    # Otari prices at its own configured rate (Claude Code's cost_usd is ignored).
    assert row.cost is not None and row.cost > 0
    # The email attribute present in the OTLP record must never be persisted.
    assert "example.com" not in (row.source_label or "")

    user = db_session.query(User).filter(User.user_id == "alice").one()
    assert float(user.spend) == 0.0  # imports never touch spend


def test_otlp_ignores_non_api_request_records(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    """Prompts, tool results, and metrics carry no billable usage and are skipped."""
    headers = _exempt_key(client, master_key_header)
    prompt_record = {
        "body": {"stringValue": "user_prompt"},
        "attributes": [_attr("event.name", "user_prompt"), _attr("prompt", "secret user text")],
    }
    resp = client.post(_PATH, json=_otlp(prompt_record), headers=headers)
    assert resp.status_code == 200
    assert db_session.query(UsageLog).filter(UsageLog.source == "claude_code").count() == 0


def test_otlp_idempotent(client: TestClient, master_key_header: dict[str, str], db_session: Session) -> None:
    headers = _exempt_key(client, master_key_header)
    _seed_pricing(client, master_key_header)
    client.post(_PATH, json=_otlp(_api_request_record("req_dup")), headers=headers)
    client.post(_PATH, json=_otlp(_api_request_record("req_dup")), headers=headers)
    assert db_session.query(UsageLog).filter(UsageLog.source_event_id == "req_dup").count() == 1


def test_otlp_requires_exempt_key(client: TestClient, master_key_header: dict[str, str]) -> None:
    client.post("/v1/users", json={"user_id": "bob"}, headers=master_key_header)
    key = client.post(
        "/v1/keys",
        json={"key_name": "budgeted", "user_id": "bob", "exclude_from_budget": False},
        headers=master_key_header,
    ).json()["key"]
    resp = client.post(_PATH, json=_otlp(_api_request_record()), headers={"Otari-Key": f"Bearer {key}"})
    assert resp.status_code == 403


def test_otlp_rejects_garbage_protobuf(client: TestClient, master_key_header: dict[str, str]) -> None:
    headers = {**_exempt_key(client, master_key_header), "Content-Type": "application/x-protobuf"}
    resp = client.post(_PATH, content=b"\xff\xfe\x01", headers=headers)
    assert resp.status_code == 400


def test_otlp_rejects_unknown_content_type(client: TestClient, master_key_header: dict[str, str]) -> None:
    headers = {**_exempt_key(client, master_key_header), "Content-Type": "text/plain"}
    resp = client.post(_PATH, content=b"hi", headers=headers)
    assert resp.status_code == 415


def _span_record(**over: Any) -> dict[str, Any]:
    """An OTLP/JSON span carrying GenAI semantic-convention attributes."""
    attrs = {
        "gen_ai.provider.name": "openai",
        "gen_ai.request.model": "gpt-4o",
        "gen_ai.usage.input_tokens": 1000,
        "gen_ai.usage.output_tokens": 50,
        "gen_ai.usage.cache_read_tokens": 200,
        "gen_ai.response.id": "resp_gen_ai_1",
        "otari.client_name": "my-app",
        "otari.session_label": "proj-x",
        **over,
    }
    return {"startTimeUnixNano": "1784000000000000000", "attributes": [_attr(k, v) for k, v in attrs.items()]}


def _otlp_traces(*spans: dict[str, Any]) -> dict[str, Any]:
    return {"resourceSpans": [{"scopeSpans": [{"spans": list(spans)}]}]}


def test_otlp_traces_gen_ai_is_ingested(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    """A span using the GenAI conventions is ingested from any instrumented app.

    The provider is openai, whose ``gen_ai.usage.input_tokens`` is inclusive of the
    cached slice, so the receiver marks it cache-in-prompt and the price de-includes
    the 200 cached tokens rather than billing them twice.
    """
    headers = _exempt_key(client, master_key_header)
    client.post(
        "/v1/pricing",
        json={
            "model_key": "openai:gpt-4o",
            "input_price_per_million": 2.5,
            "output_price_per_million": 10.0,
            "cache_read_price_per_million": 1.25,
            "effective_at": "2020-01-01T00:00:00Z",
        },
        headers=master_key_header,
    )
    resp = client.post("/v1/traces", json=_otlp_traces(_span_record()), headers=headers)
    assert resp.status_code == 200, resp.text

    row = db_session.query(UsageLog).filter(UsageLog.source_event_id == "resp_gen_ai_1").one()
    assert row.source == "my-app"  # provenance from otari.client_name
    assert row.provider == "openai" and row.model == "gpt-4o"
    # Stored token counts are the raw reported values (prompt inclusive of cached).
    assert row.prompt_tokens == 1000 and row.completion_tokens == 50 and row.cache_read_tokens == 200
    assert row.source_label == "proj-x"
    # De-included price: (1000 - 200) input + 200 cache-read + 50 output.
    expected = (800 * 2.5 + 200 * 1.25 + 50 * 10.0) / 1_000_000
    assert row.cost == pytest.approx(expected)
    assert row.counts_toward_budget is False


# --- Codex ----------------------------------------------------------------------
# Attribute names and token semantics below mirror a real Codex OTLP export driven
# live (gpt-4o-mini via the Responses/websocket path). Codex rides the logs signal
# with bare `model`/`provider_name` and OpenAI-shaped counts (cached is a subset of
# input), so it is a special case distinct from Claude Code and the GenAI path.


def _codex_sse_record(conv: str = "conv-abc", ts: str = "2026-07-23T20:04:22.609Z", **over: Any) -> dict[str, Any]:
    """A Codex `codex.sse_event` (response.completed) log record."""
    attrs = {
        "event.name": "codex.sse_event",
        "event.kind": "response.completed",
        "event.timestamp": ts,
        "model": "gpt-4o-mini",
        "conversation.id": conv,
        "input_token_count": 1000,  # OpenAI prompt tokens, inclusive of cached
        "output_token_count": 100,
        "cached_token_count": 200,
        "reasoning_token_count": 0,
        "auth_mode": "ApiKey",
        "originator": "codex_exec",
        **over,
    }
    return {"timeUnixNano": "1784000000000000000", "attributes": [_attr(k, v) for k, v in attrs.items()]}


def _seed_codex_pricing(client: TestClient, master_key_header: dict[str, str]) -> None:
    client.post(
        "/v1/pricing",
        json={
            "model_key": "openai:gpt-4o-mini",
            "input_price_per_million": 0.15,
            "output_price_per_million": 0.6,
            "cache_read_price_per_million": 0.075,
            "effective_at": "2020-01-01T00:00:00Z",
        },
        headers=master_key_header,
    )


def test_otlp_codex_sse_event_is_ingested_and_priced(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    headers = _exempt_key(client, master_key_header)
    _seed_codex_pricing(client, master_key_header)

    resp = client.post(_PATH, json=_otlp(_codex_sse_record()), headers=headers)
    assert resp.status_code == 200, resp.text

    rows = db_session.query(UsageLog).filter(UsageLog.source == "codex").all()
    assert len(rows) == 1
    row = rows[0]
    assert row.provider == "openai" and row.model == "gpt-4o-mini"
    assert row.user_id == "alice" and row.api_key_id is not None
    assert row.counts_toward_budget is False
    assert row.source_label == "conv-abc"  # conversation.id
    # Raw counts stored as reported; input stays inclusive of the cached slice.
    assert row.prompt_tokens == 1000 and row.completion_tokens == 100 and row.cache_read_tokens == 200
    # De-included price: cached 200 billed once at the cache-read rate, not twice.
    expected = (800 * 0.15 + 200 * 0.075 + 100 * 0.6) / 1_000_000
    assert row.cost == pytest.approx(expected)


def test_otlp_codex_non_completed_sse_event_skipped(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    """Only the response.completed sse_event carries final usage; deltas are skipped."""
    headers = _exempt_key(client, master_key_header)
    record = _codex_sse_record(**{"event.kind": "response.output_text.delta"})
    resp = client.post(_PATH, json=_otlp(record), headers=headers)
    assert resp.status_code == 200
    assert db_session.query(UsageLog).filter(UsageLog.source == "codex").count() == 0


def test_otlp_codex_sse_event_idempotent(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    """The synthesized (conversation + timestamp + tokens) key dedups replays."""
    headers = _exempt_key(client, master_key_header)
    _seed_codex_pricing(client, master_key_header)
    client.post(_PATH, json=_otlp(_codex_sse_record()), headers=headers)
    client.post(_PATH, json=_otlp(_codex_sse_record()), headers=headers)
    assert db_session.query(UsageLog).filter(UsageLog.source == "codex").count() == 1
    # A distinct turn (different timestamp) is a distinct row.
    client.post(_PATH, json=_otlp(_codex_sse_record(ts="2026-07-23T20:05:00.000Z")), headers=headers)
    assert db_session.query(UsageLog).filter(UsageLog.source == "codex").count() == 2


def test_otlp_codex_api_request_path(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    """The HTTP /models path emits codex.api_request with gen_ai.usage.* names."""
    headers = _exempt_key(client, master_key_header)
    _seed_codex_pricing(client, master_key_header)
    record = {
        "attributes": [
            _attr(k, v)
            for k, v in {
                "event.name": "codex.api_request",
                "event.timestamp": "2026-07-23T20:04:22.609Z",
                "model": "gpt-4o-mini",
                "provider_name": "openai",
                "conversation.id": "conv-http",
                "gen_ai.response.id": "resp_codex_http_1",
                "gen_ai.usage.input_tokens": 1000,
                "gen_ai.usage.output_tokens": 100,
                "gen_ai.usage.cache_read.input_tokens": 200,
            }.items()
        ]
    }
    resp = client.post(_PATH, json=_otlp(record), headers=headers)
    assert resp.status_code == 200, resp.text
    row = db_session.query(UsageLog).filter(UsageLog.source_event_id == "resp_codex_http_1").one()
    assert row.source == "codex" and row.provider == "openai" and row.model == "gpt-4o-mini"
    assert row.cache_read_tokens == 200
    expected = (800 * 0.15 + 200 * 0.075 + 100 * 0.6) / 1_000_000
    assert row.cost == pytest.approx(expected)


def test_otlp_codex_protobuf_roundtrip(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    """A binary protobuf Codex logs export (the real wire format) is accepted."""
    from opentelemetry.proto.collector.logs.v1.logs_service_pb2 import ExportLogsServiceRequest
    from opentelemetry.proto.common.v1.common_pb2 import AnyValue, KeyValue

    headers = {**_exempt_key(client, master_key_header), "Content-Type": "application/x-protobuf"}
    _seed_codex_pricing(client, master_key_header)
    req = ExportLogsServiceRequest()
    record = req.resource_logs.add().scope_logs.add().log_records.add()

    def kv(key: str, *, s: str | None = None, i: int | None = None) -> KeyValue:
        value = AnyValue(string_value=s) if s is not None else AnyValue(int_value=i or 0)
        return KeyValue(key=key, value=value)

    record.attributes.extend([
        kv("event.name", s="codex.sse_event"),
        kv("event.kind", s="response.completed"),
        kv("event.timestamp", s="2026-07-23T20:04:22.609Z"),
        kv("model", s="gpt-4o-mini"),
        kv("conversation.id", s="conv-pb"),
        kv("input_token_count", i=1000),
        kv("output_token_count", i=100),
        kv("cached_token_count", i=200),
    ])
    resp = client.post(_PATH, content=req.SerializeToString(), headers=headers)
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"].startswith("application/x-protobuf")
    row = db_session.query(UsageLog).filter(UsageLog.source == "codex", UsageLog.source_label == "conv-pb").one()
    assert row.prompt_tokens == 1000 and row.cache_read_tokens == 200


def test_otlp_traces_protobuf_roundtrip(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    """A binary protobuf traces export is accepted, not just JSON."""
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
    from opentelemetry.proto.common.v1.common_pb2 import AnyValue, KeyValue

    headers = {**_exempt_key(client, master_key_header), "Content-Type": "application/x-protobuf"}
    req = ExportTraceServiceRequest()
    span = req.resource_spans.add().scope_spans.add().spans.add()
    span.start_time_unix_nano = 1784000000000000000

    def kv(key: str, *, s: str | None = None, i: int | None = None) -> KeyValue:
        value = AnyValue(string_value=s) if s is not None else AnyValue(int_value=i or 0)
        return KeyValue(key=key, value=value)

    span.attributes.extend([
        kv("gen_ai.provider.name", s="anthropic"),
        kv("gen_ai.request.model", s="claude-sonnet-4-6"),
        kv("gen_ai.response.id", s="resp_pb_1"),
        kv("gen_ai.usage.input_tokens", i=10),
        kv("gen_ai.usage.output_tokens", i=20),
    ])
    resp = client.post("/v1/traces", content=req.SerializeToString(), headers=headers)
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"].startswith("application/x-protobuf")
    assert db_session.query(UsageLog).filter(UsageLog.source_event_id == "resp_pb_1").count() == 1


def test_otlp_master_key_is_refused(client: TestClient, master_key_header: dict[str, str]) -> None:
    """OTLP events carry no user attribution, so the master key has no one to bill."""
    resp = client.post(_PATH, json=_otlp(_api_request_record()), headers=master_key_header)
    assert resp.status_code == 403
    assert "budget-exempt API key" in resp.json()["detail"]


def test_otlp_gzip_payload_is_accepted(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    headers = _exempt_key(client, master_key_header)
    body = gzip.compress(json.dumps(_otlp(_api_request_record("req_gzip_1"))).encode())
    resp = client.post(
        _PATH,
        content=body,
        headers={**headers, "Content-Type": "application/json", "Content-Encoding": "gzip"},
    )
    assert resp.status_code == 200, resp.text
    assert db_session.query(UsageLog).filter(UsageLog.source_event_id == "req_gzip_1").count() == 1


def test_otlp_corrupt_gzip_is_a_client_error(client: TestClient, master_key_header: dict[str, str]) -> None:
    """A broken body must be a 4xx: OTLP exporters retry 5xx responses forever."""
    headers = _exempt_key(client, master_key_header)
    resp = client.post(
        _PATH,
        content=b"\x1f\x8b\x08\x00 definitely not gzip",
        headers={**headers, "Content-Type": "application/json", "Content-Encoding": "gzip"},
    )
    assert resp.status_code == 400


def test_otlp_oversized_body_is_rejected(client: TestClient, master_key_header: dict[str, str]) -> None:
    headers = _exempt_key(client, master_key_header)
    resp = client.post(
        _PATH,
        content=b"0" * (8 * 1024 * 1024 + 1),
        headers={**headers, "Content-Type": "application/json"},
    )
    assert resp.status_code == 413


def test_otlp_rejected_events_reported_via_partial_success(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    """Events that cannot be attributed are reported in the OTLP partialSuccess."""
    headers = _exempt_key(client, master_key_header, user_id="doomed")
    # Soft-delete the user directly, leaving the key active: simulates the race
    # where a user is removed while an exporter still holds a live key.
    db_session.query(User).filter(User.user_id == "doomed").update({"deleted_at": datetime.now(UTC)})
    db_session.commit()

    resp = client.post(_PATH, json=_otlp(_api_request_record("req_rejected_1")), headers=headers)
    assert resp.status_code == 200, resp.text
    partial = resp.json().get("partialSuccess")
    assert partial is not None
    assert int(partial["rejectedLogRecords"]) == 1
    assert db_session.query(UsageLog).filter(UsageLog.source_event_id == "req_rejected_1").count() == 0


def test_otlp_gateway_client_name_falls_back_to_otel_source(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    """A client calling itself "gateway" must not masquerade as native traffic."""
    headers = _exempt_key(client, master_key_header)
    span = _span_record(**{"otari.client_name": "gateway", "gen_ai.response.id": "resp_reserved_1"})
    resp = client.post("/v1/traces", json=_otlp_traces(span), headers=headers)
    assert resp.status_code == 200, resp.text
    row = db_session.query(UsageLog).filter(UsageLog.source_event_id == "resp_reserved_1").one()
    assert row.source == "otel"
