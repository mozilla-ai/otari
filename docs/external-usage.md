# Importing external usage

Otari can record usage that it did not proxy, such as subscription-backed coding
agents or another instrumented application. Imported events appear in Activity
and Usage with their source and an API-equivalent cost estimate.

This feature is available in standalone and hosted control planes. A hybrid
gateway has no local usage database or import API.

## Enforcement boundary

Imported usage is retrospective. It is never reserved against a budget, never
updates the enforcement ledger, and never blocks live traffic.

An API key used for import must therefore have
`exclude_from_budget: true`. Keep importer keys separate from inference keys:
the same flag also exempts live traffic through that key from budget and pricing
enforcement.

If a session already routes through Otari, do not also export its telemetry to
Otari. The proxied and imported events cannot be correlated reliably, so cost
analytics would count the session twice.

## Authentication and attribution

`POST /v1/usage/external-events` accepts either:

- A budget-exempt API key. Events bind to that key's user and workspace.
- The master key. The batch or each event must name an existing user; the
  default workspace supplies organization context.

Prefer a dedicated API key for each importer. It limits attribution mistakes and
keeps the master key out of collectors.

## Import normalized events

```bash
curl "$OTARI_URL/v1/usage/external-events" \
  -H "Authorization: Bearer $OTARI_IMPORT_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "source": "claude_code",
    "events": [{
      "source_event_id": "req_01ABC",
      "timestamp": "2026-07-22T12:34:56Z",
      "provider": "anthropic",
      "model": "claude-sonnet-4-6",
      "input_tokens": 1200,
      "output_tokens": 450,
      "cache_read_tokens": 8000,
      "session_label": "project:otari"
    }]
  }'
```

The batch contains a source, up to 1,000 events, and an optional default
`user_id`. Each event supports:

| Field | Purpose |
| --- | --- |
| `source_event_id` | Required upstream ID and idempotency key with `source`. |
| `timestamp` | Required event time used to select effective pricing. |
| `provider`, `model` | Required pricing selector. |
| `status` | `success` or `error`; defaults to success. |
| token fields | Input, output, cache read, cache write, and one-hour cache write. |
| `cache_tokens_in_prompt` | Marks cache tokens as a subset of input for OpenAI-shaped counts. |
| `duration_ms` | Recorded request latency. |
| `session_label` | Optional session or project attribution. |
| `user_id` | Optional per-event override when the credential may name users. |

The endpoint rejects prompt, completion, tool input, and tool output fields.
Only metadata and numeric usage are accepted. Use the generated OpenAPI document
for the exact schema and validation limits.

Rows are unique on `(source, source_event_id)`. Replaying a batch reports
duplicates without creating new usage rows.

## OpenTelemetry

Otari accepts OTLP over HTTP:

```text
POST /v1/traces    GenAI spans
POST /v1/logs      GenAI log events and recognized coding-agent events
POST /v1/metrics   content-free coding-agent outcome metrics
```

Protobuf and JSON are accepted, with optional gzip. gRPC is not. Authenticate
with a budget-exempt API key in the Authorization header; the master key is
refused because OTLP records do not carry trustworthy user attribution.

For a standard OTLP exporter:

```bash
export OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
export OTEL_EXPORTER_OTLP_ENDPOINT="https://otari.example.com"
export OTEL_EXPORTER_OTLP_HEADERS="Authorization=Bearer $OTARI_IMPORT_KEY"
```

Otari reads the OpenTelemetry GenAI provider, model, response ID, input-token,
output-token, and cache-token attributes. It ignores non-LLM spans and never
stores prompt or response content.

Claude Code and Codex emit recognizable usage events on the logs signal. Their
tool behavior and outcome counters can also populate content-free agent
telemetry when `capture_agent_telemetry` is enabled. See the client-specific
setup:

- [Use with Claude Code](use-with-claude-code.md)
- [Use with Codex](use-with-codex.md)

## Pricing

Each accepted event is priced at the effective rate at its timestamp:

1. The importing key's organization override
2. Deployment pricing
3. The enabled default-pricing catalog

An unpriced event is still stored with `cost: null`. `require_pricing` does not
apply because imported usage is not enforceable.

Idempotent replay does not recalculate cost, and later price changes do not
modify existing rows. Use the usage repricing API when historical rows must be
updated.

Imported cost is an estimate based on configured API rates, not an invoice or a
subscription charge. Cache pricing is only as accurate as the token fields the
source exports.

## Reading imported usage

Use the normal usage APIs and filter on `source`, `api_key_id`, user, model,
or session label. Activity identifies imported rows and Usage separates priced
and unpriced totals.

Imported rows do not appear in budget-consumption gauges because those gauges
read the enforcement ledger.

## Related documentation

- [API reference](api-reference.md)
- [Configuration](configuration.md)
- [Access control](access-control.md)
