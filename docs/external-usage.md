# Importing external usage

Otari can record usage that it did not proxy, such as subscription-backed coding
agents or another instrumented application. Imported events appear in the
dashboard's Overview with their source and an API-equivalent cost estimate.

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

`POST /api/v1/usage/external-events` accepts either:

- A budget-exempt API key. Events bind to that key's user and workspace.
- The master key. The batch or each event must name an existing user; the
  default workspace supplies organization context.

Prefer a dedicated API key for each importer. It limits attribution mistakes and
keeps the master key out of collectors.

## Backfill Claude Code from local transcripts

For Claude Code specifically, the bundled `otari import claude-code` reads the
transcripts already on disk and posts them here, which is how history from before
any exporter was configured gets in. See
[Use with Claude Code](use-with-claude-code.md#backfill-history-from-local-transcripts).

## Import normalized events

```bash
curl "$OTARI_URL/api/v1/usage/external-events" \
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
duplicates without creating new usage rows. When the database cannot take a
batch right away (another import of the same events is still writing, or the
database is unreachable), this endpoint and the OTLP endpoints below answer
`503` with `Retry-After`. Send the batch again: the events it already stored
come back as duplicates.

## OpenTelemetry

Otari accepts OTLP over HTTP:

```text
POST /otlp/v1/traces    GenAI spans
POST /otlp/v1/logs      GenAI log events and recognized coding-agent events
POST /otlp/v1/metrics   content-free coding-agent outcome metrics
```

Protobuf and JSON are accepted, with optional gzip. gRPC is not. Authenticate
with a budget-exempt API key in the Authorization header; the master key is
refused because OTLP records do not carry trustworthy user attribution.

For a standard OTLP exporter:

```bash
export OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
export OTEL_EXPORTER_OTLP_ENDPOINT="https://otari.example.com/otlp"
export OTEL_EXPORTER_OTLP_HEADERS="Authorization=Bearer $OTARI_IMPORT_KEY"
```

The endpoint is the Otari origin plus `/otlp`; the exporter appends the signal
path itself.

Otari reads the OpenTelemetry GenAI provider, model, response ID, input-token,
output-token, and cache-token attributes, and records usage from the spans that
carry tokens. While `trace_capture_enabled` is on, the spans, including an
agent's own (`invoke_agent`, `execute_tool`), are also kept on the exporting
key's agent trace, grouped into a session by `anthropic.session.id`, else
`gen_ai.conversation.id`, else `otari.session_label`, else the OTel trace. Span names and attribute values are
kept only when they are identifier-shaped; prompt, response, and tool content is
never stored.

Keeping them is best-effort, and its limits are counted rather than refused: one
export keeps at most 1,000 spans and 100 sessions, and spans past either limit,
or that arrive while the trace writer's queue is full, are dropped and counted
on `gateway_trace_spans_dropped`. The export itself, and the usage it carries,
is never refused for its traces.

### Claude Platform and the Claude SDKs

The Claude API exports a span for each Messages API request, model request and
tool use, and each Managed Agents turn, to an OTLP destination added in the
Claude Console ([Use OpenTelemetry with the Claude API](https://platform.claude.com/docs/en/manage-claude/opentelemetry/api)).
Otari can be that destination:

| Console field | Value |
| --- | --- |
| Provider | Custom |
| OTLP endpoint URL | `https://otari.example.com/otlp` |
| Request headers | `Authorization`: `Bearer <budget-exempt Otari API key>` |
| Content mode | Metadata only |

The Claude API delivers only to a public hostname over HTTPS on port 443 and
treats a redirect as fatal, so the URL must reach Otari directly. Full content
gains nothing here, since Otari keeps no prompt or response, and its larger
spans can push an export past the 8 MiB request limit. Every span of the
destination binds to the one key's user, so to keep Claude workspaces apart,
add one destination per workspace, each with its own key, and set its
**Workspaces** to **Selected** with only that workspace. The Claude API sends a
workspace's spans to every destination that covers it, so two destinations left
on **All** would each book the same usage under a different key.

Spans from the Claude Platform and from the Claude SDKs (instrumentation scope
`com.anthropic.*`) are recorded under the source `claude_platform`, and each
token is booked once even though a request repeats its usage on several spans:

- A Messages API request is booked from its `anthropic.messages.create` span,
  keyed by the response id, at the model that served it, with its cache reads
  and cache writes, and a second row for its compactions' tokens. The 1-hour
  share of its cache writes is kept as its own count and bills at twice the
  input rate ([Cache and tiered pricing](configuration.md#cache-and-tiered-pricing)),
  except from a streamed request's SDK span, which splits out only the writes
  made before the first output token, so all of its writes bill at the
  cache-write rate. When the SDK and the API both export the request, the
  second span is a duplicate.
- A Managed Agents session is booked from its `anthropic.model_request` spans,
  labeled with the session id, which also groups the session's threads into one
  agent trace.
- Outcome evaluation spans name no model, so their tokens are not booked.

The Claude API's delivery is best effort, so these rows estimate spend rather
than reconcile it; the Claude Usage API is the complete record.

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
