# Importing external usage

Otari can record usage it did not proxy, such as coding agents on a subscription
or another instrumented application. Imported events appear on the **Overview**
with their source and an estimate of what the same usage would cost on the API.

Importing works on a gateway you run, not on otari.ai today. It needs a
budget-exempt API key or the master key, and only a deployment's operator can
create either (an operator can give an exempt key to a member). To record a client's usage on otari.ai, route its requests through Otari
instead.

## Enforcement boundary

Imported usage is after the fact. It never counts toward a budget and never
blocks live traffic. That is why an import key must be budget-exempt.

- **Create a dedicated import key.** The operator ticks **Exempt from budget** in
  the deployment view's key form. Members cannot exempt their own keys.
- **Do not use it for inference.** The exemption also lifts budget and pricing
  checks on live traffic sent with that key.
- **Do not import what you route.** Proxied and imported events cannot be
  matched, so a session that does both is counted twice.

## Authentication and attribution

Events bind to whoever sends them:

- With a budget-exempt API key, events belong to that key's user and workspace.
  Prefer one key per importer.
- With the master key, the batch or each event must name an existing user, and
  the default workspace is used.

## Backfill Claude Code from local transcripts

`otari import claude-code` imports the Claude Code transcripts already on disk,
including history from before any exporter was set up. See
[Use with Claude Code](use-with-claude-code.md#backfill-history-from-local-transcripts).

## Import normalized events

Send batches of up to 1,000 events to `POST /api/v1/usage/external-events`:

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

A batch has a `source`, its `events` and an optional default `user_id`. Each
event takes:

| Field | Purpose |
| --- | --- |
| `source_event_id` | Required. The upstream id; with `source`, it makes the event unique. |
| `timestamp` | Required. Picks the price in effect at that time. |
| `provider`, `model` | Required. Pick the price. |
| `status` | `success` (default) or `error`. |
| token fields | Input, output, cache read, cache write, and one-hour cache write. |
| `cache_tokens_in_prompt` | Marks cache tokens as part of the input, as OpenAI counts them. |
| `duration_ms` | Request latency. |
| `session_label` | Optional session or project label. |
| `user_id` | Optional per-event user, when the credential may name users. |

Only metadata and token counts are accepted; prompt, completion and tool fields
are rejected. The exact schema is in the
[OpenAPI specification](https://github.com/mozilla-ai/otari/blob/e4a487c5f024d810a258b065a21391712a7b3efe/docs/public/openapi.json).

Sending a batch again is safe: events already stored come back as duplicates.
When the database is busy or unreachable, this endpoint and the OTLP endpoints
answer `503` with `Retry-After`; send the batch again.

## OpenTelemetry

Point a standard OTLP exporter at your gateway's URL plus `/otlp`, with a
budget-exempt key:

```bash
export OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:8000/otlp"
export OTEL_EXPORTER_OTLP_HEADERS="Authorization=Bearer $OTARI_IMPORT_KEY"
```

The exporter adds the signal path itself:

```text
POST /otlp/v1/traces    GenAI spans
POST /otlp/v1/logs      GenAI log events and recognized coding-agent events
POST /otlp/v1/metrics   content-free coding-agent outcome metrics
```

- OTLP over HTTP only, protobuf or JSON, optionally gzipped. No gRPC.
- Only a budget-exempt API key works. The master key is refused, because OTLP
  records carry no trustworthy user. With any other key, usage is refused and
  spans are not kept.
- Otari records usage from spans that carry the OpenTelemetry GenAI token
  attributes. It never keeps response or tool output; prompts, commands and file
  paths are kept only where trace content is on and the client sends them (see
  [Keeping what a span was about](traces.md#keeping-what-a-span-was-about)).
- The spans also appear on the key's agent trace; see
  [Agent traces](traces.md#instrumented-agents).

Emitters disagree on whether cached tokens are part of the input count. Otari
reads OpenLLMetry's Anthropic spans as counting them in, OpenLIT's as counting
them on top, and any other emitter's as on top when cached tokens outnumber
input tokens. Cache writes are read from
`gen_ai.usage.cache_write.input_tokens` or
`gen_ai.usage.cache_creation.input_tokens`, and the one-hour share from
`anthropic.usage.cache_creation.ephemeral_1h_input_tokens` where it is sent.

### Claude Platform and the Claude SDKs

The Claude API can export spans for Messages API requests and Managed Agents
turns to an OTLP destination added in the Claude Console
([Use OpenTelemetry with the Claude API](https://platform.claude.com/docs/en/manage-claude/opentelemetry/api)).
Add Otari as one:

| Console field | Value |
| --- | --- |
| Provider | Custom |
| OTLP endpoint URL | Your gateway's public HTTPS URL plus `/otlp` |
| Request headers | `Authorization`: `Bearer <budget-exempt Otari API key>` |
| Content mode | Metadata only |

- The URL must be a public hostname over HTTPS on port 443, with no redirect.
- Use **Metadata only**: Otari keeps no content, and full-content spans can
  push an export past the 8 MiB limit.
- Every span binds to the one key's user. To keep Claude workspaces apart, add
  one destination per workspace, each with its own key, and set its
  **Workspaces** to **Selected** with only that workspace. Two destinations both
  on **All** would book the same usage twice.

Spans from the Claude Platform and the Claude SDKs are recorded under the source
`claude_platform`, and each token is booked once:

- A Messages API request is booked from its `anthropic.messages.create` span,
  with its cache reads and writes, plus a row for any compaction tokens. One-hour
  cache writes bill at twice the input rate
  ([Cache and tiered pricing](https://docs.otari.ai/self-hosting/configuration#cache-and-tiered-pricing)),
  except from a streamed request's SDK span, which bills all its writes at the
  cache-write rate. A request exported by both the SDK and the API is booked
  once.
- A Managed Agents session is booked from its `anthropic.model_request` spans,
  labeled with the session id, which groups its threads into one agent trace.
- Outcome evaluation spans name no model, so their tokens are not booked.

The Claude API delivers spans on a best-effort basis, so these rows estimate
spend. The Claude Usage API is the complete record.

### Coding agents

Claude Code and Codex send recognized usage events on the logs signal. With
`capture_agent_telemetry` on, their tool and outcome counters also feed
content-free agent telemetry. Setup:

- [Use with Claude Code](use-with-claude-code.md)
- [Use with Codex](use-with-codex.md)

## Pricing

Each event is priced at the rate in effect at its timestamp, taken from the
first of:

1. the importing key's organization override
2. deployment pricing
3. the enabled default pricing catalog

An unpriced event is still stored, with no cost. `require_pricing` does not
apply. Later price changes and replays do not change stored rows; reprice them
with the usage repricing API.

Imported cost is an estimate at API rates, not an invoice or a subscription
charge, and cache pricing is only as accurate as the source's token counts.

## Reading imported usage

Imported rows show on the **Overview**: **Requests** marks them, and **Usage**
separates priced from unpriced totals. The usage APIs filter on `source`,
`api_key_id`, user, model and session label.

Budget gauges do not include imported usage.
