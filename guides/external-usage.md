# Import usage from outside Otari

Usage that does not go through Otari, such as a coding agent on a subscription or
an app that calls a provider directly, can still be recorded. It then shows on
the **Overview** beside your routed traffic, with what the same usage would cost
on the API.

Importing works on a gateway you run, not on otari.ai today. To record a client's
usage on otari.ai, route its requests through Otari instead.

## Create an import key

Imported usage is recorded after the fact: it never counts toward a budget and
never blocks a request. So it is sent with a key that is exempt from budgets,
which only the deployment's operator can create.

1. Signed in as the operator, open **API keys** and click **Create key**.
2. Name the key for the importer, open **Advanced**, and tick
   **Exempt from budget**.
3. Click **Create key** and copy the key.

Use the key for importing only, since the exemption also applies to any request
sent with it. And do not import usage you also route through Otari, or it is
counted twice.

## Send usage

Pick the way that matches where the usage comes from.

### Claude Code and Codex

They send their usage as they run once pointed at Otari; see
[Use with Claude Code](use-with-claude-code.md) and
[Use with Codex](use-with-codex.md). For Claude Code sessions from before that,
`otari import claude-code` reads the transcripts already on disk; see
[Backfill history from local transcripts](use-with-claude-code.md#backfill-history-from-local-transcripts).

### OpenTelemetry

An app instrumented with the OpenTelemetry GenAI conventions exports to your
gateway's URL plus `/otlp`:

```bash
export OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:8000/otlp"
export OTEL_EXPORTER_OTLP_HEADERS="Authorization=Bearer $OTARI_IMPORT_KEY"
```

Its spans also show as [sessions](sessions.md). What Otari reads from them is in
[OpenTelemetry](https://docs.otari.ai/reference/gateway/external-usage-reference#opentelemetry).

### The Claude API

The Claude API can export spans for Messages API requests and Managed Agents to
an OTLP destination you add in the Claude Console
([Use OpenTelemetry with the Claude API](https://platform.claude.com/docs/en/manage-claude/opentelemetry/api)).
Add Otari as one:

| Console field | Value |
| --- | --- |
| Provider | Custom |
| OTLP endpoint URL | Your gateway's public HTTPS URL plus `/otlp` |
| Request headers | `Authorization`: `Bearer <your import key>` |
| Content mode | Metadata only |

To keep Claude workspaces apart, add one destination per workspace, each with its
own key; see [Claude Platform](https://docs.otari.ai/reference/gateway/external-usage-reference#claude-platform).

### Anything else

Send token counts to `POST /api/v1/usage/external-events`, up to 1,000 events a
batch:

```bash
curl "$OTARI_URL/api/v1/usage/external-events" \
  -H "Authorization: Bearer $OTARI_IMPORT_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "source": "my_app",
    "events": [{
      "source_event_id": "req_01ABC",
      "timestamp": "2026-07-22T12:34:56Z",
      "provider": "anthropic",
      "model": "claude-sonnet-4-6",
      "input_tokens": 1200,
      "output_tokens": 450
    }]
  }'
```

Sending a batch again is safe: events already stored are skipped. Every field is
in [Event fields](https://docs.otari.ai/reference/gateway/external-usage-reference#event-fields).

## See what you imported

On the **Overview**, **Requests** marks imported rows by their source, and
**Usage** shows them beside routed spend. Their cost is an estimate at API rates,
not what a subscription charged, and an event with no price is kept without a
cost. Budget gauges leave imported usage out. How each event is priced is in
[Pricing](https://docs.otari.ai/reference/gateway/external-usage-reference#pricing).
