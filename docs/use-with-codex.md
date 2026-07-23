# Use with Codex

[Codex](https://developers.openai.com/codex) has native OpenTelemetry support: it
emits a usage event per model call carrying token counts, the model, and a session
id, but no prompt or response content. If you run Codex on its own credentials
(directly against OpenAI, not routed through Otari), Otari can still see that usage:
point Codex's OTLP export at Otari and each request lands in your usage analytics,
priced at API-equivalent rates. Nothing about how you run Codex changes.

This is standalone-only and never affects budgets (a hybrid, otari.ai-connected
gateway does not serve the OTLP endpoints, so the export 404s there). See
[Importing external usage](external-usage.md) for the shared pricing, idempotency,
and budget-exempt rules; this page is just the Codex setup.

> **Route or export, not both.** Telemetry import is for sessions that do NOT proxy
> through Otari. If Codex both routes its API traffic through Otari and exports
> telemetry to it, every call lands twice: once as `source = gateway` (enforced,
> counts toward budget) and once as `source = codex` (exempt observability). The two
> rows are not correlated, so budgets and `spend` stay correct, but cost analytics
> count the same traffic twice. Pick one path per session.

## 1. Get a budget-exempt import key (admin, once)

Imported usage is retrospective, so Otari can never block it, which is why an import
key must be **budget-exempt** (a budgeted key is refused). In the dashboard: Keys ->
create a key for the user, open **Advanced**, check **Exempt from budget**. Or over
the API with the master key:

```bash
curl -sS "$OTARI_URL/v1/keys" \
  -H "Otari-Key: Bearer $OTARI_MASTER_KEY" -H "Content-Type: application/json" \
  -d '{"key_name":"codex-importer","user_id":"alice","exclude_from_budget":true}'
```

Treat that key as a secret: `exclude_from_budget` also exempts this key's **live**
gateway traffic from reservations, spend, and budget enforcement, so a key that leaks
or is reused for routing grants unmetered access. Use a key (and ideally a dedicated
user) reserved solely for imports, and rotate it if it is exposed.

## 2. Point Codex's telemetry at Otari

Codex configures OpenTelemetry in the `[otel]` section of `~/.codex/config.toml`.
Send it to Otari with the **full** `/v1/logs` path (Codex does not append the signal
path itself), an `http` protocol, and the exempt key:

```toml
# ~/.codex/config.toml
[otel]
environment = "otari"
log_user_prompt = false
exporter = { otlp-http = { endpoint = "https://otari.example.com/v1/logs", protocol = "binary", headers = { "Authorization" = "Bearer gw-your-exempt-key" } } }
```

- `endpoint` must include `/v1/logs`. Unlike the `OTEL_EXPORTER_OTLP_ENDPOINT`
  environment variable, Codex's configured endpoint is used as-is.
- `protocol = "binary"` sends protobuf; `"json"` also works. gRPC is not accepted by
  Otari's HTTP receiver.
- `log_user_prompt = false` keeps prompts out of the export. Otari never stores prompt
  or response content regardless, but there is no reason to send it.

Otari re-prices each event at its own configured rate for the event's timestamp,
records it as `source = codex`, and dedups per request so replays never double-count.
A model with no configured price still lands with `cost: null`; add pricing to see the
cost. Codex reports OpenAI-shaped token counts (cached tokens are a subset of the
input tokens), and Otari de-includes them so cache reads are not billed twice.

## 3. See it

In the dashboard, the Activity page shows each imported request with its **API key**
column, and (expanded) its Source (`Codex`) and session; the Usage page's **Tracked
cost** total separates priced from unpriced usage. Filter the Activity log by API key
to scope it to your importer key.

## See also

- [Importing external usage](external-usage.md) for the ingestion contract, pricing, and budget behavior
- [Use with Claude Code](use-with-claude-code.md) for the equivalent Claude Code setup
- [API reference](api-reference.md)
