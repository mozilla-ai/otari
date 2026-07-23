# Use with Claude Code

[Claude Code](https://code.claude.com) speaks the Anthropic Messages API and lets
you redirect it at any compatible endpoint with a couple of environment
variables. Otari exposes that surface (`POST /v1/messages` and
`POST /v1/messages/count_tokens`) in both standalone and hybrid modes, so you
can route Claude Code through Otari to get budgets, usage tracking, and traces
without changing how you use the CLI.

## Quick start

Claude Code appends `/v1/messages` and `/v1/messages/count_tokens` to
`ANTHROPIC_BASE_URL` itself, so the base URL must be the Otari root, not
`/v1`. For local development, use `http://localhost:8000`.

### Connected to otari.ai

```bash
export ANTHROPIC_BASE_URL="https://api.otari.ai"   # your Otari base URL, no /v1
export ANTHROPIC_AUTH_TOKEN="tk_your_otari_token"  # sent as Authorization: Bearer
export ANTHROPIC_MODEL="anthropic:claude-sonnet-4-6"
claude
```

### Standalone

Claude Code attaches its own `metadata.user_id` to every request. In standalone
mode Otari binds spend to the API key's own user and, by default, rejects a
request that names a different user (`403 permission_error`). Set
`reject_user_mismatch: false` in your config so Claude Code's `user_id` is
ignored and spend is still bound to the key's user.

```yaml
reject_user_mismatch: false
```

Then run Claude Code against your local Otari:

```bash
export ANTHROPIC_BASE_URL="http://localhost:8000"
export ANTHROPIC_AUTH_TOKEN="<your-otari-api-key>"
export ANTHROPIC_MODEL="anthropic:claude-sonnet-4-6"
claude
```

Use `ANTHROPIC_AUTH_TOKEN` (not `ANTHROPIC_API_KEY`): it is sent as
`Authorization: Bearer <token>`, which is the scheme Otari accepts for
both standalone API keys and connected user tokens. `ANTHROPIC_API_KEY` is sent
as an `x-api-key` header instead. In standalone mode Otari reads that header too,
so it also authenticates; connected mode, though, expects `Authorization: Bearer`
and does not use local API keys, so `ANTHROPIC_AUTH_TOKEN` is the portable choice
that works in both modes.

### settings.json

The same configuration works in `~/.claude/settings.json` (or a project-level
`.claude/settings.json`). Replace the values with your deployment's URL, token,
and model defaults:

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "https://api.otari.ai",
    "ANTHROPIC_AUTH_TOKEN": "tk_your_otari_token",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "anthropic:claude-opus-4-8",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "anthropic:claude-sonnet-4-6",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "anthropic:claude-haiku-4-5"
  }
}
```

## Choosing a model

Claude Code speaks the Anthropic wire format, but the `model` field is just a
string Otari forwards to [any-llm](https://github.com/mozilla-ai/any-llm).
Examples in this page use `provider:model`; Otari also accepts
`provider/model` in both standalone and connected mode.

- **Claude models:** `anthropic:claude-sonnet-4-6`,
  `anthropic:claude-opus-4-8`, `anthropic:claude-haiku-4-5`
- **Standalone, any configured provider:** `openai:gpt-4o`,
  `mistral:mistral-large-latest`, `anthropic:claude-sonnet-4-6`
- **Connected to otari.ai, managed models:** `mzai:<catalog-id>`, for example
  `mzai:moonshotai/Kimi-K2.6`. These run only through otari.ai's hosted gateway.
- **Connected to otari.ai, your own provider keys:** `openai:gpt-4o`,
  `mistral:mistral-large-latest`, `anthropic:claude-sonnet-4-6`

If you use Claude Code's `opus`, `sonnet`, or `haiku` aliases, set the matching
`ANTHROPIC_DEFAULT_*_MODEL` variables so Claude Code does not fall back to a
model your Otari deployment does not serve.

### Non-Claude models

Otari can route Claude Code to non-Claude models, but Claude Code is tuned for
Claude. Expect weaker tool use on some models, and expect Anthropic-specific
features such as extended thinking or prompt caching to be dropped when the
target provider does not support them.

## Import subscription usage (without routing through Otari)

If you keep Claude Code on a subscription rather than routing it through Otari, you
do not pay API rates, but Otari also can't see that usage. You can still get it into
your usage analytics, priced at API-equivalent rates, by pointing Claude Code's
OpenTelemetry export at Otari. Nothing about how you run Claude Code changes; it
reports each request's usage as it goes. This is standalone-only and never affects
budgets. Point the exporter at a standalone gateway: a hybrid (otari.ai-connected)
gateway does not serve `/v1/logs`, so the export 404s and the telemetry is dropped.

Claude Code has native OpenTelemetry support: it emits an `api_request` log event
per model call carrying token counts, the model, and a request id, but no prompt or
response content. Otari accepts those directly at `POST /v1/logs`, so no separate
collector is required.

### 1. Get a budget-exempt import key (admin, once)

Imported usage is retrospective, so Otari can never block it, which is why an import
key must be **budget-exempt** (a budgeted key is refused). In the dashboard: Keys ->
create a key for the user, open **Advanced**, check **Exempt from budget**. Or over
the API with the master key:

```bash
curl -sS "$OTARI_URL/v1/keys" \
  -H "Otari-Key: Bearer $OTARI_MASTER_KEY" -H "Content-Type: application/json" \
  -d '{"key_name":"claude-code-importer","user_id":"alice","exclude_from_budget":true}'
```

Treat that key as a secret: `exclude_from_budget` also exempts this key's **live**
gateway traffic from reservations, spend, and budget enforcement, so a key that leaks
or is reused for routing grants unmetered access. Use a key (and ideally a dedicated
user) reserved solely for imports, and rotate it if it is exposed.

### 2. Point Claude Code's telemetry at Otari

Enable telemetry and send the **logs** signal (which carries `api_request`) to
Otari's base URL, authenticating with the exempt key. Claude Code appends `/v1/logs`
itself, so the endpoint is the Otari root, not `/v1`.

```bash
export CLAUDE_CODE_ENABLE_TELEMETRY=1
export OTEL_LOGS_EXPORTER=otlp
export OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf      # http/json also works
export OTEL_EXPORTER_OTLP_ENDPOINT="https://otari.example.com"
export OTEL_EXPORTER_OTLP_HEADERS="Authorization=Bearer gw-your-exempt-key"
claude
```

`OTEL_LOGS_EXPORTER` is the load-bearing one; a metrics-only exporter carries no
per-request usage. Set an **http** protocol (`http/protobuf` or `http/json`); the
default is gRPC, which Otari's HTTP receiver does not accept. The same settings work
in the `env` block of `~/.claude/settings.json` so every session reports
automatically:

```json
{
  "env": {
    "CLAUDE_CODE_ENABLE_TELEMETRY": "1",
    "OTEL_LOGS_EXPORTER": "otlp",
    "OTEL_EXPORTER_OTLP_PROTOCOL": "http/protobuf",
    "OTEL_EXPORTER_OTLP_ENDPOINT": "https://otari.example.com",
    "OTEL_EXPORTER_OTLP_HEADERS": "Authorization=Bearer gw-your-exempt-key"
  }
}
```

Otari re-prices each event at its own configured rate for the event's timestamp
(Claude Code's own cost estimate is ignored), records it as `source = claude_code`,
and dedups by request id, so replays never double-count. A model with no configured
price still lands with `cost: null`; add pricing to see the cost.

### 3. See it

In the dashboard, the Activity page shows each imported request with its **API key**
column, and (expanded) its Source and session; the Usage page's **Tracked cost**
total separates priced from unpriced usage, with an "unpriced" hint when a model has
no price. Filter the Activity log by API key to scope it to your importer key.

## See also

- [Importing external usage](external-usage.md) for tracking subscription-backed usage
- [Modes](modes.md) for standalone vs connected behavior
- [API reference](api-reference.md) for the Messages endpoints and auth rules
