# Use with Claude Code

Claude Code speaks the Anthropic Messages API. Otari serves
`POST /v1/messages` and `POST /v1/messages/count_tokens` in standalone and
hybrid modes.

## Route Claude Code through Otari

Claude Code appends the Messages paths itself, so `ANTHROPIC_BASE_URL` must be
the Otari origin without `/v1`.

### Connected to otari.ai

```bash
export ANTHROPIC_BASE_URL="https://api.otari.ai"
export ANTHROPIC_AUTH_TOKEN="tk_your_otari_token"
export ANTHROPIC_MODEL="anthropic:claude-sonnet-4-6"
claude
```

### Standalone

Claude Code sends a telemetry value in the request's `user` field. It is not an
Otari user ID, so create or update its API key with
`reject_user_mismatch: false`. Spend still binds to the key's user.

```bash
curl "$OTARI_URL/v1/keys" \
  -H "Authorization: Bearer $OTARI_MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "key_name": "claude-code",
    "user_id": "alice",
    "reject_user_mismatch": false
  }'
```

Then point Claude Code at the standalone gateway:

```bash
export ANTHROPIC_BASE_URL="http://localhost:8000"
export ANTHROPIC_AUTH_TOKEN="gw-your-otari-key"
export ANTHROPIC_MODEL="anthropic:claude-sonnet-4-6"
claude
```

Use `ANTHROPIC_AUTH_TOKEN`, which sends an Authorization bearer token and works
in both modes. `ANTHROPIC_API_KEY` uses `x-api-key`; standalone Otari accepts
it, but hybrid authentication does not.

The same values can go in the `env` block of Claude Code's settings file.

## Choosing a model

Otari can translate Messages requests to non-Anthropic providers. Set Claude
Code's default Opus, Sonnet, and Haiku variables when its built-in aliases do not
exist on your deployment.

Non-Claude models may lose Anthropic-specific behavior such as extended thinking,
prompt caching, or tool semantics. Test the actual agent workflow before making
one a default.

## Import Claude Code usage without routing

Claude Code can send subscription usage to Otari over OpenTelemetry. This is for
sessions that do not already route through Otari.

Create a dedicated API key with `exclude_from_budget: true`, then configure the
logs exporter. The optional metrics exporter adds content-free outcome counters.

```bash
export CLAUDE_CODE_ENABLE_TELEMETRY=1
export OTEL_LOGS_EXPORTER=otlp
export OTEL_METRICS_EXPORTER=otlp
export OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
export OTEL_EXPORTER_OTLP_ENDPOINT="https://otari.example.com"
export OTEL_EXPORTER_OTLP_HEADERS="Authorization=Bearer gw-your-import-key"
claude
```

The endpoint is the Otari origin; the exporter appends `/v1/logs` and
`/v1/metrics`. Use an HTTP protocol because Otari does not accept OTLP over
gRPC.

Imported events are priced for analytics and never count toward budgets. Do not
both route and export one session, or its cost will appear twice. See
[Importing external usage](external-usage.md) for attribution, privacy,
idempotency, and pricing behavior.
