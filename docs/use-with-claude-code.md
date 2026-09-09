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
sessions that do not already route through Otari. Send it to a standalone or
hosted gateway; hybrid gateways do not serve the OTLP endpoints.

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

## Backfill history from local transcripts

The exporter above carries sessions that run after it is configured. Everything
Claude Code did before that is already on disk, one JSONL transcript per session
under `~/.claude/projects`. `otari import claude-code` reads those transcripts and
posts them to the same import endpoint:

```bash
export OTARI_URL="https://otari.example.com"

# Preview what is on disk. No credential needed: a dry run sends nothing.
otari import claude-code --dry-run

# Then import it, with a budget-exempt API key. Usage binds to that key's own
# user, so do not pass --user-id here.
export OTARI_API_KEY="gw-your-import-key"
otari import claude-code
```

The key must be budget-exempt (`exclude_from_budget: true`), the same requirement
every import has.

The master key works too, and it is the only credential that can import on behalf
of somebody else. It requires `--user-id`, and that user must already exist
(create one with `POST /v1/users` first):

```bash
export OTARI_MASTER_KEY="your-master-key"
otari import claude-code --user-id alice
```

Passing `--user-id` with an ordinary API key does not do this: usage always binds
to the key's own user, and naming a different one is rejected. Prefer a per-user
API key where you can, and keep the master key for a one-off backfill you are
running on someone's behalf.

The first event is posted on its own, so a rejection (an unknown user, a key that
is not budget-exempt) is reported before the rest of the history is uploaded. Use
`--since 7d` (or an ISO date) to read only recently modified transcripts.

Only token counts and identifiers are read. Prompts, completions, and tool
payloads are never opened out of a transcript, and the endpoint rejects them.

Each event carries a `session_label` of `<hostname>:<project>`, where the project
is the session's working directory with the home-directory prefix dropped, so
usage can be grouped per repository. Filter on it in Activity and Usage.

Rows are unique on `(source, source_event_id)`, and the source event id here is the
Anthropic response id, so re-running imports only what is new and reports the rest
as duplicates. A single reply spans several transcript lines under one response id;
those collapse to the one API call they were.

The same warning applies as above, and it matters more for a backfill: do not
import sessions that were routed through Otari. Those are already recorded, and the
proxied and imported rows cannot be correlated, so the cost would count twice.

