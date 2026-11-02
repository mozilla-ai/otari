# Use with Claude Code

Claude Code speaks the Anthropic Messages API. Otari serves
`POST /api/v1/messages` and `POST /api/v1/messages/count_tokens`.

## Route Claude Code through Otari

Claude Code appends `/v1/messages` itself, so `ANTHROPIC_BASE_URL` ends in
`/api`, not `/api/v1`.

### On otari.ai

```bash
export ANTHROPIC_BASE_URL="https://api.otari.ai/api"
export ANTHROPIC_AUTH_TOKEN="<your-otari-key>"   # an API key from otari.ai
export ANTHROPIC_MODEL="anthropic:claude-sonnet-4-6"
claude
```

Accounts in otari.ai's EU region set `ANTHROPIC_BASE_URL="https://eu.api.otari.ai/api"` instead: `api.otari.ai` answers an EU key with `421 Misdirected Request`, naming the host to use.

Claude Code puts a telemetry value in `metadata.user_id`; the next section
says how a key accepts it.

### On your own gateway

Claude Code sends a telemetry value in `metadata.user_id`. It is not an
Otari user ID, so create or update its API key with
`reject_user_mismatch: false`, or pick **Always accept** for "Mismatched `user`
field" under the dashboard key form's **Advanced** section. Spend still binds
to the key's user; see
[Mismatched `user` fields](access-control.md#mismatched-user-fields).

```bash
curl "$OTARI_URL/api/v1/keys" \
  -H "Authorization: Bearer $OTARI_MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "key_name": "claude-code",
    "user_id": "alice",
    "reject_user_mismatch": false
  }'
```

Then point Claude Code at the gateway:

```bash
export ANTHROPIC_BASE_URL="http://localhost:8000/api"
export ANTHROPIC_AUTH_TOKEN="tk-your-otari-key"
export ANTHROPIC_MODEL="anthropic:claude-sonnet-4-6"
claude
```

`ANTHROPIC_AUTH_TOKEN` sends an Authorization bearer token, and
`ANTHROPIC_API_KEY` sends `x-api-key`; Otari accepts either.

The same values can go in the `env` block of Claude Code's settings file.

A workload on Google Cloud, or behind another OpenID Connect issuer, can skip
the key altogether: Claude Code's own workload identity
(`ANTHROPIC_FEDERATION_RULE_ID`, `ANTHROPIC_ORGANIZATION_ID` and
`ANTHROPIC_IDENTITY_TOKEN_FILE`) works against an Otari that trusts
the issuer. Unset `ANTHROPIC_AUTH_TOKEN` and `ANTHROPIC_API_KEY` when you switch:
either one, when set, takes precedence and the federation variables are
ignored. See
[Claude Code and the Anthropic SDKs](workload-identity.md#claude-code-and-the-anthropic-sdks).

## Tag Claude Code traffic

Claude Code cannot set a request body's `metadata`, but it can add headers, so
it tags its spend with the `Otari-Tags` header through
`ANTHROPIC_CUSTOM_HEADERS`:

```bash
export ANTHROPIC_CUSTOM_HEADERS="Otari-Tags: agent=bug-fix,run=1234"
claude
```

Every request the session makes is then recorded with those tags, and the usage
endpoints filter and group by them (`GET /api/v1/usage?tag=run:1234`), so one
key can split its cost by agent, run or requester. Percent-encode a `,` or `=`
inside a value. See
[Request tags](api-reference.md#request-tags) for the format and limits.

A headless run that should read as one session of its own sends a run id too,
which the dashboard's Sessions list then shows and searches by:

```bash
export ANTHROPIC_CUSTOM_HEADERS=$'Otari-Conversation-Id: bug-fix-1234\nOtari-Tags: agent=bug-fix,run=1234'
```

It takes the place of Claude Code's own session id for grouping. See
[Sessions from your own agent](traces.md#sessions-from-your-own-agent).

## Budgets

A request a budget refuses answers `403` with `Otari-Error-Code: budget_exceeded`.
Claude Code prefixes every 403 with "Failed to authenticate", so a spent budget
reads:

```text
Failed to authenticate. API Error: 403 User 'alice' has exceeded budget limit
```

The text after the prefix is the reason; the key itself is fine. Claude Code
does not retry it, and a `Retry-After` header says when the budget's window
resets, where it does.

Otari holds a request's worst case against the budget before it runs, so a call
can be refused while some budget is left. Claude Code asks for up to 32,000
output tokens on a model id it does not know, which is a large hold on an
expensive model; set `CLAUDE_CODE_MAX_OUTPUT_TOKENS` lower when budgets are
small.

## Choosing a model

Otari can translate Messages requests to non-Anthropic providers. Set Claude
Code's default Opus, Sonnet, and Haiku variables when its built-in aliases do not
exist on your deployment.

Non-Claude models may lose Anthropic-specific behavior such as extended thinking,
prompt caching, or tool semantics. Claude Code's beta flags and its
`context_management` (server-side context editing) are dropped for a model with
no Anthropic Messages API of its own, so Anthropic's server-side context editing
never runs there; Claude Code's own compaction still does.
Test the actual agent workflow before making one a default.

### OpenAI models

By default an OpenAI model is served as a chat completion, which has no way to
send a reasoning model's reasoning back, so each turn of an agent loop starts
its reasoning over, and models OpenAI serves only on the Responses API (the
codex and pro variants) are out of reach. Set `messages_api: responses` on the
OpenAI instance to serve Claude Code through the Responses API instead:

```yaml
providers:
  openai:
    api_key: ${OPENAI_API_KEY}
    messages_api: responses
```

Claude Code then receives the model's reasoning as thinking blocks and sends them
back on the next turn, which is what lets the reasoning carry over. See
[Messages through the Responses API](configuration.md#messages-through-the-responses-api)
for how fields translate.

## Let Otari choose a subagent's model

Switching the model inside a running Claude Code conversation cold-starts its
prompt cache. A new subagent has no cache yet, so the moment it is spawned is
where a cheaper model is free. Claude Code can ask Otari at that moment, even
when its own requests go to Anthropic directly.

`POST /api/v1/routing/recommend` takes the facts a harness holds at spawn
time and recommends the model to start the subagent on. Otari puts one choice
question to a decision model, such as TypeSafe's Jev: given the subagent
type, the task and the parent's model, which is the cheapest candidate that
would do the task well? The candidate it picks is the recommendation. Any
active API key may ask. The decision call is billed to the caller and recorded
in usage like a `POST /api/v1/decisions` call; nothing else is dispatched, and
the harness acts on the answer with its own provider credentials.

Who answers is the deployment's choice. An operator stores one or more
decision models on the **Subagent router** page of the deployment view, each
with its key and a price per million tokens, and switches one on; a
recommendation is then charged to the caller at that price (see
[Subagent routers](routing.md#subagent-routers)). The router may be a local
model, and both prices at 0 make it free. On otari.ai that router is managed
for you.

With no router stored, or every stored router switched off, the route answers
400 and the plugin falls back to the model the subagent would have had
anyway.

```bash
curl "$OTARI_URL/api/v1/routing/recommend" \
  -H "Authorization: Bearer $OTARI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "harness": "claude-code",
    "session_id": "bf05abe2-5ff2-4eb0-8459-ab578d5c9468",
    "tool_use_id": "toolu_01Ab3dEfGh",
    "agent_type": "Explore",
    "description": "Find the budget reservation code",
    "prompt": "Find where budgets are reserved before dispatch.",
    "parent_model": "claude-opus-5",
    "requested_model": null
  }'
```

```json
{
  "model": "haiku",
  "reason": "haiku with 81%",
  "probabilities": { "haiku": 0.81, "sonnet": 0.16, "opus": 0.03 }
}
```

`model` is one of the `agent_recommender_candidates`, Claude Code's `haiku`,
`sonnet` and `opus` aliases by default, because Claude Code sends the
subagent's requests to Anthropic itself and resolves the alias on its side.
`harness` is `claude-code` or `pi`, and it picks the list: a pi request
sends its own `candidates`, Otari model ids with what each is for, and a
Claude Code request may not send any (see
[Agent model recommendations](configuration.md#agent-model-recommendations)).
The caller's own `requested_model` is received as a fact but kept out of the
question, so the orchestrator's guess cannot pull the answer toward itself.
The task reaches the decision model shortened to its first 32,000 characters.

### Install the plugin

The Claude Code side is the `otari-router` plugin, which this repository
publishes as a plugin marketplace. In a Claude Code session:

```
/plugin marketplace add mozilla-ai/otari
/plugin install otari-router@otari
```

The install asks for the gateway URL and an API key. Both are required; the
key goes to Claude Code's secure storage, never to a settings file. The URL
must be `https` unless the gateway runs on the same machine, since the key
travels in a header. The plugin is
active from then on, in that session and
every one after. The
[plugin's README](../plugins/otari-router/README.md) has the options, the
scopes, how to update, and how to run the checkout's copy while developing.

The plugin hooks `agent.spawn`, sends the request above, and passes the
answer on with `next({ ...e, model })`. A settings-file hook cannot do this:
`SubagentStart` only adds context. A fork always inherits its parent's model,
so the plugin does not ask about one. Each spawn leaves a dim line in the
transcript naming the model Otari chose and why, and each finished subagent
one with the model it ran on and its token counts. When Otari does not
answer, the subagent starts on the model it would have had anyway, and the
line says so.

## Send Claude Code sessions to Otari

Otari can record each Claude Code session without its model traffic going
through Otari: every prompt as a turn, with its model requests, tool calls and
their cost, and each [agent guardrail](agent-guardrails.md) check the hook made
on the way. Claude Code exports its own spans and usage over OpenTelemetry, and
`otari hook` reports its checks to the same session.

You need:

- an Otari deployment that records traces (`OTARI_TRACE_CAPTURE_ENABLED`, on by
  default);
- an API key created with `exclude_from_budget: true`, since what Claude Code
  reports is imported, never charged;
- for the guardrail checks, the hook registered (`otari hook setup`, or once in
  `~/.claude/settings.json` for every repository; see
  [Agent guardrails](agent-guardrails.md)).

Put the variables in the `env` block of `~/.claude/settings.json`, so they apply
to every session and reach the hook:

```json
{
  "env": {
    "CLAUDE_CODE_ENABLE_TELEMETRY": "1",
    "CLAUDE_CODE_ENHANCED_TELEMETRY_BETA": "1",
    "OTEL_TRACES_EXPORTER": "otlp",
    "OTEL_LOGS_EXPORTER": "otlp",
    "OTEL_EXPORTER_OTLP_PROTOCOL": "http/protobuf",
    "OTEL_EXPORTER_OTLP_ENDPOINT": "https://otari.example.com/otlp",
    "OTEL_EXPORTER_OTLP_HEADERS": "Authorization=Bearer tk-your-import-key",
    "OTARI_URL": "https://otari.example.com",
    "OTARI_API_KEY": "tk-your-import-key"
  }
}
```

Each does one part:

| Variables | What they send |
| --- | --- |
| `CLAUDE_CODE_ENABLE_TELEMETRY`, `CLAUDE_CODE_ENHANCED_TELEMETRY_BETA`, `OTEL_TRACES_EXPORTER` | The session's turns, model requests and tool calls. Span export is a Claude Code beta. |
| `OTEL_LOGS_EXPORTER` | Each model request's usage, which is what prices the session. Without it a session has tokens but no cost. |
| `OTEL_EXPORTER_OTLP_*` | Where both go: the Otari origin plus `/otlp`, over HTTP, with the key. Otari does not accept OTLP over gRPC. |
| `OTARI_URL`, `OTARI_API_KEY` | The hook's guardrail checks. Claude Code passes no `OTEL_*` variable to its hooks, so the hook cannot use the settings above. |

The key sits in the settings file in plaintext, twice. A script that prints the
header (`otelHeadersHelper`; see Claude Code's
[monitoring docs](https://code.claude.com/docs/en/monitoring-usage)) replaces
only `OTEL_EXPORTER_OTLP_HEADERS`; the hook still needs `OTARI_API_KEY`. To keep
that out of the file too, leave it out of the `env` block and export it in the
shell that starts Claude Code, since a hook inherits Claude Code's environment.
On macOS, for example:

```sh
export OTARI_API_KEY="$(security find-generic-password -s otari-import-key -w)"
```

To check it works, run Claude Code, ask for something that uses a tool or two,
and open **Sessions** in the dashboard: the session shows its turns, model
requests and tools, and At a glance where its time and money went. A gate that
ran shows under the tool call it checked. A short `claude -p` run can exit
before its last spans are sent.

By default nothing says what a span was about: no prompt, command or file path.
To keep them, the deployment needs `OTARI_TRACE_CONTENT_ENABLED=true` and
`OTARI_SECRET_KEY`, and Claude Code and the hook have to send them:

```json
{
  "env": {
    "OTEL_LOG_USER_PROMPTS": "1",
    "OTEL_LOG_TOOL_DETAILS": "1",
    "OTARI_HOOK_REPORT_CONTENT": "1"
  }
}
```

See [Keeping what a span was about](traces.md#keeping-what-a-span-was-about).
Do not both [route](#route-claude-code-through-otari) and export one session,
or its cost appears twice.

## Import Claude Code usage without routing

Claude Code can send subscription usage to Otari over OpenTelemetry. This is for
sessions that do not already route through Otari. It is the usage part of
[sending sessions to Otari](#send-claude-code-sessions-to-otari), without the
session's turns and tools. Send it to any Otari deployment.

Create a dedicated API key with `exclude_from_budget: true`, then configure the
logs exporter. The optional metrics exporter adds content-free outcome counters.

```bash
export CLAUDE_CODE_ENABLE_TELEMETRY=1
export OTEL_LOGS_EXPORTER=otlp
export OTEL_METRICS_EXPORTER=otlp
export OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
export OTEL_EXPORTER_OTLP_ENDPOINT="https://otari.example.com/otlp"
export OTEL_EXPORTER_OTLP_HEADERS="Authorization=Bearer tk-your-import-key"
claude
```

The endpoint is the Otari origin plus `/otlp`; the exporter appends `/v1/logs`
and `/v1/metrics`. Use an HTTP protocol because Otari does not accept OTLP over
gRPC.

Imported events are priced for analytics and never count toward budgets. Do not
both route and export one session, or its cost will appear twice. See
[Importing external usage](external-usage.md) for attribution, privacy,
idempotency, and pricing behavior.

## Backfill history from local transcripts

The exporter above carries sessions that run after it is configured. Everything
Claude Code did before that is already on disk, one JSONL transcript per session
under `~/.claude/projects`. `otari import claude-code` reads those transcripts and
posts them to the same import endpoint. The command installs on its own, without
the server: `brew install mozilla-ai/tap/otari`.

```bash
export OTARI_URL="https://otari.example.com"

# Preview what is on disk. No credential needed: a dry run sends nothing.
otari import claude-code --dry-run

# Then import it, with a budget-exempt API key. Usage binds to that key's own
# user, so do not pass --user-id here.
export OTARI_API_KEY="tk-your-import-key"
otari import claude-code
```

The key must be budget-exempt (`exclude_from_budget: true`), the same requirement
every import has.

The master key works too, and it is the only credential that can import on behalf
of somebody else. It requires `--user-id`, and that user must already exist
(create one with `POST /api/v1/users` first):

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
usage can be grouped per repository. Filter on it on the Overview.

Rows are unique on `(source, source_event_id)`, and the source event id here is the
Anthropic response id, so re-running imports only what is new and reports the rest
as duplicates. A single reply spans several transcript lines under one response id;
those collapse to the one API call they were.

The same warning applies as above, and it matters more for a backfill: do not
import sessions that were routed through Otari. Those are already recorded, and the
proxied and imported rows cannot be correlated, so the cost would count twice.

