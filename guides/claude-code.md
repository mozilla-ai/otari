# Claude Code

Claude Code speaks the Anthropic Messages API, which Otari serves. Point it at
Otari to put its requests under your budgets, usage tracking and traces.

## Route Claude Code through Otari

1. On the dashboard's **API keys** page, create a key. Under **Advanced**, set
   **Mismatched `user` field** to **Always accept**. Claude Code puts its own
   telemetry value in that field, and a key left on the default refuses its
   requests with `403`. Spend still goes to the key's user; see
   [Mismatched `user` fields](api-keys.md#mismatched-user-fields).
2. Set the environment and start Claude Code:

   ```bash
   export ANTHROPIC_BASE_URL="https://api.otari.ai/api"
   export ANTHROPIC_AUTH_TOKEN="<your-otari-key>"
   export ANTHROPIC_MODEL="anthropic:claude-sonnet-4-6"
   claude
   ```

`ANTHROPIC_BASE_URL` ends in `/api`, not `/api/v1`, because Claude Code adds
`/v1/messages` itself. Accounts in otari.ai's EU region use
`https://eu.api.otari.ai/api`, and a gateway you run uses its own URL plus
`/api`, such as `http://localhost:8000/api`; see
[Base URL](https://docs.otari.ai/reference#base-url).

- The same variables can go in the `env` block of Claude Code's settings file.
- `ANTHROPIC_API_KEY` works in place of `ANTHROPIC_AUTH_TOKEN`.
- A workload on Google Cloud, or behind another OpenID Connect issuer, can use
  Claude Code's workload identity instead of a key; see
  [Claude Code and the Anthropic SDKs](https://docs.otari.ai/self-hosting/operations/workload-identity#claude-code-and-the-anthropic-sdks).

## Tag Claude Code traffic

Claude Code can add headers to every request through
`ANTHROPIC_CUSTOM_HEADERS`. Use it to tag a session's spend with `Otari-Tags`,
so one key can split its cost by agent, run or requester:

```bash
export ANTHROPIC_CUSTOM_HEADERS="Otari-Tags: agent=bug-fix,run=1234"
claude
```

Usage can then be filtered and grouped by those tags. Percent-encode a `,` or
`=` inside a value. See [Request tags](https://docs.otari.ai/reference#request-tags).

Two more headers are useful:

- `Otari-Conversation-Id` makes a headless run its own session in the
  dashboard's **Sessions** list, in place of Claude Code's own session id. See
  [Sessions from your own agent](sessions.md#collect-sessions).
- `Otari-Trace-Content: keep` keeps each turn's prompt, tool arguments and, where
  allowed, tool output, if the deployment and the workspace keep content. See
  [Keeping what a span was about](sessions.md#keep-what-a-session-was-about).

Setting `ANTHROPIC_CUSTOM_HEADERS` again replaces it, so put several headers in
one value, one per line:

```bash
export ANTHROPIC_CUSTOM_HEADERS=$'Otari-Conversation-Id: bug-fix-1234\nOtari-Tags: agent=bug-fix,run=1234'
```

## Budgets

Claude Code prefixes every `403` with "Failed to authenticate", so a spent
budget reads:

```text
Failed to authenticate. API Error: 403 User 'alice' has exceeded budget limit
```

The key is fine; the text after the prefix is the reason. Claude Code does not
retry it.

Otari holds a request's worst-case cost against the budget before it runs, so a
call can be refused while some budget is left. Claude Code asks for up to 32,000
output tokens on a model it does not know, which is a large hold on an expensive
model. When budgets are small, set `CLAUDE_CODE_MAX_OUTPUT_TOKENS` lower. See
[Budgets](budgets.md).

## Choosing a model

Otari can serve Claude Code's requests from non-Anthropic models. Set Claude
Code's default Opus, Sonnet and Haiku model variables when its built-in aliases
do not exist on your deployment.

A non-Claude model may lose Anthropic-specific behavior such as extended
thinking, prompt caching or tool semantics, and Anthropic's server-side context
editing does not run (Claude Code's own compaction still does). Test your actual
workflow before making one a default.

To list only the open-weight models a key can use, ask the catalog for them:
`GET /api/v1/catalog/models?capability=open_weights`. Each entry's `selectors`
are the names to configure.

On a gateway you run, OpenAI reasoning models work better through the Responses
API: their reasoning carries over between turns, and the codex and pro models
become reachable. Set `messages_api: responses` on the OpenAI provider; see
[Messages through the Responses API](https://docs.otari.ai/self-hosting/configuration#messages-through-the-responses-api).

## Let Otari choose a subagent's model

The `otari-router` plugin has Otari pick the model each new Claude Code subagent
starts on. A new subagent has no prompt cache yet, so starting it on a cheaper
model costs nothing in cache misses. The plugin works even when Claude Code's
own requests go straight to Anthropic.

Each recommendation is one decision call billed to your key. On otari.ai a
managed router answers. On a gateway you run, an operator sets one up on the
**Subagent router** page; see [Subagent routers](https://docs.otari.ai/reference/gateway/routing#subagent-routers).
How Otari decides is in
[Agent model recommendations](https://docs.otari.ai/reference/gateway/routing#agent-model-recommendations).

### Install the plugin

In a Claude Code session:

```
/plugin marketplace add mozilla-ai/otari
/plugin install otari-router@otari
```

The install asks for the gateway URL and an API key. The key goes to Claude
Code's secure storage. The URL must be `https` unless the gateway runs on the
same machine. The plugin stays active in every session after that; its
[README](https://github.com/mozilla-ai/otari/blob/2e5d930796c0200e72bb33749d47858c133e7ff1/plugins/otari-router/README.md) has the options and how to update.

Each spawn then leaves a dim line in the transcript naming the model Otari chose
and why. If Otari does not answer, the subagent starts on the model it would
have had anyway, and the line says so. Forks always inherit their parent's
model.

## Send Claude Code sessions to Otari

Otari can record each Claude Code session without routing its model traffic:
every prompt as a turn, with its model requests, tool calls and cost. Claude
Code exports this over OpenTelemetry.

This works on a gateway you run, not on otari.ai today. It needs a
budget-exempt API key, which only the deployment's operator can create (with
**Exempt from budget** under **Advanced** in the key form; see
[Create an import key](external-usage.md#create-an-import-key)). To see Claude Code
sessions on otari.ai, [route them through Otari](#route-claude-code-through-otari)
instead.

1. Have the operator create a budget-exempt key. Traces must be on
   (`OTARI_TRACE_CAPTURE_ENABLED`, the default).
2. Add the variables to the `env` block of `~/.claude/settings.json`, with your
   gateway's URL:

   ```json
   {
     "env": {
       "CLAUDE_CODE_ENABLE_TELEMETRY": "1",
       "CLAUDE_CODE_ENHANCED_TELEMETRY_BETA": "1",
       "OTEL_TRACES_EXPORTER": "otlp",
       "OTEL_LOGS_EXPORTER": "otlp",
       "OTEL_EXPORTER_OTLP_PROTOCOL": "http/protobuf",
       "OTEL_EXPORTER_OTLP_ENDPOINT": "http://localhost:8000/otlp",
       "OTEL_EXPORTER_OTLP_HEADERS": "Authorization=Bearer tk-your-import-key"
     }
   }
   ```

3. Run Claude Code, ask for something that uses a tool or two, and open
   **Sessions** in the dashboard. A short `claude -p` run can exit before its
   last spans are sent.

| Variables | What they send |
| --- | --- |
| `CLAUDE_CODE_ENABLE_TELEMETRY`, `CLAUDE_CODE_ENHANCED_TELEMETRY_BETA`, `OTEL_TRACES_EXPORTER` | Turns, model requests and tool calls (a Claude Code beta) |
| `OTEL_LOGS_EXPORTER` | Each request's usage, which prices the session; without it there is no cost |
| `OTEL_EXPORTER_OTLP_*` | Where both go: your Otari URL plus `/otlp`, over HTTP (not gRPC), with the key |

To keep the key out of the settings file, Claude Code's `otelHeadersHelper`
(see its [monitoring docs](https://code.claude.com/docs/en/monitoring-usage))
can replace `OTEL_EXPORTER_OTLP_HEADERS`.

By default no span records what it was about (no prompt, command or file path).
To keep that, the deployment needs `OTARI_TRACE_CONTENT_ENABLED=true` and
`OTARI_SECRET_KEY`, and Claude Code must send it:

```json
{
  "env": {
    "OTEL_LOG_USER_PROMPTS": "1",
    "OTEL_LOG_TOOL_DETAILS": "1"
  }
}
```

See [Keeping what a span was about](sessions.md#keep-what-a-session-was-about).

Do not both route and export the same session: each turn and its cost would
appear twice.

## Import Claude Code usage without routing

To record only usage and cost, without turns and tools, export just the logs.
The same limit applies: it needs a budget-exempt key, so it works on a gateway
you run, not on otari.ai today.

```bash
export CLAUDE_CODE_ENABLE_TELEMETRY=1
export OTEL_LOGS_EXPORTER=otlp
export OTEL_METRICS_EXPORTER=otlp
export OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:8000/otlp"
export OTEL_EXPORTER_OTLP_HEADERS="Authorization=Bearer tk-your-import-key"
claude
```

The metrics exporter is optional and adds content-free outcome counters.
Imported usage is priced for analytics and never counts toward budgets. Do not
export a session you also route. See
[Import usage from outside Otari](external-usage.md).

## Backfill history from local transcripts

Claude Code keeps one transcript per session under `~/.claude/projects` (or
under `CLAUDE_CONFIG_DIR`, if set). `otari import claude-code` imports them, so
history from before you set up the exporter gets in too. It needs a
budget-exempt key or the master key, so it works on a gateway you run, not on
otari.ai today.

Install the command on its own with `brew install mozilla-ai/tap/otari`, then:

```bash
export OTARI_URL="http://localhost:8000"   # your gateway's URL

# Preview what is on disk. No credential needed: a dry run sends nothing.
otari import claude-code --dry-run

# Then import it, with a budget-exempt API key. Usage binds to that key's own
# user, so do not pass --user-id here.
export OTARI_API_KEY="tk-your-import-key"
otari import claude-code
```

- `--since 7d` (or an ISO date) reads only recently modified transcripts.
- Re-running imports only what is new.
- Only token counts and identifiers are read, never prompts, completions or tool
  payloads.
- Each session is labeled `<hostname>:<project>`, so you can group usage per
  repository on the **Overview**.
- The first event is sent on its own, so a rejection (an unknown user, a key that
  is not budget-exempt) shows before the rest is uploaded.

To import on someone else's behalf, use the master key with `--user-id`. The
user must already exist. An ordinary API key always imports for its own user.

```bash
export OTARI_MASTER_KEY="your-master-key"
otari import claude-code --user-id alice
```

Do not import sessions that were routed through Otari. They are already
recorded, and their cost would count twice.
