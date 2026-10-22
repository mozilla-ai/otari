# Agent traces

Otari records an **agent trace** for every completion it serves: the whole
session an agent ran, the turns inside it, and inside each turn every request,
LLM call, routing attempt, guardrail check, MCP connection and tool call, with
their timing and cost. An LLM call is one kind of span among several, so a trace
answers what an agent run did and cost, not only what one call did.

No prompt, model output or tool content is recorded. A trace holds names, ids,
timings, token counts, outcomes and costs.

An LLM call's input tokens are its whole prompt, cached tokens included,
whichever way its provider reports them. Beside them the call keeps how many
input tokens were read from or written to the prompt cache, and how many output
tokens went to reasoning.

## What makes a session

Requests are grouped into one session only when the client names the session.
Otari never guesses that two requests belong together from their content.

| Signal | Example | Shown as |
| --- | --- | --- |
| The `Otari-Conversation-Id` header | `Otari-Conversation-Id: chat-42` | `client` |
| The `session_label` request field | `"session_label": "chat-42"` | `client` |
| A `session_id` request tag | `"metadata": {"session_id": "chat-42"}` | `client` |
| The agent harness's own session id | Claude Code's `x-claude-code-session-id`, sent on every request | `harness` |
| The agent harness's own session id | Codex's `thread-id`, or `session-id` without one, on a request whose `originator` or `User-Agent` starts with `codex` | `harness` |
| An OTLP span's Claude session | `anthropic.session.id` on an exported span | `otlp` |
| An OTLP span's conversation | `gen_ai.conversation.id` on an exported span | `otlp` |
| A Claude Code span's session | `session.id` on a span Claude Code exported | `otlp` |

A request that names no session is a trace of its own. To group a custom
client's requests, send one of the signals above.

A session belongs to the user whose API key sent it: two people who choose the
same label get two sessions, and a session id never reaches another workspace.

A gateway request's usage row records its session's trace id as `source_label`,
so Usage's spend by session, and `by_source_label` in `GET /api/v1/usage/summary`,
group the same sessions this page lists, whether or not traces are captured. The
id is the one a session's entry shows, so a row in one finds its match in the
other. A request that names no session reads as "(no session)" there. Usage
imported from outside the gateway keeps the label it was sent with.

## Sessions from your own agent

Claude Code and Codex are recognized on their own. An agent or harness of your own
needs two things on its requests to read well as a trace.

**Send one session id on every request of a run.** The `Otari-Conversation-Id`
header is the one to use: it works the same on every completion endpoint, and it
leaves `session_label` free for whatever you report spend by. Send it on each
request of the run, including the follow-ups that hand tool results back and any
retries, since a request without it starts a trace of its own.

- When more than one signal is present, the header wins, then `session_label`,
  then the `session_id` tag, then a harness's own id. A label you set per
  customer or per automation for spend therefore groups every run under it into
  one session, unless each run also sends its own header.
- The value is any string up to 512 characters. Otari hashes it into the trace
  id, so the id differs per workspace and user. When the value is
  identifier-shaped (up to 128 letters, digits and `_ . : / @ + -`, and not
  shaped like an email address, a file path or a credential), it is also stored
  as the session's name: the Sessions list shows it, and its search box finds
  the session by any part of it, in any case (`hackbot-1234`, `hackbot-` or
  `BOT-12`). The same box also matches any part of a session's id, agent or
  user. Any other value is not stored. A harness's own id and an OTLP session id are never
  stored as a name.
- Give each run its own id rather than reusing one for days: after
  `trace_session_max_age_days` a reused id starts a new trace anyway.
- Give a subagent an id of its own. Under its parent's id, the task it was handed
  reads as a new turn of the parent's session.

**Name the harness in `User-Agent`.** The first product token is shown as the
session's harness: `my-agent/1.4.0` shows `my-agent`. An SDK's default
(`OpenAI/Python 1.x`, `Anthropic/Python 0.x`) shows the SDK's name instead.

Turns and the tools your agent runs are read from each request's input, so they
need nothing extra when the request carries the conversation as usual: the
user's message last when a turn starts, the tool results last when the model's
tool calls are answered, each with the call id the model gave it. In a Messages
request, any `system` messages after them are skipped; Claude Code sends one on
almost every request. A Claude Code subagent sends its parent's session id and
an `x-claude-code-agent-id` of its own, and its requests open no turn, since it
works inside its parent's turn. A Responses request that continues with
`previous_response_id` carries only the new items, so its tools are recorded
without a name.

## Turns, steps and tools

- A **step** is one request to the gateway, named for the endpoint that served
  it, such as `/v1/chat/completions`. Every retry is a step of its own, with its
  own cost.
- A **turn** starts at a step whose last input is the user's own text, and
  includes the steps after it that hand tool results back to the model. A turn
  is **failed** when its last step failed, **incomplete** when the response was
  cut off, **active** while the session has been busy in the last ten minutes,
  and **completed** otherwise. A routing attempt that failed and was recovered by
  a fallback does not fail its turn.
- **Tools the gateway runs** (web search, web fetch, code execution, MCP tools)
  are timed as they run.
- **Tools the client runs** (a shell, a file read in a coding agent) are
  recorded from the request that carries their result: the tool's name, whether
  it returned an error, and when the result arrived. Their start is when the step before
  ended, so their duration is marked approximate.

## Sessions and failures

A session's error count is every span in it that ended in an error a fallback
did not recover: a failed request, a failed provider call, a guardrail failure,
a failed connection to an MCP server, and a tool that returned an error. The
session series (`GET /api/v1/traces/series`, the Sessions chart) counts a
session as **failed** only when one of those errors is not a tool's. A tool's
error, such as a missing file or a search that found nothing, goes back to the
model, which usually carries on, so a run whose requests all succeeded counts as
succeeded however many tool errors it recovered from. That includes an MCP
tool: a call the server answers with `isError`, or one that fails in transport,
reaches the model as a tool error too. Only a failed connection to an MCP
server fails the session. The error count, and the `has_error` filter, still
include every one of them.

## Instrumented agents

An agent that exports OpenTelemetry spans to `/otlp/v1/traces` (see
[Importing external usage](external-usage.md)) has every span kept on its
trace, including its own `invoke_agent` and `execute_tool` spans, with the span
tree it sent. Span names and attribute values are kept only when they are
identifier-shaped. A span's `gen_ai.response.finish_reasons` array is kept as
one value, its reasons joined by a comma: `stop,length`. The read API returns
each of these spans with the `otel_trace_id` and `otel_span_id` it was exported
with, in lowercase hex, so it can be joined back to the same trace in another
backend. A span the gateway records itself has neither.

### Claude Code without routing

Claude Code can export its own spans, so a session is traced without sending
its model traffic through Otari. Use a budget-exempt API key, as for
[importing its usage](use-with-claude-code.md#import-claude-code-usage-without-routing):

```bash
export CLAUDE_CODE_ENABLE_TELEMETRY=1
export CLAUDE_CODE_ENHANCED_TELEMETRY_BETA=1
export OTEL_TRACES_EXPORTER=otlp
export OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
export OTEL_EXPORTER_OTLP_ENDPOINT="https://otari.example.com/otlp"
export OTEL_EXPORTER_OTLP_HEADERS="Authorization=Bearer tk-your-import-key"
claude
```

Span export is a Claude Code beta. Each prompt becomes a turn, with its model
calls and the tools it ran. A tool call keeps a digest of its id, never the id
itself, and records who allowed it
or refused it: a tool a hook refused, such as an
[agent guardrail](agent-guardrails.md), is `blocked` with the decision source
`hook`. A blocked tool call is a guardrail doing its job, not an error, and does
not count toward a session's errors. Nor does a tool call that returned an
error, such as a command that exited non-zero: the agent reads that error like
any other result, and `grep` finding nothing or a test failing on purpose
returns one too. A tool the gateway runs itself (an MCP tool, web search, web
fetch or code execution) still records an error result as an error, as
[Sessions and failures](#sessions-and-failures) describes. A short `claude -p` run can exit before its last spans are
exported.

### Keeping what a span was about

A trace keeps no prompt, command or file path. A deployment can keep them by
setting `OTARI_TRACE_CONTENT_ENABLED=true`. They are stored encrypted with
`OTARI_SECRET_KEY`, the key that encrypts stored provider credentials, and are
not kept at all on a deployment without it. They are kept only when the agent
sends them:

- Claude Code sends the prompt that opened a turn with `OTEL_LOG_USER_PROMPTS=1`,
  and each tool's command or file path with `OTEL_LOG_TOOL_DETAILS=1`. Otari
  never keeps a tool's output, even when Claude Code exports it
  (`OTEL_LOG_TOOL_CONTENT=1`).
- `otari hook` sends the command or paths a gate checked, and the gate's own
  detail, with `OTARI_HOOK_REPORT_CONTENT=1`.

A span shows what it was about as its input, and a tool call is named for its
command or file in the tree. A span's content is sealed for that span alone, and
reads back empty once no configured key opens it, so keep a rotated-out key in
`OTARI_SECRET_KEY` for as long as you keep traces. Only the deployment operator, an
owner or admin of the organization, and the user a session belongs to read its
content; any other member reads the session with every span's content empty.

## Reading traces

| Endpoint | Who |
| --- | --- |
| `GET /api/v1/traces`, `/count`, `/series`, `/{trace_id}`, `/requests/{request_id}` | the deployment operator, every workspace of their active organization |
| `GET /api/v1/organizations/me/traces`, and the same four | a signed-in member: an owner or admin reads every workspace of their active organization, anyone else the workspaces they belong to |

`/series` counts the sessions active in each bucket of a window: a session is
active from its start to its last activity, so a long one counts in every bucket
it ran through. `bucket` is `1m`, `5m`, `30m`, `hour`, `6h` or `day`.
Every read takes the window the same way: `start` and `end` keep the sessions
active between them (last active at or after `start`, started before `end`), so a
session still running past `end` is listed in the window it was counted in.

A list read narrows to one workspace with `workspace_id`, inside the caller's
scope and never beyond it. `q` matches the start of a session's trace id or of
its name, case-sensitively, inside the same scope. A trace is keyed by its
workspace and its id, so a detail read takes the `workspace_id` its list entry
names; without it, an id in more than one of the caller's workspaces answers 409
rather than a guess.
`/requests/{request_id}` answers the session a gateway request was traced into,
by the `Otari-Request-ID` it was sent (a usage row's `request_group_id`), and 404
where the scope holds no trace of it. A trace outside the caller's scope answers
404.

Deleting a user erases their traces, as it erases their telemetry.

## Settings

| Setting | Default | |
| --- | --- | --- |
| `trace_capture_enabled` | `true` | Record traces and serve the read API. Requires restart. |
| `trace_retention_days` | `30` | Delete a trace once it has had no activity for this long. |
| `trace_session_max_age_days` | `90` | Delete a session's trace this long after it started, even while it is active, so a reused session id starts a new trace. |

Traces are written off the request path, within fixed limits: a trace never
slows or fails a request, and when the writer cannot keep up it drops whole
requests' spans rather than parts of them. One OTLP export adds at most
1,000 spans in at most 100 sessions. The
`gateway_trace_spans_dropped` metric counts what was dropped, by reason.

The cost on a trace is a snapshot of the usage rows' cost. Usage stays the
billing record.
