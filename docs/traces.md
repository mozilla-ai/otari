# Agent traces

Otari records an **agent trace** for every completion it serves: the whole
session an agent ran, the turns inside it, and inside each turn every request,
LLM call, routing attempt, guardrail check, MCP connection and tool call, with
their timing and cost. An LLM call is one kind of span among several, so a trace
answers what an agent run did and cost, not only what one call did.

No prompt, model output or tool content is recorded. A trace holds names, ids,
timings, token counts, outcomes and costs.

## What makes a session

Requests are grouped into one session only when the client names the session.
Otari never guesses that two requests belong together from their content.

| Signal | Example | Shown as |
| --- | --- | --- |
| The `Otari-Conversation-Id` header | `Otari-Conversation-Id: chat-42` | `client` |
| The `session_label` request field | `"session_label": "chat-42"` | `client` |
| A `session_id` request tag | `"metadata": {"session_id": "chat-42"}` | `client` |
| The agent harness's own session id | Claude Code's `x-claude-code-session-id`, sent on every request | `harness` |
| An OTLP span's conversation | `gen_ai.conversation.id` on an exported span | `otlp` |

A request that names no session is a trace of its own. To group a custom
client's requests, send one of the signals above.

A session belongs to the user whose API key sent it: two people who choose the
same label get two sessions, and a session id never reaches another workspace.

## Turns, steps and tools

- A **step** is one request to the gateway. Every retry is a step of its own,
  with its own cost.
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
  it failed, and when the result arrived. Their start is when the step before
  ended, so their duration is marked approximate.

## Instrumented agents

An agent that exports OpenTelemetry spans to `/otlp/v1/traces` (see
[Importing external usage](external-usage.md)) has every span kept on its
trace, including its own `invoke_agent` and `execute_tool` spans, with the span
tree it sent. Span names and attribute values are kept only when they are
identifier-shaped.

## Reading traces

| Endpoint | Who |
| --- | --- |
| `GET /api/v1/traces`, `/count`, `/series`, `/{trace_id}` | the deployment operator, every workspace |
| `GET /api/v1/organizations/me/traces`, and the same three | a signed-in member: an owner or admin reads every workspace of their active organization, anyone else the workspaces they belong to |

A list read narrows to one workspace with `workspace_id`, inside the caller's
scope and never beyond it. A trace is keyed by its workspace and its id, so a
detail read takes the `workspace_id` its list entry names; without it, an id in
more than one of the caller's workspaces answers 409 rather than a guess. A
trace outside the caller's scope answers 404.

Deleting a user erases their traces, as it erases their telemetry.

A hybrid data plane records no traces yet: it holds no database to keep them in.

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
