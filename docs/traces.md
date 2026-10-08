# Agent traces

Otari records an **agent trace** for every completion it serves: the whole
session an agent ran, the turns inside it, and inside each turn every request,
LLM call, routing attempt, guardrail check, MCP connection and tool call, with
their timing and cost. An LLM call is one kind of span among several, so a trace
answers what an agent run did and cost, not only what one call did.

By default no prompt, model output or tool content is recorded: a trace holds
names, ids, timings, token counts, outcomes and costs. A workspace admin can turn
content capture on for that workspace (see below).

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
identifier-shaped. OTLP/JSON trace and span ids are read as hex, as the
specification has them, and span times are clamped to the retention window.

One export's traces are bounded: at most 10,000 of its spans are kept, and spans
past the first 1,000 distinct sessions in one export are dropped; both are
counted in `gateway_trace_spans_dropped`. The export itself, and its usage, are
never refused for this. Spans sent with an API key that has no
user are grouped per key, so two such keys never share a session.

## Hybrid gateways

A hybrid data plane has no database of its own, so it files each request's trace
under the workspace and user its control plane's resolve answer names, and
records each provider attempt it reports upstream as an LLM call (without a
cost, which the control plane computes). A request whose resolve answer names
no workspace is not traced. Where the spans go is the deployment's choice: by
default a hybrid data plane binds a store that keeps nothing, and a deployment
that wants its data plane's traces binds its own `TraceStoragePort`. Content is
never captured on a hybrid gateway.

## Reading traces

| Endpoint | Who |
| --- | --- |
| `GET /api/v1/traces`, `/count`, `/series`, `/{trace_id}` | the deployment operator, every workspace |
| `GET /api/v1/organizations/me/traces`, and the same four | a signed-in member: an owner or admin reads every workspace of their active organization, anyone else the workspaces they belong to |
| `GET /api/v1/organizations/me/traces/{trace_id}/spans/{span_id}/content` | captured content: the session's own user, or an organization admin where the workspace allows it (see below) |
| `POST /api/v1/traces/{trace_id}/spans/{span_id}/content/break-glass` | captured content, for the deployment operator, with a recorded reason |

A trace outside the caller's scope answers 404. The list, `/count` and
`/series` take an optional `workspace_id`, which narrows the caller's scope to
one workspace and never widens it: a workspace outside the scope reads as empty.
The dashboard's Activity page passes the workspace selected in the sidebar, as
its request log does.

## Content capture

Content capture is **off in every workspace** until an admin of that workspace
turns it on there (`PUT /api/v1/workspaces/{workspace_id}/trace-settings`, or
**Tools > Trace content** in the dashboard). Turning it on in one workspace says
nothing about the others. Only owners and admins of the workspace may read or
change the setting or purge its content; a member is refused (403), and anyone
outside the organization finds no such workspace (404).

Capture is available only where the deployment permits it: the operator raises
`trace_content_capture_max` above `off` (its default) and configures content
encryption (below). Turning capture on is refused with 409 ("Content
encryption is not configured on this deployment") while the encryption backend
is unusable, rather than accepted and then silently keeping nothing.

| Level | Keeps |
| --- | --- |
| `off` | nothing (the default) |
| `tool_io` | each tool call's arguments and result, for tools the gateway runs and tools the agent runs |
| `full` | also each request's input and the model's output for that request |

- **Encrypted per session, before it is stored.** Each session's content is
  sealed with a data key of its own (AES-256-GCM), bound to the workspace and the
  session. Keys are stored only wrapped by `OTARI_SECRET_KEY` (or an AWS KMS key,
  `trace_content_key_backend: aws_kms`). Without a usable key, content is not
  kept.
- **Stored in the file store, not the database.** Sealed content is written
  through the deployment's file store (`files_backend`: a local directory by
  default, or S3); the database keeps only a reference per span and each
  session's wrapped key.
- **Kept for `trace_content_retention_days`** (7 by default), after which the
  stored content is deleted. When a session itself expires or is purged, its
  content and its key go with it, which makes any remaining copy unreadable
  through Otari. Copies in database or bucket backups remain until those backups
  expire.
- **Purge at any time** (`POST .../trace-settings/purge-content`): every span's
  stored content in the workspace, and every session key, are deleted. Traces
  stay. Other workers notice within 30 seconds; content one of them seals in
  that window cannot be read.
- **Readable by the session's own user.** Content is readable by the person
  whose key made the session, through
  `GET /api/v1/organizations/me/traces/{trace_id}/spans/{span_id}/content`.
  Anyone else who can see the session gets 403: they see its metadata, not what
  it said.
- **Organization admins only where the workspace allows it.** A workspace's
  admins can let the organization's owners and admins read its content
  (`admin_content_access` in the trace settings, off by default).
- **Platform operators only by breaking glass.** The deployment operator has no
  plain content read. `POST /api/v1/traces/{trace_id}/spans/{span_id}/content/break-glass`
  with a stated `reason` (10 to 500 characters, such as a legal request) reads
  one span's content.
- **Every read is recorded.** Each read writes a record of who read it (a
  dashboard user's id, or `master_key`), as what (`owner`, `admin` or
  `break_glass`), the session and span, and the reason for a break-glass read,
  before the content is returned. A workspace's admins list them with
  `GET /api/v1/workspaces/{workspace_id}/trace-settings/content-access`, or
  under **Tools > Trace content**. The records outlive the content they are about.
  While the key store cannot be reached, a read answers 503 and can be retried.
- **Turning it off takes effect within 30 seconds.** Each worker keeps
  a workspace's level for up to 30 seconds, so a request in that window may
  still be captured. Content already stored stays until retention or a purge.
- **Limited by** `trace_content_capture_max`: the most any workspace may keep.
  It never turns capture on.

## Settings

| Setting | Default | |
| --- | --- | --- |
| `trace_capture_enabled` | `true` | Record traces and serve the read API. Requires restart. |
| `trace_retention_days` | `30` | Delete a trace once it has had no activity for this long. |
| `trace_content_capture_max` | `off` | The most content any workspace may keep: `off`, `tool_io` or `full`. Only limits; capture stays off until the operator raises it. Requires restart. |
| `trace_content_retention_days` | `7` | Delete captured content, and its keys, after this long. |
| `trace_content_key_backend` | `secret_box` | `secret_box` (`OTARI_SECRET_KEY`) or `aws_kms`. |

Traces are written off the request path, within fixed limits: a trace never
slows or fails a request, and when the writer cannot keep up it drops whole
requests' spans rather than parts of them. The
`gateway_trace_spans_dropped` metric counts what was dropped, by reason.

The cost on a trace is a snapshot of the usage rows' cost. Usage stays the
billing record.
