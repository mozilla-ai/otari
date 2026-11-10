# Agent traces

Otari records an **agent trace** for the completions it serves. Capture is
best-effort: it can be turned off, and spans are dropped when the writer is
full (see [Settings](https://docs.otari.ai/self-hosting/operations/trace-storage#settings)). A trace is the
whole session an agent ran: its turns, and inside each turn every request, LLM
call, routing attempt, guardrail check, MCP connection and tool call, with their
timing and cost. It answers what an agent run did and cost, not only what one
call did.

A trace holds names, ids, timings, token counts, sizes, outcomes and costs. It
records no model or tool output, and no prompt, command or file path unless a
workspace chooses to keep them (see
[Keeping what a span was about](#keeping-what-a-span-was-about)).

## Reading traces

Open **Overview** and pick the **Sessions** tab. It lists sessions in the
selected time range and applies the user, API key, model and status filters the
other tabs share. The tab is offered where the deployment records traces, which
is the default.

- Search finds a session by any part of its name, id, agent or user.
- The chart counts sessions per hour or day as succeeded or failed.
- Click a session to open it in a side panel, as a tree of its turns, LLM calls
  and tool calls, as a timeline, or as a flat log.
- From the **Requests** tab, **View trace** in a request's detail opens its
  session on that LLM call.

The same reads are on the API: see [Agent traces](https://docs.otari.ai/reference#agent-traces)
in the API reference.

### At a glance

Before you pick a span, a session's panel opens on **At a glance**: how long the
session took and what it cost, each split into parts, then what filled the
model's context window, then the gates its hooks ran and what they decided.

- **Time** splits the session into model requests, hooks, running tools, waiting
  for you to approve a call, and idle time between turns. Long idle time is drawn
  shorter, marked `//`, with the true figure beside it.
- **Cost** splits the total by meter (fresh input, output, cache reads, cache
  writes) when requests were priced by the token, and shows what the cache saved.
  Requests priced whole, as a routed request is, have no split. A session with
  nothing priced says so rather than showing $0.00.
- **Cost by tool** estimates what each tool's calls cost, with a subagent's calls
  listed under the call that ran it. Use it before moving a tool's work, or a
  subagent, to a smaller model.

A tool call's cost is an estimate, marked `≈`, because providers bill requests,
not tool calls. It adds up the tool's share of the output that called it, how
much its result grew the next request's context, and the cost of re-reading that
result on later requests until the context is compacted. Calls made together
share the estimate equally. A subagent's own requests are counted exactly.

### The context map

The context map shows what fills the model's context window, as blocks sized by
tokens, with the unused part as **Free**. It reads the session's last LLM call,
and also names the fullest call when that was an earlier one. Hover, focus or tap
a block to see its tokens and share of the window.

- **Every call** is split into cache reads, cache writes and fresh input.
- **A request Otari served itself** is also split into system prompt, tool
  definitions, files, messages and tool results. That split is an estimate
  (Otari counts characters, then scales them to the exact token count), and the
  map says so. An image, audio clip or document counts as about 1,000 tokens.

When the model is not in the catalog, the window size is unknown and the map
shows no free space. To build the estimate a span stores eight numbers, never
text, as `otari.context.*` attributes (characters per part, tool count and
message count).

## What makes a session

Requests are grouped into one session only when the client names the session.
Otari never guesses from content.

| Signal | Example | Shown as |
| --- | --- | --- |
| The `Otari-Conversation-Id` header | `Otari-Conversation-Id: chat-42` | `client` |
| The `session_label` request field | `"session_label": "chat-42"` | `client` |
| A `session_id` request tag | `"metadata": {"session_id": "chat-42"}` | `client` |
| Claude Code's own session id | `x-claude-code-session-id`, sent on every request | `harness` |
| Codex's own session id | `thread-id`, or `session-id` without one, when `originator` or `User-Agent` starts with `codex` | `harness` |
| A Claude Code span's session | `session.id` on an exported span | `otlp` |
| An OTLP span's Claude session | `anthropic.session.id` on an exported span | `otlp` |
| An OTLP span's conversation | `gen_ai.conversation.id` on an exported span | `otlp` |
| An OTLP span's Otari label | `otari.user_session_label`, else `otari.session_label`, on an exported span | `otlp` |

- A request that names no session is a trace of its own.
- When a request carries more than one signal, the first row in the table wins.
- An exported span that names no session joins its nearest ancestor's session,
  then the session another span in its OTel trace names (within the same export),
  and otherwise is grouped by its OTel trace.
- A Claude Code session routed through Otari and the spans exported for it land
  on one trace, because they share an id. Its source is whichever reached Otari
  first, so filtering by `harness` or `otlp` alone can miss it.
- A session belongs to the user whose API key sent it. Two people using the same
  label get two sessions, and a session never crosses workspaces.

Usage groups spend by the same sessions: a request's usage row records its
session's trace id as `source_label`, so a session in one view finds its match in
the other. A request with no session reads as "(no session)".

## Sessions from your own agent

Claude Code and Codex are recognized automatically. For your own agent:

1. **Send one `Otari-Conversation-Id` on every request of a run**, including the
   follow-ups that return tool results, and retries. A request without it starts
   a new trace. The header leaves `session_label` free for reporting spend.
2. **Give each run its own id.** A reused id starts a new trace anyway once the
   session reaches its maximum age (90 days by default, see
   [Settings](https://docs.otari.ai/self-hosting/operations/trace-storage#settings)).
3. **Give each subagent its own id.** Otherwise its task reads as a new turn of
   the parent's session.
4. **Name your agent in `User-Agent`.** The first product token is shown as the
   session's harness: `my-agent/1.4.0` shows `my-agent`. An SDK's default shows
   the SDK's name.

The id is any string up to 512 characters. When it looks like an identifier (up
to 128 letters, digits and `_ . : / @ + -`, and not like an email address, a file
path or a credential), it is also stored as the session's name and is
searchable. Any other value only feeds the trace id and is not stored.

Turns and client-run tools need nothing extra when the request carries the
conversation as usual: the user's message last when a turn starts, and tool
results last, with the call ids the model gave them. A Responses request that
continues with `previous_response_id` carries only new items, so its tools are
recorded without a name.

## Turns, steps and tools

- A **step** is one request to the gateway, named for its endpoint, such as
  `/v1/chat/completions`. Each retry is its own step, with its own cost.
- A **turn** starts at a step whose last input is the user's text, and includes
  the steps that hand tool results back. A turn is **failed** when its last step
  failed, **incomplete** when the response was cut off, **active** while the
  session was busy in the last ten minutes, and **completed** otherwise. A failed
  attempt that a fallback recovered does not fail the turn.
- **Tools the gateway runs** (web search, web fetch, code execution, MCP tools)
  are timed as they run. An MCP tool called through
  [caller-orchestrated MCP](https://docs.otari.ai/reference/gateway/mcp#execution) also names the access rule that
  allowed or refused it.
- **Tools the client runs** (a shell, a file read) are recorded from the request
  that carries their result: name, whether it returned an error, and when. Their
  duration is approximate.

## Sessions and failures

A session's error count includes every error a fallback did not recover: failed
requests and provider calls, guardrail failures, failed MCP connections, and tool
errors.

A session counts as **failed** only when one of those errors is not a tool's. A
tool error, such as a missing file or an empty search, goes back to the model,
which usually carries on. An MCP tool that answers with an error, or fails in
transport, counts as a tool error; a failed connection to the MCP server fails
the session. The error count and the error filter still include tool errors.

## Instrumented agents

An agent that exports OpenTelemetry spans to Otari (see
[Importing external usage](external-usage.md#opentelemetry)) has its spans kept on
the trace of the exporting key's user, with the span tree it sent, within the
limits under [Settings](https://docs.otari.ai/self-hosting/operations/trace-storage#settings). Its sessions are grouped by
the OTLP attributes in [What makes a session](#what-makes-a-session). Span names
and attribute values are kept only when they look like identifiers.

Each LLM call a span reports is priced like imported usage and counted in the
session's cost. When two instrumentations report the same call (the same
`gen_ai.response.id`), both spans are kept but the cost counts once. Exported
spans keep their `otel_trace_id` and `otel_span_id`, so you can join them to the
same trace in another backend.

### Claude Code without routing

Claude Code can export its own spans, so a session is traced, guardrail checks
included, without routing its model traffic through Otari. Set it up with
[Send Claude Code sessions to Otari](use-with-claude-code.md#send-claude-code-sessions-to-otari).

- Each prompt becomes a turn, with its model calls and tool calls.
- A tool call a hook refused, such as an [agent guardrail](agent-guardrails.md),
  shows as `blocked`. Blocked calls and tool errors (a command exiting non-zero,
  a `grep` finding nothing) do not count as session errors.
- Cost comes from Claude Code's exported logs, priced at the deployment's API
  rates. Without exported logs a session has tokens but no cost. On a Claude
  subscription, this is what the requests are worth at API prices, not what was
  charged.
- A judge gate that [reports its model calls](https://docs.otari.ai/reference/agent-guardrails/agent-guardrails-reference#what-otari-hook-reports)
  is priced too, counted in the session's cost but not in any turn's, and
  **At a glance** shows what judge gates cost in all.

## Keeping what a span was about

By default a trace keeps no prompt, command or file path. A workspace can keep
them where the deployment allows it; otherwise the setting says so and cannot be
turned on. On a deployment you run, [Store agent traces](https://docs.otari.ai/self-hosting/operations/trace-storage) says
how to allow it. Content is stored encrypted.

### Turn it on for a workspace

A workspace owner or admin, or an organization owner or admin:

1. Open **Workspaces** and click the workspace.
2. On its **General** tab, turn on **Keep what agent sessions were about**.
3. Where the deployment also keeps tool output, **Keep what tools returned** is
   shown too, on by default. Turn it off to keep prompts but not tool output.

Turning content off keeps nothing new from then on. Deleting content removes what
is already stored and keeps the sessions. Over the API, these are
`PATCH /api/v1/workspaces/{id}` with `trace_content_enabled` or
`trace_content_tool_output`, and `DELETE /api/v1/workspaces/{id}/trace-content`.

Content is kept only for sessions sent with an API key that belongs to a current
member of the workspace (or an owner or admin of the organization). Mint tracing
keys for real members: a key with no user, or one minted for a billing user with
no dashboard identity, keeps no content.

### What the client must send

The workspace setting is not enough on its own; the client has to send content
too:

- **Requests routed through Otari** send the header `Otari-Trace-Content: keep`.
  Otari then keeps the prompt that opened each turn, as the client sent it
  (including anything the harness added, such as Claude Code's
  `<system-reminder>` blocks), and the arguments of each tool call. It never
  keeps the replayed history, the system prompt or attachments. Any other header
  value is refused with a 400. The client answers for its users' consent.
- **Claude Code exporting spans** sends prompts with `OTEL_LOG_USER_PROMPTS=1`,
  and each tool's command or file path with `OTEL_LOG_TOOL_DETAILS=1`.
- **`otari hook`** sends what a gate checked with `OTARI_HOOK_REPORT_CONTENT=1`.

What a tool returned is kept only from routed requests: the results the client
hands back, and the output of tools Otari ran itself. Otari never keeps tool
output exported over OTLP, nor a model's own output.

A span shows what it was about as its input and what a tool returned as its
output. Content that no key the deployment holds can open reads back empty (see
[How content is sealed](https://docs.otari.ai/self-hosting/operations/trace-storage#how-content-is-sealed)).

### Who reads content

A session's content is private to the user it belongs to. Other members, admins,
organization owners and the deployment's operator still see the session (its
tools, models, costs, outcomes and guardrail verdicts) but not what it was about.

| Reader | Reads a session's content |
| --- | --- |
| The session's own user | While they are a member of the workspace. |
| An organization owner or admin | When the user shared it with the organization's admins. |
| The deployment's operator | When the user shared it with the operator. |
| An organization owner, by break-glass | Any of the organization's sessions, recorded with a reason. |
| The deployment's operator, by break-glass | Any session, where the deployment enables it, recorded with a reason and a reference. |
| Anyone else | Never. |

**Sharing a session.** On a session you own, click **Share this session**,
choose who to share with (your organization's owners and admins, or the team
that runs the deployment), and add an optional note. A session has one running
share per audience: sharing again with the same audience replaces the note on
the running share. The share lasts until you end it, until you lose access to
the workspace, or until retention deletes the session. You see who read it and
when; reads are recorded at most once an hour per reader, and the record
outlives the share. A share with the operator of a deployment that serves
several organizations, such as otari.ai, needs the operator to act in your
organization.

**Break-glass.** For an investigation the owner cannot be asked to share for,
such as an HR or legal matter, an organization owner (not an admin) can click
**Open by break-glass** on the session and give a reason of at least ten
characters. Where the deployment enables it
([`trace_platform_break_glass_enabled`](https://docs.otari.ai/self-hosting/configuration#common-settings), off by
default), the operator can do the same with a reason and a reference. Each
opening:

- is recorded before any content opens, with who, which session, when and why;
- is listed under **Break-glass openings** on the Sessions tab for the
  organization's owners (the operator's openings are listed only on the
  operator's side, since a legal request may forbid telling the organization);
- with AWS KMS, also shows in CloudTrail as a `Decrypt` naming the session;
- is not told to the session's owner.

These records are never deleted, even with the organization. Erasing a user
removes their id from the records of openings of their sessions.

**When access changes.**

- A member who leaves the workspace or organization ends their shares for good.
  What was stored stays until retention deletes it, and only break-glass opens it.
- Deactivating an account ends its shares, and reactivating it restores none.
  While deactivated, nothing new is kept for its sessions.
- The master key reads no session's content and cannot break glass, since it
  names no person to record. Sign in to read what your account may.

This is an access rule the gateway enforces, not end-to-end encryption. Whoever
runs the gateway holds the key, so a self-hosted operator could read content by
changing the code or the database. What it guarantees is that nobody reads
content through Otari without the owner's say or a recorded break-glass, and
that the database or a backup alone shows none.

The share, break-glass and audit routes are under
`/api/v1/organizations/me/traces/{trace_id}/shares`,
`/api/v1/organizations/me/traces/{trace_id}/break-glass` and
`/api/v1/organizations/me/traces/audit/break-glass`, with operator equivalents
under `/api/v1/traces`. `GET /api/v1/trace-content/policy` tells a signed-in
caller whether the deployment keeps content and tool output, and for how long.

Deleting a user erases their traces, as it erases their telemetry.
