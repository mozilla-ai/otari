# Agent traces

Otari records an **agent trace** for every completion it serves: the whole
session an agent ran, the turns inside it, and inside each turn every request,
LLM call, routing attempt, guardrail check, MCP connection and tool call, with
their timing and cost. An LLM call is one kind of span among several, so a trace
answers what an agent run did and cost, not only what one call did.

No prompt, model output or tool content is recorded, unless a workspace keeps
what its sessions were about ([Keeping what a span was about](#keeping-what-a-span-was-about)).
A trace holds names, ids, timings, token counts, sizes, outcomes and costs. The
sizes are numbers only, described under [The context map](#the-context-map).

An LLM call's input tokens are its whole prompt, cached tokens included,
whichever way its provider reports them. Beside them the call keeps how many
input tokens were read from or written to the prompt cache, and how many output
tokens went to reasoning.

## The context map

A session's **At a glance**, shown before a span is picked, maps what fills the
model's context window, as blocks sized by tokens, with the unused part of the
window as **Free**. It is read from the session's last LLM call that recorded
input, since each call carries the whole conversation so far, and it also names
the fullest call when that was an earlier one. Hover, focus or tap a block for its
tokens and share of the window. The window comes from the model catalog when the
model is listed there. Without it the map shows no free space and says the
window is unknown. The total is always the provider's exact input-token count.
How it is split depends on what the call recorded:

- **Every call**, including those a coding agent exports over OTLP, is split into
  cache reads, cache writes and fresh input, from the cache counts the call
  already keeps.
- **A request Otari served itself** (chat completions, Messages or Responses) is
  also split into the system prompt, tool definitions, files, messages and tool
  results. Otari has no tokenizer, so it counts characters per part as the
  request arrives, and the dashboard scales those weights so they add up to the
  exact input tokens. The split is an estimate and the map says so. An image,
  audio clip, file or document counts for a flat allowance of 4,000 characters
  (about 1,000 tokens), because those are billed by size and format rather than
  by bytes. Tokens the provider adds itself, such as a hidden tool prompt, are
  spread across the parts.

For that estimate a span keeps eight whole numbers, never text: the characters
in the system prompt, the tool definitions, the user messages, the assistant
messages, the tool results and the files, and the number of tool definitions and
of messages. They are stored as the span attributes
`otari.context.system_chars`, `otari.context.tools_chars`,
`otari.context.user_chars`, `otari.context.assistant_chars`,
`otari.context.tool_result_chars`, `otari.context.file_chars`,
`otari.context.tool_count` and `otari.context.message_count`. Every attempt of a
request, a fallback included, carries the same numbers, and turning trace
capture off stores none. A request that runs a tool loop inside the gateway
reports the sizes of the request as it arrived, not of each later round.

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
its model traffic through Otari. [Send Claude Code sessions to Otari](use-with-claude-code.md#send-claude-code-sessions-to-otari)
has the whole setup, guardrail checks included. In short, with a budget-exempt
API key:

```bash
export CLAUDE_CODE_ENABLE_TELEMETRY=1
export CLAUDE_CODE_ENHANCED_TELEMETRY_BETA=1
export OTEL_TRACES_EXPORTER=otlp
export OTEL_LOGS_EXPORTER=otlp
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

A session's cost comes from its logs, not its spans: Otari prices each model
request's `api_request` event at the deployment's rates, cache reads and writes
included, and shows the cost on the request and its session. Without
`OTEL_LOGS_EXPORTER=otlp` a session has its tokens but no cost. The cost is
what the requests are worth at API prices, so on a Claude subscription it is
not what was charged.

A session's header counts its tool calls and the gates its hooks ran, each
broken down on hover: the calls by tool, and the gates by what they decided.
**At a glance** opens a session with where its time and money went: how long
it took, with its turns and tool calls, and what it cost, with its requests and
tokens, each above one bar and a figure per part. Below them, what fills the
model's context window (see [The context map](#the-context-map)), then the gates
its hooks ran and what they decided.

- **Time** splits the session into model requests, hooks, running tools,
  waiting for you to approve a call, and idle between turns. When idle is more
  than half the session, the bar draws it shorter, marked `//`, and its figure
  keeps the true time. Running tools opens onto each tool's calls and
  running time. A call that started a subagent runs until the subagent's last
  span, so its time also counts as model time, and the panel says when parts
  overlap like that. Tools that ran inside the request that called them, as a
  gateway's built-in web search and code execution do, get a lane of their own
  under the model's part. Each gate is timed by its own run, so a slow judge
  gate shows on its own.
- **Cost** is the session's total, split by meter (fresh input, output, cache
  reads and cache writes) when the requests were priced by the token, with what
  reading from the cache saved over paying for those tokens as fresh input.
  Requests priced whole, as a gateway prices a routed request, have no split.
  A session with nothing priced says so rather than $0.00.

A judge gate that [reports its model calls](agent-guardrails.md#reporting-to-otari)
is priced like a request and counted in the session's cost, though not in any
turn's. Its check shows the model, the tokens and the cost, and **At a glance**
says what the judge gates cost in all.

A turn shows what its requests cost. A tool call shows an estimate, marked `≈`,
because a provider bills requests, not tool calls. It is read off the requests
either side of the call, in the context the call ran in (a subagent has its
own):

- **Calling it:** its share of the output of the request that called it.
- **Adding its result:** how much the next request's context grew, at what that
  request paid for new input, usually a cache write.
- **Re-reading it:** the result again on every later request, at the cache-read
  rate, or the cache-write rate once the cache expired, until the context is
  compacted.

Calls made together share these equally, since a request reports how much its
context grew, not how much of that each result was. An `Agent` call also carries
what its subagent's requests cost, which is exact. The estimate assumes one
main model per context, so in a session that switches models with `/model`
partway through, the calls around the switch get rougher estimates.

**Cost by tool** adds the estimates up for the session: each tool's calls and
cost, and its share of the session. A subagent's calls are listed under the
call that ran it, with the models the subagent used and what its own reasoning
and answers cost, and are counted there once. What judge gates cost is listed
apart. What no tool call accounts for, such as the prompts and the answers, is
listed last. It is where to look before
moving a tool's work, or a subagent, to a smaller model.

### Keeping what a span was about

A trace keeps no prompt, command or file path. A deployment can keep them by
setting `OTARI_TRACE_CONTENT_ENABLED=true`. They are stored encrypted, under
`OTARI_SECRET_KEY` (the key that encrypts stored provider credentials) by
default or under an AWS KMS key, and are not kept at all on a deployment with
neither.

Each workspace then chooses for itself. Content is off in every workspace until
one of its managers (a workspace owner or admin, or an organization owner or
admin) turns on **Keep what agent sessions were about** in the workspace's
settings, or sends `PATCH /api/v1/workspaces/{id}` with
`{"trace_content_enabled": true}`. Turning it off keeps nothing new from the
next export on; `DELETE /api/v1/workspaces/{id}/trace-content` also deletes what
is already stored, and the sessions stay. Either one also covers content the
gateway accepted before and had not stored yet, on every replica: it is never
stored. `GET /api/v1/trace-content/policy`, for any signed-in caller, reports
`content_available` (whether the deployment allows content at all),
`platform_break_glass`, and the retention a session's content is kept for. It is
not on the public bootstrap: whether a deployment keeps prompts, and whether its
operator can open them, is nothing to tell an anonymous visitor.

Content is kept only for a session a current member of the workspace owns: the
API key that exports it belongs to an active owner or admin of the organization,
or an active member of the workspace. A key with no user, or minted for a
billing user no dashboard identity stands behind (as `POST /api/v1/keys` mints
for any `user_id`), keeps no content, because nobody could ever read it. Mint
tracing keys for real members.

Content is kept only when the agent sends it too:

- Claude Code sends the prompt that opened a turn with `OTEL_LOG_USER_PROMPTS=1`,
  and each tool's command or file path with `OTEL_LOG_TOOL_DETAILS=1`. Otari
  never keeps a tool's output, even when Claude Code exports it
  (`OTEL_LOG_TOOL_CONTENT=1`).
- `otari hook` sends the command or paths a gate checked, and the gate's own
  detail, with `OTARI_HOOK_REPORT_CONTENT=1`.

A span shows what it was about as its input, and a tool call is named for its
command or file in the tree. A span's content is sealed for that span alone, and
reads back empty once no configured key opens it, so keep a rotated-out key in
`OTARI_SECRET_KEY`, or in `trace_content_kms_previous_key_ids`, for as long as
you keep traces.

### Who reads content

A session's content is private to the user it belongs to. Nobody else reads it
in the normal course, not the workspace's other members, not its admins, not the
organization's owners and admins, and not the deployment's operator. They all
still see the session: that it ran, its tools, models, costs, outcomes and
guardrail verdicts. A span whose content they may not open says so
(`content_private: true`) and shows none.

| Reader | Reads a session's content |
| --- | --- |
| The session's own user | While they are a member of the workspace. |
| An organization owner or admin | When the user shared it with the organization's admins. |
| The deployment's operator | When the user shared it with the operator. |
| An organization owner, by break-glass | Any of the organization's sessions, recorded with a reason. |
| The deployment's operator, by break-glass | Any session, where `trace_platform_break_glass_enabled` is on, recorded with a reason and a reference. |
| Anyone else | Never. |

**Sharing.** A user shares one session at a time, from the session's page or with
`POST /api/v1/organizations/me/traces/{trace_id}/shares`, naming the audience
(`organization_admins` or `deployment_operator`) and an optional note saying why.
The share has no end date of its own: it ends when the user ends it
(`DELETE .../shares/{share_id}`), when the user loses the workspace (they leave it
or the organization, or lose the admin role that was their only way into it), or
when retention deletes the session. A share that ended stays ended if they return. Reads
through a share are recorded, at most once an hour per reader so a page left open
is one read, and the user sees who read it and when: `GET .../shares` lists every
share, running ones first and ended ones after, with its newest reads and how
many it has had, and `GET .../shares/{share_id}/reads` pages through all of them.
The record of each read outlives the share and the session: otari never deletes
it, and it names the workspace, the session and the audience it was read through.
Erasing the session's user clears whose session it was.

A share with the deployment's operator is read on the operator's trace routes,
which cover the operator's own organization. On a deployment where the operator
supports other organizations, as on otari.ai, reaching a share in another
organization needs the operator to act in that organization.

**A member who leaves.** Leaving the workspace or the organization ends their
shares, for good. Nothing new is kept for them, but what was already stored stays
until retention deletes it, and only break-glass opens it.

**Break-glass.** For an investigation a session's owner cannot be asked to share
for, such as an HR or legal matter, an owner of the organization (not an admin)
can open any of its sessions' content with
`POST /api/v1/organizations/me/traces/{trace_id}/break-glass`, giving a reason of
at least ten characters. Where `trace_platform_break_glass_enabled` is on, off by
default, the deployment's operator can open any organization's session with
`POST /api/v1/traces/{trace_id}/break-glass?workspace_id=...`, giving a reason and
a reference such as a legal request's number. Either way it takes a signed-in
person: a header master key names nobody to record, so it opens nothing this way.
Each opening:

- is recorded, with who, which session, when, the reason and the reference, in
  the same transaction that reads the session and before any content opens, so
  nothing opens without its record;
- is listed for the organization's owners at
  `GET /api/v1/organizations/me/traces/audit/break-glass` (their own openings),
  and the operator's openings for the operator at `GET /api/v1/traces/audit/break-glass`.
  An owner's reason can name the person under investigation, so the operator's
  list does not include the owners' openings. The organization's list does not
  include the operator's either, by design: an operator answering a legal request
  may not be free to tell the organization, so its openings are reviewed on the
  operator's side, and a deployment that wants them visible to the organization
  has to disclose them some other way;
- is logged as an event naming ids only, and, with AWS KMS, opens its data keys
  afresh rather than from the gateway's cache, so each opening is a `Decrypt` in
  CloudTrail naming the session. That outside record covers content sealed under
  the KMS key only: content sealed under `OTARI_SECRET_KEY`, before a deployment
  moved to KMS or by an earlier version, opens with no call to KMS, so the
  gateway's own record is its only one. Content ages out with its sessions, so
  after `trace_session_max_age_days` from the move all of it is under KMS;
- is not told to the session's owner.

Otari never deletes these records. They are kept apart
from the session and the organization: deleting an organization does not delete
the record of the operator's access to it. Erasing a user clears their id from
the records of openings of their sessions; the openings stay recorded, with the
email of whoever opened them and the reason given, which may mention anyone.

A trace's detail says how the read reached its content in `content_access`:
`owner`, `share`, `break_glass` or `none`.

This is an access rule the gateway enforces, not end-to-end encryption: whoever
runs the gateway holds the key that opens content, so a self-hosted deployment's
operator could read it by changing the code or the database. What the rule and
the encryption give is that nobody reads content through Otari without its
owner's say or a recorded break-glass, and that the database alone, or a backup
of it, shows none. With AWS KMS, every data key the gateway fetches is also in
CloudTrail.

At startup, where content is on, the gateway makes one data key and opens it
again, for a session no export names. A key it cannot use, such as a KMS key
whose policy or region is wrong, is logged as an error then, rather than only as
content quietly not kept (`gateway_trace_content_unsealed`). A deployment with
no key to seal with at all does not offer workspaces content, and nor does one
whose startup check failed, until the key source next makes a key.

A header master key reads no session's content, on any route: it names no person
a read or a share could be recorded against. Signed in, the operator reads what
their account may.

Deactivating an account ends every share it made, and reactivating it brings
none back. While it is deactivated no content is kept for its sessions, and
nobody reads through its shares.

An account whose address was unproven when someone proved the mailbox (a
password reset, or a first sign-in with an identity provider) keeps its sessions
and shares: the person who proved the mailbox is taken to be the account's
owner, and sees every share it made, ended or running, on each session.

### How content is sealed

Each span's content is encrypted with AES-256-GCM under a data key, and the
data key is stored beside it in wrapped form: wrapped with `OTARI_SECRET_KEY`,
or made by AWS KMS. The ciphertext is bound to its workspace, its session and
its span, so a value copied onto another span, another session or another
workspace reads back empty.

Each session has data keys of its own, so a key that leaked would open one
session. A data key serves its session for up to five minutes, so the key source
is asked once every few minutes per active session and replica rather than once
per span. A busy Claude Code session, around 14,000 spans an hour, costs KMS about
12 `GenerateDataKey` calls an hour, and reading that hour back about 12 `Decrypt`
calls, then none while they are cached. A batch fetches the keys it lacks
together, so a batch spanning many sessions waits for them at most 1.5 seconds
in all.

The key source's request quota is shared by every workspace, and one export can
name many sessions, so each workspace may ask a replica for 64 new data keys at
once and one a second after that. A session that starts past that allowance is
stored without its content (`gateway_trace_content_unsealed{reason="limited"}`),
and its next spans are sealed once the allowance refills. One second is room for
about 300 sessions active at once in a workspace, on each replica.

A cached data key is dropped five minutes after it was fetched, whether or not
the gateway is busy, and every cached key is dropped when the gateway stops.
Dropping a key removes the gateway's reference to it; Python does not overwrite
the memory it occupied, which stays as it was until it is reused.

When the key source cannot be reached, refuses the gateway, or takes more than
1.5 seconds to answer, spans are still stored, without their content, and the
gateway does not ask it again for 30 seconds; reads in those 30 seconds do not
extend them. Content that cannot be opened reads back empty. Both are counted, on
`gateway_trace_content_unsealed` and `gateway_trace_content_unopened`, by reason.

Content stored by earlier versions, sealed with `OTARI_SECRET_KEY` alone, still
reads back, and ages out with the session; nothing needs re-encrypting.

### Sealing content with AWS KMS

Set `trace_content_key_backend: aws_kms` and name a symmetric customer-managed
key. The gateway needs the `kms` extra (`boto3`) and AWS credentials from the
usual chain (an ECS task role, an instance profile, environment variables).

| Setting | Purpose |
| --- | --- |
| `trace_content_key_backend` | `secret_key` (the default), `aws_kms`, or `aws_kms_only`. |
| `trace_content_kms_key_id` | The key new content is sealed under: a key id, key ARN or alias. Required by `aws_kms` and `aws_kms_only`. Set beside `secret_key`, it still opens what it sealed, so a deployment can move back. |
| `trace_content_kms_previous_key_ids` | Earlier keys whose content still opens, separated by commas. |
| `trace_content_kms_region` | The key's region. Unset uses the AWS SDK's default. |
| `trace_content_kms_endpoint_url` | A KMS endpoint other than the region's, such as a VPC endpoint. |

On `aws_kms`, content sealed with `OTARI_SECRET_KEY` (before the deployment
moved to KMS) still opens, with no KMS call and so no CloudTrail record.
`aws_kms_only` leaves `OTARI_SECRET_KEY` off the keyring: such content reads back
empty, and nothing opens without the KMS key. Choose it where you need to say
that only your KMS key opens session content.

The gateway calls `kms:GenerateDataKey`, `kms:Decrypt` and `kms:DescribeKey`,
always with the encryption context `otari:purpose=trace-content`,
`otari:workspace=<workspace id>` and `otari:session=<trace id>`, so a policy can
allow only that purpose, and require every call to name a workspace and a
session:

```json
{
  "Effect": "Allow",
  "Action": ["kms:GenerateDataKey", "kms:Decrypt"],
  "Resource": "<key arn>",
  "Condition": {
    "StringEquals": {"kms:EncryptionContext:otari:purpose": "trace-content"},
    "ForAllValues:StringEquals": {"kms:EncryptionContextKeys": ["otari:purpose", "otari:workspace", "otari:session"]},
    "Null": {"kms:EncryptionContext:otari:workspace": "false", "kms:EncryptionContext:otari:session": "false"}
  }
}
```

plus `kms:DescribeKey` on the key and on each previous key listed, which the
gateway calls every few minutes to learn which keys it may open, so an alias
pointed at a new key is followed without a restart. Every `GenerateDataKey` and `Decrypt` lands in CloudTrail with
the workspace it was for. Those record each data key fetched, not each span
opened: the gateway opens a session's content with a cached key for five
minutes without asking KMS again, so CloudTrail shows when content was opened
for a session, and the share read records show who read which session.

Before relying on a key, check it from a development account with
`uv run python scripts/check_trace_content_kms.py --profile <profile> --region <region> --account <account id>`.
It creates throwaway keys, runs otari's own sealing code against them (KMS
refusing another session or workspace, rotation, an alias moved to a new key, the
batch deadline across many sessions, break-glass calling KMS afresh), and
schedules them for deletion. It makes real AWS calls, so it never runs in CI, and
it refuses to run when the profile resolves to any other account.

To rotate:

- **KMS automatic rotation** needs nothing: KMS keeps the earlier key material
  inside the same key.
- **Replacing the key**: set `trace_content_kms_key_id` to the new key and list
  the old one in `trace_content_kms_previous_key_ids` until traces older than the
  change have aged out. Content under a key that is not listed reads back empty.
- **Moving from `OTARI_SECRET_KEY` to KMS**: switch the backend. New content is
  sealed with KMS, and content sealed before opens with `OTARI_SECRET_KEY` for as
  long as it is set.

### Upgrading a deployment that keeps content

The upgrade turns every workspace's content off. During a rolling deploy,
replicas still on the earlier version neither read that switch nor take the
locks that erasure and turning content off rely on, so where
`trace_content_enabled` is on they go on storing content in every workspace
until they stop, and the upgraded replicas show it to its owner. Either:

- set `trace_content_enabled` to `false` for the deploy, and back to `true` once
  no earlier replica runs; or
- stop every earlier replica before the first upgraded one starts.

Content an earlier replica stored after the upgrade began can be removed with
`DELETE /api/v1/workspaces/{workspace_id}/trace-content`.

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
| `trace_platform_break_glass_enabled` | `false` | Let the deployment's operator open any organization's session content, recorded with a reason and a reference. |

Retention runs whether or not traces are captured, so a deployment that stops
capturing still deletes what it captured on this schedule. Erasing a user deletes
their traces on every replica: spans another replica took in for them before the
erasure, and still holds, are never stored. To enforce that, the gateway stores
nothing taken in more than 24 hours earlier, however long an upload or request
stalled, and keeps the erased user's id for about a day, then deletes it. The id
of a user never erased is deleted by the next hourly retention run.

Traces are written off the request path, within fixed limits: a trace never
slows or fails a request, and when the writer cannot keep up it drops whole
requests' spans rather than parts of them. One OTLP export adds at most
1,000 spans in at most 100 sessions. The
`gateway_trace_spans_dropped` metric counts what was dropped, by reason.

The cost on a trace is a snapshot of the usage rows' cost. Usage stays the
billing record.
