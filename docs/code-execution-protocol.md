# Code-execution protocol

**Contract version: 1.**

When a request uses the `otari_code_execution` tool, Otari does not run the code
itself. It leases a session from a **code-execution backend**, runs each tool
call against that session, and releases it. This document specifies the contract
between the two, so that a backend other than the reference implementation can
be built against something canonical.

The reference backend is
[otari-sandbox-container](https://github.com/mozilla-ai/otari-sandbox-container),
a single container that stands alone with no orchestrator. mozilla.ai operates a
second, pool-backed backend for [otari.ai](https://otari.ai). Both implement this
contract, and Otari cannot tell which one answered.

The contract is HTTP/JSON, described by OpenAPI. Its machine-readable form is
[`public/code-execution-openapi.yaml`](public/code-execution-openapi.yaml),
which a backend implementer can generate a server stub or a client from
directly. The two are normative in different registers, and neither is
redundant: the OpenAPI document is normative for shapes, paths, and status
codes; this document is normative for the semantics a schema cannot carry, such
as session statefulness, how a failed program is reported, and the extension
policy. `tests/unit/test_code_execution_contract.py` fails when they disagree.

> **Why HTTP, and when that gets revisited.** A transport-neutral IDL (proto,
> serving gRPC and HTTP/JSON alike) was weighed and declined: it would be a
> second contract paradigm in a house whose SDKs are already generated from
> OpenAPI, with no workload today to pay for it. Server-streamed output
> (incremental `stdout` and `stderr` while code runs) rides HTTP over SSE or
> chunked responses, so streaming did not decide it. Two triggers reopen the
> question: the first backend that is not reachable over HTTP, and the first
> bidirectional interactive workload, meaning a live PTY where input is fed while
> output is read on one connection. Discrete tool calls, which is all the
> contract carries today, need neither.

## Roles

| Role | Who | Responsibility |
|---|---|---|
| Client | Otari | Leases a session, submits tool calls, releases the session |
| Backend | `otari-sandbox-container` or another implementation | Executes untrusted, model-generated code and returns results |
| Control plane | The platform, in hybrid mode | Authorizes the caller, enforces per-workspace policy, injects tenancy, meters usage |

The backend does not authorize callers, enforce quota, or meter usage. In
standalone mode there is no control plane at all: Otari addresses a backend it
was configured with. In hybrid mode the platform interposes a proxy that
authenticates the caller and enforces policy before forwarding; the contract
below is unchanged either way, which is what lets the same backend serve both.

## Operations

Six operations, of which the first three are the whole execution path. A
backend MUST implement those three; the file operations are OPTIONAL and are
used only by clients that move files in or out of a session.

| Operation | Purpose | Request | Response |
|---|---|---|---|
| `CreateSession` | Lease a session | Optional lifetime and image hints | A session handle |
| `Execute` | Run one tool call in a session | A tool call | A result block, plus execution metadata |
| `DestroySession` | Release a session | A session id | Empty |
| `ListFiles` | Enumerate a session's workspace | A session id, a path | File metadata |
| `GetFile` | Read one file from the workspace | A session id, a path | File bytes |
| `PutFile` | Write one file into the workspace | A session id, a path, file bytes | The stored path and size |

### CreateSession

Leases a session and returns a handle.

Request fields, all OPTIONAL:

| Field | Meaning |
|---|---|
| `image` | The image this session should run |
| `idle_timeout_seconds` | Reclaim the session after this long without activity |
| `max_lifetime_seconds` | Reclaim the session this long after creation |

Response fields:

| Field | Required | Meaning |
|---|---|---|
| `session_id` | yes | A string addressing this session in every later operation |
| `idle_timeout_seconds` | no | The idle timeout actually in force |
| `max_lifetime_seconds` | no | The maximum lifetime actually in force |
| `created_at` | no | When the session was created, in POSIX seconds |
| `last_activity_at` | no | When the session was last used, in POSIX seconds |

`session_id` is a string, not a number: a client may use it to address the
session without reformatting it.

A backend MAY accept the client's lifetime hints or clamp them to its own
ceilings, and reports what is in force. A client MUST NOT assume a session
outlives the values the handle reports.

`image` is a hint in the same sense and a weaker one: a backend that leases from
a fixed pre-baked pool MAY ignore it, and MUST NOT fail the request because it
was sent. Otari sends it when a workspace's code-execution policy pins an image,
or when the deployment names one in `sandbox_image`, and omits the field
entirely otherwise, so a backend built before this field existed sees the body it
always saw. Which images a workspace may pin is the operator's decision, not the
backend's: Otari refuses one that is not on the operator's allow-list before any
session is leased.

A backend MAY refuse when at capacity. This is a retryable condition in
principle, distinct from a malformed request, and a backend SHOULD signal it
distinctly so a client can tell the two apart. Note that Otari does not
currently retry it: it surfaces as a failed request rather than a backoff, so a
backend should not rely on the client re-offering the work.

### Execute

Runs one tool call against a leased session and returns its result. Requests
carry:

| Field | Required | Meaning |
|---|---|---|
| `tool` | yes | Which tool kind to run (see [Tool kinds](#tool-kinds)) |
| `input` | yes | The tool's input, shaped per its kind |
| `timeout_seconds` | no | How long the backend may spend executing |
| `tool_use_id` | no | Correlation id; the backend generates one if absent |

**Sessions are stateful.** Interpreter state (variables, imports) and the
workspace filesystem persist across calls within a session, and are destroyed
with it. This is the reason the contract is sessioned rather than one-shot, and
it is a guarantee clients rely on: a model may build up state over several calls
in a single request.

`timeout_seconds` bounds the backend's execution, not the client's patience. A
client MUST allow more wall-clock than it grants, since its own budget also
covers transport and the backend's teardown; otherwise a legitimate
near-limit execution is reported as an unreachable backend.

A backend MAY impose a ceiling on `timeout_seconds`, and MAY either clamp a
larger value or refuse the request as malformed, so the contract sets no maximum
of its own. A client that needs a long-running call cannot assume the value it
sent was honored.

### DestroySession

Releases the session and destroys its state. A backend SHOULD reclaim sessions
that are never released (an abandoned client, a crashed one), which is why the
lifetime bounds on the handle exist. Releasing a session that does not exist is
not an error worth distinguishing: it is already in the desired state.

### ListFiles, GetFile, and PutFile

Enumerate, read, and write files under the session's workspace, so a client can
seed inputs and retrieve artifacts the code produced. All three MUST confine
access to the addressed session's own workspace: a path escaping it MUST be
refused rather than served or written. A backend MAY cap the size of a written
file and MUST refuse one that exceeds the cap rather than truncating it.

## Tool kinds

`Execute` dispatches on one of three tool kinds, each with its own input shape.

| Tool kind | Input | Runs |
|---|---|---|
| `code_execution` | `code` | Source in a persistent interpreter |
| `bash_code_execution` | `command` | A shell command |
| `text_editor_code_execution` | `command`, `path`, and command-specific fields | A file view or edit |

`text_editor_code_execution` mirrors Anthropic's text-editor command set:
`view`, `create`, `str_replace`, `insert`, `undo_edit`. Its command-specific
fields (`file_text`, `old_str`, `new_str`, `insert_line`, `view_range`) are
validated per command, not per request.

A client is not required to expose every kind. Otari currently drives only
`code_execution`, and advertises exactly that one tool to the model; a backend
MUST still accept the kind it is asked for and MUST refuse an unknown one.

## Result blocks

An `Execute` response carries the outcome in a **result block**, under a
`result_block` field, alongside execution metadata:

| Field | Required | Meaning |
|---|---|---|
| `result_block` | yes | The outcome of the call (below) |
| `tool_use_id` | yes | The call's correlation id, echoed from the request or generated |
| `execution_time_ms` | no | How long the backend spent executing |

The result block itself matches Anthropic's `code_execution_20250825` content
blocks, so a consumer that already parses Anthropic responses needs no
translation layer for it. A full response:

```json
{
  "tool_use_id": "srvtoolu_...",
  "execution_time_ms": 84,
  "result_block": {
    "type": "code_execution_tool_result",
    "tool_use_id": "srvtoolu_...",
    "content": {
      "type": "code_execution_result",
      "stdout": "...",
      "stderr": "...",
      "return_code": 0,
      "content": [
        {"type": "code_execution_output", "file_id": "...", "filename": "chart.png"}
      ]
    }
  }
}
```

Note the envelope: the result block is nested under `result_block`, not returned
bare. A backend that returns the block at the top level is not conforming, and a
client validating against this contract will reject it.

Result-block fields:

| Field | Required | Meaning |
|---|---|---|
| `type` | yes | The tool kind that ran (below) |
| `tool_use_id` | yes | The call's correlation id |
| `content` | yes | The outcome payload |

`content` is REQUIRED even for a run that produced nothing: a run with no output
returns a `content` whose `stdout` and `stderr` are empty and whose
`return_code` is `0`, not a block with `content` omitted. Its own fields are all
OPTIONAL and default as shown in the example.

Each entry in the nested `content` list describes one file the execution
produced, and carries a `file_id` addressing it plus the `filename` the
execution gave it. Both are REQUIRED on an entry: a reference to a file a client
can neither name nor fetch is not worth emitting.

The block's `type` corresponds to the tool kind that ran:
`code_execution_tool_result`, `bash_code_execution_tool_result`, or
`text_editor_code_execution_tool_result`.

Two details are easy to get wrong, and both are load-bearing:

- **`content` is a single object, not a list.** The outer `content` is one
  `code_execution_result`; the *inner* `content` is the list, of file references
  the execution produced. Code that treats the outer field as a list of mixed
  content blocks will not parse a conforming response.
- **Failure is reported in the payload, not out of band.** There is no
  top-level error flag. A non-zero `return_code`, or output on `stderr`, is how
  a backend reports that the code failed. An `Execute` that ran code and
  collected its failure output is a *successful* operation; only a backend that
  could not run the call at all fails the operation itself.

  Concretely, this rules out the error variant of Anthropic's content-block
  union: a version 1 backend MUST NOT return a `content` of type
  `code_execution_tool_result_error` carrying an `error_code`. Report the
  failure through `return_code` and `stderr` instead. A client reading this
  contract has no reason to inspect `content.type`, so an error variant would be
  read as an empty successful run, and the call would be billed as one.

For a `text_editor_code_execution` `create`, the file's content is not echoed
back in the result: it is already on the originating tool-use input as
`file_text`.

### Version pinning, and why this contract has its own version

The result shapes are pinned to Anthropic's `code_execution_20250825` blocks.
Anthropic's own tool has since moved on to later versions with different result
shapes, which is precisely why this contract carries a version of its own: a
backend implements *contract version 1*, and stays conforming regardless of what
any upstream model provider's tool version does next. Re-pinning to a newer
upstream shape would be a new contract version, not a silent change to this one.

## Extension policy

This contract evolves **additively**, and both sides MUST tolerate that:

- A backend MAY return fields not described here.
- A consumer MUST ignore fields it does not recognize, rather than rejecting the
  response. Otari's client models drop unknown keys.
- A new tool kind, or a new result-block `type`, MAY be added. A client MUST NOT
  reject a result block solely because its `type` is unfamiliar; Otari treats
  `type` as an opaque string and reads the payload.
- Removing a field, renaming one, or changing the meaning of an existing one is
  a **breaking** change and requires a new contract version.

The practical consequence: a backend built against version 1 keeps working as
the contract grows, and a client written against version 1 keeps parsing newer
backends.

## Authentication and tenancy

The contract itself carries no authentication. The reference backend has none:
it assumes a single trusted client reached over a private network, and must not
be exposed to an untrusted one.

Authentication is therefore a property of the deployment, not of the contract. A
client MAY be configured to present a bearer credential on every operation, and
a backend (or a proxy in front of one) MAY require it. In Otari's hybrid mode
this is how the platform's authenticated proxy admits the request and derives
tenancy from the caller's workspace, so the backend behind it never has to.

Tenancy, when a backend is multi-tenant, is injected by whichever component
authenticates the caller. A backend that expects tenancy MUST fail closed when
it is absent, rather than defaulting to a placeholder tenant.

## HTTP/JSON binding

Payloads are JSON; the operation names above map to:

| Operation | Method and path |
|---|---|
| `CreateSession` | `POST /sessions` |
| `Execute` | `POST /sessions/{session_id}/exec` |
| `DestroySession` | `DELETE /sessions/{session_id}` |
| `ListFiles` | `GET /sessions/{session_id}/files/list` |
| `GetFile` | `GET /sessions/{session_id}/files?path=...` |
| `PutFile` | `POST /sessions/{session_id}/files` (multipart: `file`, optional `path`) |

Paths are relative to the backend's base URL, which Otari takes from
`sandbox_url` (`OTARI_SANDBOX_URL`). Field names on the wire are exactly the
names used above.

Status codes:

| Condition | Status |
|---|---|
| Session created, file written | `201` |
| Execute succeeded (including code that failed) | `200` |
| Session destroyed | `204` |
| Credential missing or rejected, where the deployment requires one | `401` |
| Unknown session, or unknown file | `404` |
| Malformed request, or unknown tool kind | `400` or `422` |
| Path outside the session workspace | `403` |
| File larger than the backend's cap | `413` |
| At capacity, session not leased | `503` |

A bearer credential, where the deployment uses one, is sent as
`Authorization: Bearer <token>`.

Server-streamed output (incremental `stdout` and `stderr` while code runs) fits
this binding over SSE or chunked responses and is a planned addition; it is not
part of version 1.

## Reference implementation, and checking conformance

[otari-sandbox-container](https://github.com/mozilla-ai/otari-sandbox-container)
is the reference backend: a single container, no orchestrator, published as
`mzdotai/otari-sandbox-container`, and the implementation the contract above was
read off. It is the one to read when a clause here is ambiguous, and the one to
start from when building another backend. Two behaviors of it are its own, not
the contract's: it refuses a `timeout_seconds` above 120 rather than clamping it,
and it never populates the file-reference list.

To check a backend against the published contract, point the conformance script
at a running instance:

```bash
uv run python scripts/check_code_execution_conformance.py --base-url http://localhost:8080
```

It leases a session, runs a call of each tool kind, exercises the file
operations, releases the session, and validates every response against
`docs/public/code-execution-openapi.yaml`. The three session operations are
required, so a failure there is a non-conforming backend; the file operations are
optional, and a backend that does not serve them reports as skipped rather than
failing. Running it is how a second implementation shows it is interchangeable
with the reference one rather than merely similar to it.

## Configuration

| Setting | Env var | Meaning |
|---|---|---|
| `sandbox_url` | `OTARI_SANDBOX_URL` | Base URL of the backend. Unset, `otari_code_execution` requests are rejected. |
| `sandbox_purpose_hint` | `OTARI_SANDBOX_PURPOSE_HINT` | Default purpose hint for the tool, when a request supplies none. |

See [Configuration](configuration.md) for the full settings reference and
[Built-in tools](tools.md) for the user-facing view of the tool.
