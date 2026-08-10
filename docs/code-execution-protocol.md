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

> **Transport is not yet fixed.** This specification describes operations and
> payload semantics, deliberately independent of how they are carried. Today
> there is exactly one binding, HTTP/JSON, described under
> [HTTP/JSON binding](#httpjson-binding) and implemented by every backend that
> exists. That binding is **provisional**: while only one transport has ever been
> implemented, committing the versioned contract to it would be premature. The
> operations and payloads below are the stable part; the binding may gain
> siblings.

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

Five operations, of which the first three are the whole execution path. A
backend MUST implement those three; the file operations are OPTIONAL and are
used only by clients that move files in or out of a session.

| Operation | Purpose | Request | Response |
|---|---|---|---|
| `CreateSession` | Lease a session | Optional lifetime hints | A session handle |
| `Execute` | Run one tool call in a session | A tool call | A result block |
| `DestroySession` | Release a session | A session id | Empty |
| `ListFiles` | Enumerate a session's workspace | A session id, a path | File metadata |
| `GetFile` | Read one file from the workspace | A session id, a path | File bytes |

### CreateSession

Leases a session and returns a handle. The handle's `session_id` addresses the
session in every later operation. A backend MAY apply an idle timeout and a
maximum lifetime, and MAY accept the client's hints for them or clamp them to
its own ceilings; a client MUST NOT assume a session outlives the values the
handle reports.

A backend MAY refuse when at capacity. This is a retryable condition, distinct
from a malformed request.

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

### DestroySession

Releases the session and destroys its state. A backend SHOULD reclaim sessions
that are never released (an abandoned client, a crashed one), which is why the
lifetime bounds on the handle exist. Releasing a session that does not exist is
not an error worth distinguishing: it is already in the desired state.

### ListFiles and GetFile

Enumerate and read files under the session's workspace, so a client can retrieve
artifacts the code produced. Both MUST confine access to the addressed session's
own workspace: a path escaping it MUST be refused rather than served.

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

Every `Execute` returns a **result block** whose shape matches Anthropic's
`code_execution_20250825` content blocks, so a consumer that already parses
Anthropic responses needs no translation layer:

```json
{
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
```

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
- A consumer MUST ignore fields it does not recognise, rather than rejecting the
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

Tenancy, when a backend is multi-tenant, is injected by whatever authenticated
the caller. A backend that expects tenancy MUST fail closed when it is absent
rather than defaulting to some placeholder tenant.

## HTTP/JSON binding

The one binding that exists today. Provisional, per the note at the top of this
document. Payloads are JSON; the operation names above map to:

| Operation | Method and path |
|---|---|
| `CreateSession` | `POST /sessions` |
| `Execute` | `POST /sessions/{session_id}/exec` |
| `DestroySession` | `DELETE /sessions/{session_id}` |
| `ListFiles` | `GET /sessions/{session_id}/files/list` |
| `GetFile` | `GET /sessions/{session_id}/files?path=...` |

Paths are relative to the backend's base URL, which Otari takes from
`sandbox_url` (`OTARI_SANDBOX_URL`). Field names on the wire are exactly the
names used above.

Status codes:

| Condition | Status |
|---|---|
| Session created | `201` |
| Execute succeeded (including code that failed) | `200` |
| Session destroyed | `204` |
| Unknown session | `404` |
| Malformed request, or unknown tool kind | `400` or `422` |
| Path outside the session workspace | `403` |
| At capacity, session not leased | `503` |

A bearer credential, where the deployment uses one, is sent as
`Authorization: Bearer <token>`.

Server-streamed output (incremental `stdout` and `stderr` while code runs) fits
this binding over SSE or chunked responses and is a planned addition; it is not
part of version 1. Bidirectional interactive execution, where input is fed while
output is read on one connection, is the case this binding strains at, and is
the open question behind not yet fixing the transport.

## Configuration

| Setting | Env var | Meaning |
|---|---|---|
| `sandbox_url` | `OTARI_SANDBOX_URL` | Base URL of the backend. Unset, `otari_code_execution` requests are rejected. |
| `sandbox_purpose_hint` | `OTARI_SANDBOX_PURPOSE_HINT` | Default purpose hint for the tool, when a request supplies none. |

See [Configuration](configuration.md) for the full settings reference and
[Built-in tools](tools.md) for the user-facing view of the tool.
