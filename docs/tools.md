# Built-in tools

Otari can run three tools during Chat Completions, Messages, and Responses:

- `otari_code_execution`, a sandboxed code session
- `otari_web_search`, a search backend
- `otari_web_fetch`, a bounded public-web fetcher

Code execution and Search need separate backends. Fetch runs in the gateway.
Search and Fetch may be combined, but web tools cannot be combined with code
execution or MCP servers.

Inspect the tools available on a running deployment:

```bash
curl http://localhost:8000/api/v1/tools \
  -H "Authorization: Bearer $OTARI_API_KEY"
```

Unavailable but recognized tools remain in the response with
`"available": false`.

## Who runs a tool

An `otari_*` type is executed by Otari. A provider-native web-search type is
forwarded to the provider unless [interception](#web-search-interception) is on.
A provider-native code-execution type is decided by the request's
[executor](#code-execution-executor). Function tools remain the caller's
responsibility.

### Web-search interception

Some clients can declare only provider-native search types. Set
`web_search_intercept: true` to execute `web_search`,
`web_search_<date>`, and `web_search_preview` through Otari's configured
backend. A function named `web_search` is never intercepted.

Interception is off by default because enabling it changes who performs searches
for providers that already support a native search tool. It requires
`web_search_url`.

### Bounding the searches one request runs

A declaration may carry `max_uses` to cap how many searches Otari runs for that
request. Because a gateway-run search is billed per successful call, the cap is
honored on every endpoint (`/api/v1/messages`, `/api/v1/chat/completions`,
`/api/v1/responses`) and on every declaration shape, including `otari_web_search`.
A failed search does not count against it, matching what is billed.

`max_uses: 0` is a cap of zero searches, not the absence of a cap, so every
search is refused. A negative or non-integer value is a caller mistake and is
rejected with a 400 rather than guessed at; `1.5` and `5.0` are both non-integer
here, so a float that happens to be whole is rejected too.

Past the cap, the model is told the search was refused and can answer without it.
An Anthropic-native declaration is answered in its own vocabulary, a
`web_search_tool_result` carrying `error_code: max_uses_exceeded`; every other
caller gets the same `[tool error]` string a failed tool produces. Even without
`max_uses`, requests remain bounded by `max_tool_iterations` and the shared
[10-call Search and Fetch limit](#web-fetch).

### Who may read the settings

`GET /api/v1/tool-settings` answers any signed-in identity, so a member is told how
the built-in tools behave on their requests. A caller who does not operate the
deployment is answered without `web_search_url`, `sandbox_url` and
`guardrails_url`: those name this deployment's own infrastructure, which is what
the network-safety gates on the Settings page are set against, and no tenant
acts on them. Changing any setting stays a deployment operator's to do.

## Pricing a gateway-run tool

In standalone mode, gateway-run tools are priced per successful call:

```text
otari:code_execution
otari:web_search
otari:web_fetch
```

The dashboard accepts dollars per call. The pricing API stores the value in
`input_price_per_million`, so one cent per call is `10000`.

With `require_pricing: true`, an unpriced gateway-run tool is refused before the
model call. A failed tool invocation is recorded but not charged. Charges settle
on the final usage row alongside model tokens.

Direct `POST /api/v1/search` uses a different price key,
`<provider>:<search-tool-name>`, because it calls a configured search provider
without a model tool loop.

## What clients receive

Otari consumes its own tool calls, sends results back to the model, and returns
the final answer.

Responses and Messages can expose native server-tool result blocks when their
wire format expects them. Chat Completions returns only the final assistant
message. Calls for tools the client must execute are preserved.

For gateway-run MCP activity in Messages streams, see
[Messages streaming activity](mcp.md#messages-streaming-activity).

## Code execution

Start the bundled sandbox:

```bash
docker compose --profile code-exec up
```

Request it with:

```json
{
  "model": "anthropic:claude-sonnet-4-6",
  "messages": [{"role": "user", "content": "Compute 23 factorial."}],
  "tools": [{"type": "otari_code_execution"}]
}
```

The sandbox speaks the [code-execution protocol](code-execution-protocol.md),
which is what `sandbox_url` points at. A deployment that would rather not run
one sets `sandbox_provider: e2b` instead and Otari runs the code on
[E2B](https://e2b.dev) itself, with no backend of its own and no `sandbox_url`;
everything below is the same either way. A runnable example lives under
`demo/code-exec/`.

### Reusing a sandbox across requests

Within one request, every code call the model makes shares one sandbox, so a
run can build on the last one's variables and files. Across requests, nothing is
held unless you ask: a request sends `container: "auto"` to say it wants a
sandbox that outlives it, and the response comes back naming the one it got.
Send that name on the next request and you get the same sandbox, with the
workspace as the last run left it. A request that asks for nothing is served
exactly as it was before any of this existed, and holds nothing, because a held
sandbox costs the deployment for as long as it is held.

This is the same contract as Anthropic's `container` and OpenAI's
`code_interpreter` container:

- On Messages, send `"container": "auto"` at the top level. The response carries
  Anthropic's `container` object, `{"id": "otari_cntr_…", "expires_at": "…"}`,
  on the final message and on the `message_start` event. Send the id back as the
  same top-level field.
- On Responses, send OpenAI's own `"container": {"type": "auto"}` on the
  `code_interpreter` tool entry. Every `code_interpreter_call` item carries the
  id as `container_id`; send it back as `"container": "otari_cntr_…"` on that
  entry.
- On every dialect, the response headers `X-Otari-Container-Id` and
  `X-Otari-Container-Expires-At` name it, and the `otari_code_execution` tool
  entry takes a `container` string, `"auto"` or an id.

`sandbox_container_idle_ttl_sec` (600 by default) is how long a held sandbox
lives, not whether one is held. Setting it to 0 refuses to hold one at all, so
an operator can take the capability away whatever a request asks for.

A resumed sandbox has the workspace the last run left, so files a run produced
are still there under their names, and a request's uploads are seeded on top.
The idle clock restarts on every use; a second, hard clock
(`sandbox_container_max_lifetime_sec`, 3600 by default) starts at the first
lease and never restarts, so one conversation cannot hold a sandbox open on the
provider indefinitely.

An id that is unknown, expired, another tenant's, or from a different sandbox
provider is refused with a 400 whose detail reads `Container '…' has expired or
does not exist.`, the phrasing clients of the provider APIs already treat as
"drop the id and start over". A container id is bound to the user and workspace
that leased it, and the refusal is the same for all four cases, so an id never
reveals whether someone else's sandbox exists. The same 400 answers any
container on a hybrid gateway, which has no local database to hold a lease in.

A sandbox runs one request at a time. Naming a container another of your own
requests is still using is a 409, not a 400: the id is good and a retry works,
where a 400 would have a client throw a live sandbox away. Two requests sharing
one workspace would interleave their code and each collect the other's files.

A response carries a container only when the sandbox really will still be there.
A `protocol` backend that declines the idle timeout Otari asks for is told to
destroy the session at the end of the request, as it was before reuse existed,
and the response names no container rather than one that is already gone.

`container` is only the gateway's where the gateway runs the code. A request
whose code execution the provider serves (see
[Code-execution executor](#code-execution-executor)) keeps the provider's own
`container` field, forwarded untouched, so an id they minted still resumes their
container. A value only Otari could have named, `auto` or an `otari_cntr_…` id,
is refused there rather than forwarded: it would reach the provider as a
malformed id of theirs, and the error would name a word Otari told you to send.
Sending an id Otari minted is the exception, because it names one sandbox and
the files in it, which already says where the code runs: under `auto` it pins
execution here rather than letting a model swap hand the request to the
provider and take the sandbox away. So a client can keep a container across
turns without also tracking which executor served each one. Only an id does
this; `auto` names no sandbox. Anything that outranks a request still outranks
it, so `Otari-Code-Execution: provider` and a workspace policy pinned to the
provider both win, and the id is refused as above.

`container: "auto"` is best-effort, unlike an id. A deployment that holds
nothing, because an operator set the TTL to 0 or because it is a hybrid gateway
with no local database to keep a lease in, serves the request exactly as it
would have and reports no container. An id is a requirement rather than a wish,
so the same deployment refuses it: the caller asked for specific files, and an
empty sandbox in their place is the wrong answer.

The provider's own timer does the reclaiming: the gateway tells it the idle
timeout when the request ends, and a periodic sweep only drops the rows that
named a sandbox nobody can resume any more.

### Code-execution executor

A request written for a provider's own sandbox keeps its provider's vocabulary:
Anthropic's `{"type": "code_execution_20250825"}` on `/api/v1/messages`, OpenAI's
`{"type": "code_interpreter"}` on `/api/v1/responses`, or the bare
`{"type": "code_execution"}`. The **executor** decides who runs the code such a
declaration asks for:

| Executor | Who runs the code |
| --- | --- |
| `auto` (default) | The provider, when it runs that tool natively for the dispatched model and wire format; otherwise Otari's sandbox. A request naming a container Otari minted runs here, since the sandbox it names is here (see [Reusing a sandbox across requests](#reusing-a-sandbox-across-requests)). |
| `otari` | Always Otari's sandbox. |
| `provider` | Always the provider; the declaration is forwarded untouched. |

`auto` is what makes a model swap transparent. An Anthropic Messages request
carrying `code_execution_20250825` runs natively against an Anthropic model, and
the same request against an open model runs on the sandbox, with the same
`server_tool_use` and `code_execution_tool_result` blocks coming back. An
`anthropic-beta` header travels only as far as it can be honored: against a
provider with no Messages API of its own it is dropped rather than refused,
because a beta names an Anthropic feature that provider was never going to
serve, and refusing it would make the request fail purely because its model
changed. On
Responses a claimed `code_interpreter` is answered with a `code_interpreter_call`
item. Chat Completions has no native shape, so a claimed declaration there
resolves inside the tool loop and only the final message is returned. Nothing
runs natively on Chat Completions, and the bare `code_execution` form is no
provider's, so under `auto` both always run on the sandbox.

Three layers choose the executor. The workspace pin wins over both of the
others; the header wins over the deployment default:

1. The deployment default, `code_execution_executor` (`OTARI_CODE_EXECUTION_EXECUTOR`),
   editable on the Tools page. Unset means `auto`.
2. A [workspace policy](#per-workspace-code-policy) may pin `executor`. A pin is
   a decision the request cannot argue with: a header that disagrees is refused
   with 403.
3. The `Otari-Code-Execution` header (`auto`, `otari` or `provider`) chooses per
   request where the workspace has not pinned. A value outside that vocabulary
   is a 400.

With no `sandbox_url` there is nothing to bring the code to, so a provider
declaration is always forwarded and no policy is read for it. Asking for `otari`
without a sandbox is a 400. The explicit `otari_code_execution` type is always
run by Otari, whatever the executor says. When a claimed declaration runs on the
sandbox, an `otari_code_execution` entry beside it is folded in rather than
refused; when the declaration stays with the provider, the two together are
still refused, because one request cannot address two sandboxes.

A gateway-run execution is described back in the caller's vocabulary with ids
Otari reserves (`otari_srvtoolu_…`, `otari_ci_…`, `otari_cntr_…`). When a client
echoes such a turn on its next request, a Messages pair is folded into a text
block, and a Responses item into an assistant message, so the model keeps the
code and its output; a provider's own items carry the provider's ids and pass through
untouched. Uploaded files the request references are seeded into the sandbox and
files the code produces come back as stored files, announced in Anthropic's
`code_execution_output` entries by their `file_id`, and on Responses as an
`image` output naming the URL Otari serves each produced image from (under
`public_base_url` where it is set, otherwise under the address the request
arrived on; any other produced file is listed and downloadable by id). A file the *provider's* own sandbox produced is copied
into Otari's store under the provider's own ID. See
[Files and code execution](files.md#files-and-code-execution). A sandbox
session still lives for one request, so a `container` id from a previous turn
addresses the provider's container, not the sandbox.

In hybrid mode the control plane's policy is consulted only once the decision
already points at the sandbox, so a declaration the provider serves natively is
never turned into a 403 for a workspace the control plane has not enabled.

**Upgrading from 0.6.** A deployment with `sandbox_url` set that forwarded
`code_interpreter`, `code_execution_<date>` or `code_execution` to a provider
with no native tool for the model now runs it on the sandbox and bills it as a
sandbox tool call. Set `code_execution_executor: provider` to keep forwarding.

### Per-workspace code policy

A workspace policy can disable code execution or narrow the deployment limits:

- `enabled`
- `max_iterations`
- `exec_timeout_s`
- `default_purpose_hint`
- allowed tool kinds
- an allowed sandbox image
- `executor`, the one field that is a choice rather than a narrowing (see above)

Manage it under
`/api/v1/workspaces/{workspace_id}/code-execution-policy` or from Tools. A policy
cannot enable a missing deployment backend or exceed the deployment limits.
Workspace-selected images must come from
`sandbox_allowed_session_images` or the deployment's own session image.

The authenticating API key determines the workspace. With no policy, deployment
defaults apply. In hybrid mode, the control plane resolves the policy instead.

## Web retrieval

### Web fetch

Fetch is disabled by default because it permits model-directed outbound requests.
Enable it at deployment time with `web_fetch_enabled: true` in `config.yml` or
`OTARI_WEB_FETCH_ENABLED=true` in the environment. When disabled,
`otari_web_fetch` remains discoverable with `"available": false`, and requests
declaring it are rejected without affecting other tools or ordinary completion
requests.

Declare Fetch on any completion API with `{"type": "otari_web_fetch"}`. The
model-facing function accepts exactly one field:

```json
{"url": "https://example.com/article"}
```

Otari retrieves one public HTTP or HTTPS URL, validates every redirect, and
extracts HTML, textual formats (including Markdown, JSON, XML, and JavaScript),
or text-bearing PDF content. Private, loopback, link-local, and otherwise unsafe
destinations are rejected. Provider-native Fetch declarations are passed
through unchanged rather than intercepted.

The tool result contains the display-safe source URL, the content type, an
optional requested URL when redirects changed the destination, extracted
content, and a warning that fetched content is untrusted. URL query strings and
fragments are omitted from the displayed metadata. The complete result is
limited to 50 KiB of valid UTF-8; the downloaded body is limited to 5 MiB and
PDF extraction is also bounded by page, time, memory, and intermediate-output
limits.

Search and Fetch share a limit of 10 attempted calls per request across model
turns and routing attempts. Invalid, blocked, and failed calls consume that
allowance. Attempting an eleventh call aborts the tool loop rather than returning
a `[tool error]` for the model to recover from. Non-streaming requests return
HTTP `422`; a response that is already streaming emits an error event and ends
without a completed answer.

Successful Fetch calls are billed under `otari:web_fetch`; ordinary Fetch
failures return sanitized tool errors, are counted as errors, and are not billed.

Declaring Fetch authorizes the model to make arbitrary public GET requests
within the workspace domain policy. Fetched content can contain prompt
injection. For example, a malicious page may instruct the model to encode
conversation or tool data into a later Fetch URL on an attacker-controlled
domain. Disable Fetch or restrict allowed domains when that residual egress risk
is unacceptable.

### Web search

Otari reaches a licensed search API directly. Set `web_search_provider` to
`tavily` or `brave` and `web_search_provider_api_key` to that provider's key;
the key stays in the gateway process and never reaches a caller.

For evaluation, the bundled SearXNG backend needs no key:

```bash
docker compose --profile web-search up
```

Request it with:

```json
{
  "model": "anthropic:claude-sonnet-4-6",
  "messages": [{"role": "user", "content": "Find the latest Python release."}],
  "tools": [{"type": "otari_web_search"}]
}
```

Public SearXNG engines may rate-limit automated traffic, so prefer a licensed
provider for production. `web_search_url` points at any other backend exposing
a SearXNG-compatible `/search?format=json` endpoint, and a configured provider
wins over it.

When content extraction is enabled, Otari retrieves each result through its
bounded public-web client. It validates and pins the resolved address before
connecting, validates every redirect, rejects HTTPS-to-HTTP downgrades, follows
at most five redirects, and applies one five-second network deadline across DNS,
connection setup, redirects, and body streaming. The decoded response body is
limited to 5 MiB.

Result-page retrieval ignores environment proxies by default. To use an
operator-controlled proxy that blocks unsafe destination addresses, set
`web_retrieval_trust_env_proxy: true` in the gateway configuration or
`OTARI_WEB_RETRIEVAL_TRUST_ENV_PROXY=true`. Retrieval then honors `HTTP_PROXY`,
`HTTPS_PROXY`, and the `ALL_PROXY` fallback (including lowercase forms), with
`NO_PROXY` exclusions. Only HTTP(S) proxy URLs are supported. Proxy settings
are read when the retrieval client is created.

This opt-in delegates connection-time address safety to the proxy, which must
block internal and other unsafe addresses after resolving each destination.
A general forwarding proxy does not provide this protection automatically.
Otari still applies local DNS/address checks, domain policy, redirect checks,
and response limits, so destination DNS must also work on the gateway.
Requests without an applicable proxy, including `NO_PROXY` matches, remain
IP-pinned. A failed proxy request never falls back to a direct connection.
This deployment-only setting cannot be enabled by a tool request or workspace.

Otari extracts HTML, textual formats (including Markdown, JSON, XML, and
JavaScript), and text-bearing PDFs. HTML and PDF parsing runs in a supervised
single-worker process with fixed time, memory, queue, page, and intermediate
output limits. Unsafe, unreachable, unsupported, empty, timed-out, or
unextractable results fall back to the search provider's snippet. A provider's
own `extracted_content` still takes precedence and is not fetched locally.

Each complete `web_search` tool result is limited to 50 KiB of valid UTF-8,
including any truncation notice. The existing 1,500-character per-result content
limit still applies before this overall result limit.

Where the search key must not sit on the machine serving traffic, a deployment
can also serve the search itself at `GET /api/v1/web-search/search`. This is the
hosted shape: the control plane holds the key and runs the query, and its
**hybrid** data plane calls it by setting `web_search_url` to
`{control-plane}/api/v1/web-search`. That address has to be under the gateway's
`PLATFORM_BASE_URL`, because a gateway forwards its platform token only to its
own control plane, and that token is what the route recognizes.

Set `web_search_backend_token` on the serving process to the token its own
gateway presents, which is that gateway's `OTARI_AI_TOKEN`. The two are one
secret, so rotating the platform token stops web search for that data plane
until both sides are updated. Without it the route is not mounted at all, since
it spends the deployment's search quota and a control plane is reachable from
the internet. A gateway someone else self-hosts presents a credential of its own and
is refused, deliberately: that is the same boundary that keeps a deployment's
provider keys off a foreign process. Such a gateway configures its own
`web_search_provider` or `web_search_url` instead, and the per-workspace policy
below still governs it.

Migrating from the Brave or Tavily adapter container: on the process that holds
the key, set `web_search_provider` and `web_search_provider_api_key` and unset
`web_search_url`. A hybrid data plane holds no key, so it keeps a
`web_search_url` and points it at its control plane's `/api/v1/web-search` instead.
The adapters, and the `web-search-brave` and `web-search-tavily` compose
profiles that ran them, were removed. A `web_search_url` still pointing at one
keeps the deployment looking configured while every search fails, so change both
together.

A runnable example lives under `demo/web-search/`.

### Per-workspace web-access policy

A workspace web-access policy can:

- disable `otari_web_search`, `otari_web_fetch`, and `POST /api/v1/search`
- lower `max_results`
- narrow allowed domains or add blocked domains for Search results and Fetch
  destinations, including redirects
- provide a default purpose hint
- supply provider options

Manage it under `/api/v1/workspaces/{workspace_id}/web-search` or from Tools.
`max_results`, the purpose hint, and provider options apply only to Search.
Workspace values can narrow deployment policy but cannot enable a missing
backend, enable deployment-disabled Fetch, or relax an operator limit. Fetch
remains available subject to deployment and workspace policy when no Search
backend is configured.

The policy also applies to direct search where relevant. In hybrid mode, the
connected control plane supplies workspace search configuration.

## Direct search

In standalone mode, `POST /api/v1/search` lets the caller submit a query directly
instead of waiting for a model to request one. Configure its named providers
through [`search_tools`](configuration.md#search-tools) or the Search tools API.

A SearXNG search tool can reuse `web_search_url`, so model-initiated and direct
search can share one backend. They remain distinct surfaces with separate
pricing keys.
