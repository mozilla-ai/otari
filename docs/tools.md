# Built-in tools

Otari can run three tools during Chat Completions, Messages, and Responses:

- `otari_code_execution`, a sandboxed code session
- `otari_web_search`, a search backend
- `otari_web_fetch`, a bounded public-web fetcher

Code execution needs a sandbox: a `sandbox_url` for the default `protocol`
provider, or a hosted `sandbox_provider` such as `e2b`, which needs none. Search
needs a search provider (`web_search_provider` or `web_search_url`). Fetch runs
in the gateway.
One request may declare any of them together, and MCP servers beside them: see
[Combining tools](#combining-tools).

Inspect the tools available on a running deployment:

```bash
curl http://localhost:8000/api/v1/tools \
  -H "Authorization: Bearer $OTARI_API_KEY"
```

Unavailable but recognized tools remain in the response with
`"available": false`.

## Who runs a tool

An `otari_*` type is executed by Otari. A provider-native web-search type is
forwarded to the provider unless [interception](#web-search-interception) or the
request's `Otari-Web-Search` header says otherwise. Anthropic's web-fetch type
is forwarded unless the request's [`Otari-Web-Fetch`](#anthropics-web-fetch)
header says otherwise.
A provider-native code-execution type is decided by the request's
[executor](#code-execution-executor). Each is decided for every attempt on its
own, so a fallback can run Otari's tool where the first model ran its own.
Function tools remain the caller's responsibility.

### Web-search interception

Some clients can declare only provider-native search types: `web_search`,
`web_search_<date>`, and `web_search_preview`. By default such a declaration is
forwarded to the provider. Set `web_search_intercept: true` to run every one of
them on Otari's configured backend instead. Interception is off by default
because enabling it changes who performs searches for providers that already
support a native search tool.

Without interception, a request can choose for itself with the
`Otari-Web-Search` header:

- `auto` is decided per attempt: the provider's own search on an attempt whose
  model runs the keyword natively, Otari's on an attempt whose model does not.
  Only Anthropic's dated keyword (`web_search_<date>`) on `/api/v1/messages`
  against an Anthropic model, and OpenAI's `web_search` / `web_search_preview`
  on `/api/v1/responses` against an OpenAI model, count as native. DeepSeek runs
  both on its compatible APIs: the dated keyword on `/api/v1/messages` with any
  of its models, and `web_search` on `/api/v1/responses` with `deepseek-v4-pro`
  only, since `deepseek-flash` ignores it there and answers without searching.
  So a request
  written for Claude's search keeps searching when its model is swapped, or
  falls back, to one with no search of its own, and on Messages a dated keyword
  is still answered with `server_tool_use` / `web_search_tool_result` blocks. A
  request its first model answers natively opens and pays for no search.
- `otari` always runs it on Otari's backend.
- `provider` always forwards it and never runs Otari's search, so on a
  fallback to a model that cannot run the keyword itself, that attempt fails
  like any other. Use `auto` to keep the provider's search and run Otari's only
  on a fallback that needs it.

An attempt that runs Otari's search follows the rules for a gateway-run search
below. One that cannot be served that way (the workspace has search off, or no
price under `require_pricing`) is skipped and recorded as `Skipped:` in the
activity log, and the request fails only when no model is left.

The header can add a claim but never remove one: with `web_search_intercept`
on, every search runs on Otari's backend whatever the header says, and
`provider` is refused with a 403, because interception is what puts every
search under the workspace's web-search policy and tool pricing.

Whenever Otari runs the search, the rules for a gateway-run search apply: the
workspace's web-search policy and tool pricing. All of this
requires something to search with, the [in-loop default](#web-search): a search
instance, or the legacy `web_search_provider` or `web_search_url`; without one,
the declaration is forwarded unchanged. A function named `web_search` is never
claimed.

### Anthropic's web fetch

Anthropic's fetch keyword (`web_fetch_<date>`) is forwarded to the provider by
default, so a request that falls back to a model with no fetch of its own sends
it a tool nobody runs. The `Otari-Web-Fetch` header takes it over the same way
`Otari-Web-Search` does for search, with the same three values:

- `auto` is decided per attempt: Anthropic's own fetch on an attempt to an
  Anthropic model on `/api/v1/messages`, the only place the keyword is native,
  and Otari's on any other. On Messages the result still comes back as
  `server_tool_use` / `web_fetch_tool_result` blocks. A fallback that cannot be
  served by Otari's fetch is skipped, as for search.
- `otari` always runs it on Otari.
- `provider` always forwards it and never runs Otari's fetch, so on a
  fallback to a model that cannot run the keyword itself, that attempt fails
  like any other. Use `auto` to keep Anthropic's fetch and run Otari's only on
  a fallback that needs it.

It is a header of its own, not part of `Otari-Web-Search`, because the two
tools are native in different places and a request may want one taken over and
not the other. There is no deployment-wide toggle like `web_search_intercept`
for fetch. A claimed declaration is an [Otari fetch](#web-fetch): it needs
`web_fetch_enabled` (without it the keyword is forwarded unchanged), follows the
workspace policy and pricing, and honors the declaration's `max_uses`,
`allowed_domains` and `blocked_domains`. The URL in the returned blocks omits
its query string and fragment, as the tool result does. When such a turn is
echoed back on a later request, Otari drops its own fetch blocks before calling
the model, as it does for search.

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
`web_search_tool_result` carrying `error_code: max_uses_exceeded`. Otherwise the
model gets the same `[tool error]` string a failed tool produces, and a Chat
Completions client sees the search in `otari_tool_calls` as `failed`, with the
tag `call_limit`. Even without
`max_uses`, requests remain bounded by `max_tool_iterations` and the
[call limit Search and Fetch share](#web-fetch).

### Who may read the settings

`GET /api/v1/tool-settings` answers any signed-in identity, so a member is told how
the built-in tools behave on their requests. A caller who does not operate the
deployment is answered without `web_search_url`, `sandbox_url` and
`guardrails_url`: those name this deployment's own infrastructure, which is what
the network-safety gates on the Settings page are set against, and no tenant
acts on them. Changing any setting stays a deployment operator's to do.

## Pricing a gateway-run tool

Gateway-run tools are priced per successful call:

```text
otari:code_execution
otari:web_search
otari:web_fetch
```

The dashboard's Web search and Code execution pricing pages, in the deployment
view, set these rates, and Web search each search or fetch tool's own, in
dollars per call. The pricing API stores the value in
`input_price_per_million`, so one cent per call is `10000`.

With `require_pricing: true`, a gateway-run tool whose calls no row would price
is refused before the model call, with a `402` and the code `pricing_required`.
The check looks for the rows settlement prices the calls with, described below:
a search or fetch on an instance passes on its own row, and a call the
deployment pays for does not pass on an organization's own override. A provider
that reports its cost still needs a row, and a zero rate counts as one. The
refusal names the tool's `otari:` row, which prices it on any instance. A failed
tool invocation is recorded but not charged. Charges settle on the final usage
row alongside model tokens.

The usage row's pricing breakdown has one line per tool and label, and the
dashboard's request detail shows each line with its label. The label says what
served the calls:

- `<provider>:<instance>` for a search or fetch instance, and
  `builtin:builtin_fetch` for the built-in fetcher;
- `org:<provider>` for a workspace's
  [organization search or fetch key](#organization-search-keys);
- `otari:<tool>` for everything else: code execution, MCP tools, and web search
  on the legacy settings.

A call is priced at the cost its provider reported, when it reports one, and
otherwise at the first of its label's rows that has a price: the instance's own
search or fetch row (its `price_per_call`, or a price row of kind `search` or
`fetch` keyed by the instance's name, `builtin_fetch` for the built-in fetcher),
then the `<provider>:<instance>` model-kind row it was priced under before tools
had a kind of their own (read until a later release, with a startup warning),
then `otari:web_search` or `otari:web_fetch`. A tool's own row and a model's
never share a key, so a search provider named like a model provider, such as
DeepSeek, cannot price its models. The `otari:<tool>` rows of gateway-run tools
stay model-kind rows. An organization key's calls take the `otari:` row only,
since the deployment cannot price the organization's own account. A search also
adds what its enrichment fetches reported. The line's `unit_rate` is its cost
over its calls.

Whose account runs a call decides whose prices count, as it does for a model
call. A call on the deployment's account takes only the deployment's price, or
an override the deployment operator set: code execution, a search or fetch on an
instance or the built-in fetcher, and a search on the legacy settings. A call on
an [organization search or fetch key](#organization-search-keys) is priced at the
deployment's price by default, and the organization's own override replaces it.

Where the deployment charges organizations for what its account runs (see
[Deployment-paid credentials](hosted-providers.md#deployment-paid-credentials)),
those tool calls are charged to the organization's funds whoever pays for the
model call: a request on the organization's own model key that searches on the
deployment's search tool is charged for the searches, and one on a hosted model
that searches on the organization's own key is charged for the model alone. A
request that may run a tool on the deployment's account is refused with `402`
(`insufficient_funds`) before the model is called when the organization has no
funds paid in, and so is a direct search on one of the deployment's tools.

Every fetch is labeled by the instance that served it, `builtin:builtin_fetch`
when `web_fetch_default_tool` names none, and so is every in-loop search on a
search instance. A search on the legacy settings, or on a Tavily or Brave
organization key, which the older backend still runs, is labeled
`otari:web_search`. Searches a model's provider runs itself carry the provider's
label, as the next section says.

A search's enrichment adds only what its fetcher reported, even when the fetch
instance has a rate of its own, which prices fetch tool calls only. Exa reports
what a fetch costs; Tabstack and the built-in fetcher report nothing. So hits
enriched through Tabstack cost the caller nothing more, while the deployment pays
Tabstack for each page, up to one per hit. Price that into the search instance's
rate, or keep the instance's enrichment on the built-in fetcher with
`fetch_tool: builtin_fetch`.

### Searches a provider runs itself

A model's own search tool, such as Anthropic's `web_search_<date>` forwarded to
an Anthropic model, is run and billed by the provider, per search. Otari reads
the count the provider reports (Anthropic's
`usage.server_tool_use.web_search_requests`) and adds those searches to the row
as `web_search` calls labeled `<provider>:web_search`, so the row and the
budget carry what the provider will charge. They are priced at the operator's
`anthropic:web_search` row when there is one, otherwise, with `default_pricing`
on, at the provider's list price: Anthropic's is $10 per 1,000 searches. An
organization's override prices them as it prices the call's tokens, so a
search on the organization's own provider key takes its override. With
neither, the searches stay on the row at no cost. Searches across the rounds
of a gateway tool loop are added up, as the rounds' tokens are. A search Otari runs in the
model's place, under [interception](#web-search-interception) or the
`Otari-Web-Search` header, is not the provider's. It is billed once, under the
label of what served it in Otari (see
[Pricing a gateway-run tool](#pricing-a-gateway-run-tool)), and never also as
`<provider>:web_search`.

Direct `POST /api/v1/search` prices a search on a tool at that tool's own search
price, then at its old `<provider>:<search-tool-name>` row, because it calls a
configured search provider without a model tool loop. The deployment pays for
that search, so only its own price, or an override the deployment operator set,
prices it. A search a workspace's [organization search key](#organization-search-keys)
served is priced under `otari:web_search` instead, since the deployment cannot
price the organization's own provider account, and the organization's own
override applies to it. A provider's reported cost, when it reports one, comes first.

With `require_pricing: true`, the default, a direct search with no price is
refused with a `402` and the code `pricing_required`, its detail naming the
price to add, as an unpriced model is. That holds for a provider that reports its
charge too, such as Exa, and a zero rate is a valid price. A key exempt from
budgets is not refused.

**Upgrading from 0.29.** Direct searches were exempt from `require_pricing`, so
where it is on, a search on a tool with no price of its own now answers `402`.
Give each search tool a price, and `otari:web_search` one if workspaces search on
an organization search key. In `config.yml`:

```yaml
search_tools:
  exa-search:
    provider: exa
    api_key: ${EXA_API_KEY}
    price_per_call: 0.005
```

or with the master key:

```bash
curl http://localhost:8000/api/v1/pricing \
  -H "Authorization: Bearer $OTARI_MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{"kind": "search", "model_key": "exa-search", "input_price_per_million": 5000, "output_price_per_million": 0}'
```

Set the rate at about what one search costs: it is what a search holds against
the caller's budget before it runs, and it is billed when the provider reports
no charge of its own. A provider's reported charge still comes first. A zero
rate is valid too and bills as before, but holds nothing. A row written as
`exa:exa-search` before tools had a kind still counts. The dashboard's Web
search page sets a tool's rate too, except one written from
`price_per_call`, which the configuration file restores on every start.
`require_pricing: false` serves unpriced searches again, and unpriced models with
them. And whether `require_pricing` is on or not, an organization's own pricing
override no longer prices a search on one of the deployment's tools.

**Upgrading from 0.31.** An organization's own override of `otari:code_execution`
or `otari:web_search` no longer prices code execution or a search on the legacy
settings, since both run on the deployment's account: set the price the
deployment charges with the master key. Where an overlay charges organizations,
those calls are now charged to the organization's funds even when the model call
was on the organization's own key, so such a request with no funds paid in is
refused with `402`.

## What clients receive

Otari consumes its own tool calls, sends results back to the model, and returns
the final answer.

Responses and Messages can expose native server-tool result blocks when their
wire format expects them. Chat Completions has no such blocks, so Otari reports
its web searches and fetches in a field of its own, `otari_tool_calls`, at the
top level of the response and of each streamed chunk: a search with its query
and the sources it found, a fetch with the URL it read, and a failure with what
may be shown about it. See [Web tool calls on Chat
Completions](api-reference.md#web-tool-calls-on-chat-completions). Code execution
and MCP calls are not reported there, so Chat Completions returns only the final
assistant message for them. Calls for tools the client must execute are preserved.
The dashboard Playground's own completion route is the one exception: its stream
also reports every gateway-run call and its result in an `otari_tool_activity`
field, on a chunk of its own before and after each call, so the page can show
them.

For gateway-run MCP activity in Messages streams, see
[Messages streaming activity](mcp.md#messages-streaming-activity).

## Combining tools

One request may declare any mix of `otari_web_search`, `otari_web_fetch`,
`otari_code_execution` and MCP servers (`mcp_servers` or `mcp_server_ids`), on
Chat Completions, Messages and Responses, streamed or not. The model is offered
all of them at once and may call any of them in the same tool loop, for example
searching, fetching a page and then running code on what it found. Each call
runs on the tool it names, and each tool keeps the rules it has alone:

- Its workspace policy and its pricing apply. A declared tool its policy refuses
  is never dropped while the others run: an attempt that needs it is refused as
  it would be alone, and [Who runs a tool](#who-runs-a-tool) says what that
  means on a fallback chain.
- A `max_uses` caps only its own tool: once search reaches its cap, the model is
  told so for each further search while code and MCP calls keep running.
- Each built-in comes back in the same native result blocks or items as it does
  alone ([What clients receive](#what-clients-receive)), and so does MCP activity.
- Usage counts each tool's calls under the tool's own name.

The tools share one loop, so its limit is shared too: `max_tool_iterations`
bounds the rounds of the whole request, and so does a workspace code policy's
`max_iterations` when the request runs code on Otari's sandbox. The last round
`max_tool_iterations` allows offers none of Otari's tools and tells the model to
answer from the results it already has, so a model that keeps searching ends with
an answer rather than a failed request ([MCP](mcp.md) says the same of its loop).

A tool declared in a provider's own words is still [decided per
attempt](#who-runs-a-tool), so on a fallback chain one attempt may run the
provider's own code execution beside Otari's search, and another run both on
Otari, in the same loop.

What one tool returns reaches the model before its next call, so a fetched page
or a search result can steer what the model asks of the sandbox or an MCP
server. Declare together only tools you would let untrusted web content drive.

A built-in tool's name belongs to the built-in in a request where Otari runs it,
on any of the request's attempts. When a connected MCP server exposes a tool of
the same name (`web_search`, `web_fetch` or `code_execution`), the request is
refused with a `400` naming the tool, before any model is called or a sandbox is
opened. Leave that tool out of the server's `allowed_tools`, or declare only one
of the two. In a request where Otari runs no built-in of that name, the server's
tool of that name runs as usual.

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

E2B needs an API key. Store it on the Tools page under Code execution, or with
`PUT /api/v1/tool-settings/sandbox/credential`, and it is used for the next
sandbox on every replica within 30 seconds, so rotating or replacing a revoked
key needs no redeploy. A stored key wins over `E2B_API_KEY`, which serves when
none is stored; clearing the stored key falls back to it. The key is held
encrypted with `OTARI_SECRET_KEY` and never shown again, only its last four
characters. With no key in either place the gateway still starts, logs an
error, and refuses each code-execution call until one is set.

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
- On every dialect, the response headers `Otari-Container-Id` and
  `Otari-Container-Expires-At` name it, and the `otari_code_execution` tool
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
reveals whether someone else's sandbox exists.

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

The provider's own id is never forwarded on the deployment's provider key, the
one a hosted provider serves a model on, because every organization the
deployment serves shares that account and a container id addresses state on it
rather than on a workspace. Such a request is refused with a 400, and a routing
policy passes over that candidate for one on a key of your own. Asking the
provider for a new container (OpenAI's `{"type": "auto"}`) still passes.

`container: "auto"` is best-effort, unlike an id. A deployment that holds
nothing, because an operator set the TTL to 0, serves the request exactly as it
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
| `auto` (default) | Decided per attempt: the provider, when the model that attempt goes to runs that tool natively in that wire format; otherwise Otari's sandbox. A request naming a container Otari minted runs here, since the sandbox it names is here (see [Reusing a sandbox across requests](#reusing-a-sandbox-across-requests)). |
| `otari` | Always Otari's sandbox. |
| `provider` | Always the provider; the declaration is forwarded untouched and Otari's sandbox is never used. On a fallback to a model that cannot run it, that attempt fails like any other. Use `auto` to keep the provider's tool and use the sandbox only on a fallback that needs it. |

`auto` is what makes a model swap transparent. An Anthropic Messages request
carrying `code_execution_20250825` runs natively against an Anthropic model, and
the same request against an open model runs on the sandbox, with the same
`server_tool_use` and `code_execution_tool_result` blocks coming back.

Fallbacks are decided the same way, one attempt at a time. A routing policy or
catalog ID that falls back from Claude to Claude on Vertex or Bedrock sends
Anthropic its own tool, and only if Anthropic fails before any output does the
fallback run the code on the sandbox; a request Anthropic answers never opens
or pays for one. A fallback that cannot run on the sandbox (code execution off
for the workspace, no sandbox price under `require_pricing`, or a container id
from another provider) is skipped rather than refusing the whole request, and
the request is refused only when no attempt can run. A
request that also declares `otari_code_execution` decides every attempt alike
instead, running the code on the sandbox unless every model runs it natively.

An `anthropic-beta` header travels only as far as it can be honored: against a
model its provider serves with no Messages API of its own (any non-Claude model
on Bedrock, say, or a Bedrock Claude request that sets `output_format`) it is
dropped rather than refused, and so is a `context_management` field, which
configures one of those betas, because a beta names an Anthropic feature that
model was never going to serve, and refusing it would make the request fail
purely because its model changed. On Responses a claimed `code_interpreter` is answered with a
`code_interpreter_call` item per run, in the order the calls ran among the
gateway's other native items (a `web_search_call`, say). Chat Completions has no
native shape, so a claimed declaration there resolves inside the tool loop and
only the final message is returned.

Gemini runs code natively on both Chat Completions and Messages, so under `auto`
a Gemini model keeps the bare `code_execution` form on either format, and
Anthropic's `code_execution_<date>` on Messages. Otari rewrites the declaration
into Gemini's own `{"code_execution": {}}` for each attempt that goes to Gemini
(a fallback to another provider still receives it as written). A request may
also send `{"code_execution": {}}` itself; Otari reads it as the bare form, so a
workspace pin or a fallback with no native code execution decides it the same
way. On Messages the run comes back as the same `server_tool_use` and
`code_execution_tool_result` blocks Anthropic returns. On Chat Completions,
which has no native shape, it comes back on the assistant message as
`extra_content.google.code_execution`: a list of `executable_code`,
`code_execution_result` and `code_execution_output` items in the order Gemini
produced them. For every other provider nothing runs natively on Chat
Completions, and the bare form is no other provider's, so under `auto` those run
on the sandbox.

The executor also decides where an attached file goes. Code running on Otari's
sandbox is given the file from Otari's own store, and code running in the
provider's container is given a short-lived copy at that provider, because such
a container reads only files that provider holds. Either way the file reaches
the code rather than only the model, and an attempt that falls back to the other
place is given the file there. See
[Files and code execution](files.md#files-and-code-execution).

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

With no sandbox (no `sandbox_url` for the `protocol` provider, and no hosted
`sandbox_provider` such as `e2b`) there is nothing to bring the code to, so a
provider
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
[Files and code execution](files.md#files-and-code-execution). A `container` id
the provider issued addresses the provider's container, not the sandbox; to keep
Otari's sandbox across turns, see
[Reusing a sandbox across requests](#reusing-a-sandbox-across-requests).

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
`/api/v1/workspaces/{workspace_id}/code-execution-policy`. The dashboard's Code
execution page sets `enabled`; every other field is set through the API. A policy
cannot enable a missing deployment backend or exceed the deployment limits.
Workspace-selected images must come from
`sandbox_allowed_session_images` or the deployment's own session image.

The authenticating API key determines the workspace. A workspace with no policy
gets what `code_execution_workspace_default` says: `true`, deployment defaults
apply; `false`, code execution is off until a workspace owner or admin turns it
on, and the request is refused as if a disabled policy were stored. It defaults
to `true`. The Playground's tools menu and the policy endpoint's `default_enabled`
field read the same setting.

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
or text-bearing PDF content. From an HTML page it keeps the content as light
Markdown and leaves out the page's navigation, banners, footers and sidebars: the
page's `<main>` or `<article>` elements where it marks them, its text without those
landmarks otherwise. Private, loopback, link-local, and otherwise unsafe
destinations are rejected. This is the built-in fetcher, `builtin_fetch`, which
serves the tool unless `web_fetch_default_tool` names a fetch instance of a
[fetch provider](#fetch-providers). Anthropic's own fetch keyword is passed through
unchanged unless the request [asks Otari to run it](#anthropics-web-fetch).

The tool result contains the display-safe source URL, the content type (left out
when a fetch provider does not report one), an optional requested URL when
redirects changed the destination, extracted content, and a warning that fetched
content is untrusted. URL query strings and
fragments are omitted from the displayed metadata. The complete result is
limited to 50 KiB of valid UTF-8; the downloaded body is limited to 5 MiB and
PDF extraction is also bounded by page, time, memory, and intermediate-output
limits.

Search and Fetch share a limit of `web_search_max_calls` attempted calls per
request, 10 unless the operator sets another, across model turns and routing
attempts. Invalid, blocked, and failed calls consume that allowance; enriching a
search's results does not. A call past it gets
`[tool error] web search call limit reached for this request` (or `web fetch`),
which the model can recover from, and a Chat Completions client sees the call in
`otari_tool_calls` as `failed` with the tag `call_limit`.

A request the older search backend serves keeps a fixed limit of 10 instead:
one whose search runs on the legacy settings or on a Tavily or Brave organization
key, and one that declares Fetch alone. There, an eleventh call aborts the tool
loop rather than returning a `[tool error]`. Non-streaming requests return HTTP
`422`; a response that is already streaming emits an error event and ends
without a completed answer.

Successful Fetch calls are billed as [Pricing a gateway-run
tool](#pricing-a-gateway-run-tool) says: under the fetch instance that served
them, then `otari:web_fetch`. Ordinary Fetch failures return sanitized tool
errors, are counted as errors, and are not billed.

Declaring Fetch authorizes the model to make arbitrary public GET requests
within the workspace domain policy. Fetched content can contain prompt
injection. For example, a malicious page may instruct the model to encode
conversation or tool data into a later Fetch URL on an attacker-controlled
domain. Disable Fetch or restrict allowed domains when that residual egress risk
is unacceptable.

### Fetch providers

A fetch instance, under `fetch_tools` (see [Search tools](configuration.md#search-tools)),
names a provider for `otari_web_fetch` and for enriching search results.
`web_fetch_default_tool` names the instance the fetch tool runs on, and the one
that enriches search results unless the search instance names its own
`fetch_tool`, so setting it moves enrichment too. `POST
/api/v1/search-tools/{name}/test` with a `url` tries an instance.

On a fetch instance, the provider fetches the page and follows its redirects on
its own side. Otari checks the requested URL before the call, and the URL the
provider reports finding the page at after it. Its host must pass the domain
policy, and may not be `localhost` or an IP address, in any of its forms, that is
not public. Other hostnames are the provider's to resolve, so Otari does not check
where they lead, as the built-in fetcher does. DeepSeek, Exa and Tabstack answer
with the URL as asked even after a redirect (`reports_final_url` is false in the provider
catalog), so a redirect they follow escapes the domain policy; where a workspace's
domain lists must hold on every hop, keep the fetch tool, and enrichment, on
`builtin_fetch`. Enriching a search's results, Otari applies the same checks
under the search's own domain rules. Each
provider's limits are in the
[any-fetch README](https://github.com/mozilla-ai/otari/tree/main/any-fetch#exa).
A provider's failure reaches the model as
`[tool error] <provider> fetch failed (<tag>)`, with the HTTP status when there is
one, and never with the provider's own message.

- **DeepSeek** (`deepseek`) needs an API key, read from the instance's `api_key`,
  such as `${DEEPSEEK_API_KEY}`, the key DeepSeek's models take too. Name the
  instance apart from a DeepSeek search tool, such as `deepseek-fetch`, since
  names are unique across `search_tools` and `fetch_tools`. It reports no cost, so
  a fetch is billed at the instance's own rate (its `price_per_call`, or a `fetch`
  price set on the dashboard's Web search page or through the API), else
  `otari:web_fetch`; a price for one of DeepSeek's models never prices it.
  DeepSeek's [pricing](https://api-docs.deepseek.com/guides/search), read
  2026-10-19, is $0.75 per 1,000 successful fetches, with at most 25 fetches a
  second per account, apart from searches, and a limit per domain; its free period
  covers only calls through DeepSeek's own Harness, Claude Code or Codex, not
  Otari's. A page it cannot turn into markdown, a missing one or a PDF included,
  fails the fetch.
- **Exa** (`exa`) needs an API key, read from the instance's `api_key`, and
  reports what each fetch costs (`costDollars.total`).
- **Tabstack** (`tabstack`) needs an API key, read from the instance's
  `api_key`, such as `${TABSTACK_API_KEY}`. It reports no cost, so a fetch is
  billed at the instance's own rate (its `price_per_call`, or a `fetch` price set
  through the API), else `otari:web_fetch`. Tabstack's
  [pricing](https://docs.tabstack.ai/pricing), read 2026-10-13, is 10 credits a
  page, and a credit costs $0.0001 to $0.0003 depending on the plan and on
  whether the plan's monthly credits are used up, so $0.001 to $0.003 a page.
  Tabstack documents up to 15 seconds a page at its default `effort` and up to
  60 at `effort: max`, so give the instance a `timeout` to match.

### Web search

The in-loop tool searches with the deployment's in-loop default:
`web_search_default_tool` when it names a search instance, else the legacy
settings below while they are set, else the only search instance when there is
exactly one (see [Search tools](configuration.md#search-tools)).
`web_search_default_tool: none` turns it off. A workspace's
[organization search key](#organization-search-keys) comes before all of them.

On a search instance, the search runs on the instance's
[provider](#search-providers), key, options and `timeout`, and asks for the
request's result count, which the workspace's `max_results` may lower and nothing
else caps: Exa returns up to 100. The declaration's `provider_options` are checked
against the provider, as on [direct search](#direct-search): a key it does not
take, or takes only from the deployment, is refused with a 400 before the model is
called, while a workspace's such key is left out with a warning. A provider's
failure reaches the model as
`[tool error] <provider> search failed (<tag>)`, with the HTTP status when there
is one, and never with the provider's own message; a Chat Completions client sees
the same fields in `otari_tool_calls`. Results are numbered across a request's
searches, so a second search's first result follows the first search's last, and
each shows its date as an ISO date, or date and time when the provider gives one.

A result that comes without page text is enriched through the fetch instance the
search instance names in `fetch_tool`, else `web_fetch_default_tool`, else the
built-in fetcher, for the first 20 such results of a search; the rest show their
snippets. The built-in fetcher's page replaces the result's snippet, as
on the legacy settings below. A [fetch instance](#fetch-providers) is asked for
at most the 1,500 characters a result shows, within the fetch instance's
`timeout`, and its page follows the snippet rather than replacing it, because a
provider such as Tabstack answers a site's error page as the page's text. A page
that cannot be fetched, or that the domain checks refuse, leaves the result its
snippet.

On the legacy settings, Otari reaches a licensed search API directly. Set
`web_search_provider` to `tavily` or `brave` and `web_search_provider_api_key` to
that provider's key; the key stays in the gateway process and never reaches a
caller.

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

Migrating from the Brave or Tavily adapter container: on the process that holds
the key, set `web_search_provider` and `web_search_provider_api_key` and unset
`web_search_url`. The adapters, and the `web-search-brave` and `web-search-tavily` compose
profiles that ran them, were removed. A `web_search_url` still pointing at one
keeps the deployment looking configured while every search fails, so change both
together.

A runnable example lives under `demo/web-search/`.

### Per-workspace web-access policy

A workspace web-access policy can:

- disable `otari_web_search`, `otari_web_fetch`, and `POST /api/v1/search`
- lower `max_results`, for `otari_web_search` and `POST /api/v1/search`; unset,
  only the provider's own maximum bounds it
- narrow allowed domains or add blocked domains for Search results and Fetch
  destinations, including redirects
- provide a default purpose hint
- supply provider options

Manage it under `/api/v1/workspaces/{workspace_id}/web-search`. The dashboard's
Tools > Web search page sets `enabled`, `max_results` and the domain lists; the purpose
hint and provider options are set through the API.
`max_results`, the purpose hint, and provider options apply only to Search.
Workspace values can narrow deployment policy but cannot enable a missing
backend, enable deployment-disabled Fetch, or relax an operator limit. Fetch
remains available subject to deployment and workspace policy when no Search
backend is configured.

On direct search, the policy's switch and its `max_results` apply; its domain
lists, purpose hint and provider options do not.

A workspace with no policy gets what `web_search_workspace_default` says: `true`,
nothing is narrowed; `false`, web access (all three doors above) is
off until a workspace owner or admin turns it on, and a request is refused as if
a disabled policy were stored, whatever search key the organization holds. The
Playground's tools menu and the policy endpoint's `default_enabled` field read
the same setting. It defaults to `true`.

### Organization search keys

An organization can bring its own provider keys, so its workspaces' searches and
fetches are paid by its own accounts and count against its own quotas, rather
than the deployment's. Each key has a kind, fixed when it is added:

- a `search` key serves model-initiated search and a
  [direct search](#direct-search) that names no tool. Its provider is one
  `GET /api/v1/search-tools/providers` lists other than `searxng`, or `tavily` or
  `brave`;
- a `fetch` key serves the fetch tool and the enrichment of search hits. Its
  provider is one `GET /api/v1/search-tools/providers?kind=fetch` lists.

Organization owners and admins manage the keys under
`/api/v1/organizations/me/web-search-keys`: add, rename or replace a key,
archive it, and make one key of each kind the organization's default. Only the
last four characters of a key are ever shown. A key carries an API key when its
provider needs one, and may carry its own `api_base` and provider `options`,
which are checked against the provider when the key is written. An `api_base`
must use https when the key carries an API key, and must reach public addresses
only, whatever `OTARI_PROVIDER_ALLOW_PRIVATE_HOSTS` says, since an organization
rather than the operator chose it. Calls to it are held to the same rule: each
one connects to the address that was checked, follows no redirect, and bypasses
any environment proxy, as calls to a workspace's own model endpoint do. Moving
a key that holds a secret to another `api_base` needs the secret in the same
request (`api_key`, and any masked `options` entry), since the next call sends
it there. An organization's model provider keys follow the same rule, with
`client_args` in place of `options`. A
`tavily` or `brave` key takes neither, since the older backend that runs them
calls their public endpoints.
A key's calls never fall back to the deployment's own key or endpoint for its
provider.

Each workspace uses one key of each kind:

1. the key of that kind the workspace pinned as its own, if any;
2. otherwise the organization's default of that kind;
3. otherwise the organization's oldest key of that kind.

A key the workspace turned off, an archived key, or one this deployment cannot
decrypt is passed over. A workspace with no usable key of a kind uses the
deployment's instance exactly as before. A workspace's owners and admins pin or
turn off keys under `/api/v1/workspaces/{workspace_id}/web-search-keys`, which
any member of the workspace can read to see which keys it uses and, as
`search_provider` and `fetch_provider`, the providers its searches and fetches
run on.

A workspace with a usable search key serves an explicit `otari_web_search`
declaration even where the deployment has no search of its own, and the
workspace's web-access policy still applies. Three things stay with the
deployment's own configuration: `GET /api/v1/tools` and the Playground report
web search only when the deployment can search, and a provider-native search
declaration or the `Otari-Web-Search` header is handed to Otari only then, since
that is decided before the workspace's policy is read. Elsewhere such a
declaration goes to the model's provider.

A key's calls are labeled `org:<provider>` and priced under `otari:web_search`
or `otari:web_fetch`, at the cost the provider reported when it reports one (see
[Pricing a gateway-run tool](#pricing-a-gateway-run-tool)). A `tavily` or `brave`
key's in-loop searches are labeled `otari:web_search` while the older backend runs
them.

**Upgrading from 0.31.** An organization had one default per provider and now has
one per kind. Where it had both a Tavily and a Brave default, the Tavily one stays
the default. A workspace that had turned the Tavily default off, and had neither
pinned a key nor turned the Brave default off, gets the Brave key pinned, so it
keeps searching with Brave. One case changes: where this deployment cannot
decrypt the Tavily default, the workspaces that searched with the Brave default
in its place now search with the organization's oldest usable key. Check the
keys' `usable` field before upgrading.

## Direct search

`POST /api/v1/search` lets the caller submit a query directly
instead of waiting for a model to request one. Configure its named providers
through [`search_tools`](configuration.md#search-tools) or the Search tools API.

A SearXNG search tool can reuse `web_search_url`, so model-initiated and direct
search can share one backend. They remain distinct surfaces with separate
pricing keys.

A direct search that names no tool runs on the workspace's usable
[organization search key](#organization-search-keys), if it has one, in place of
the deployment's default tool. That tool must still be configured, and the request
is still checked against the key's `allowed_tools`, rate-limited and timed out
as that tool and logged under its name, while the search is priced under
`otari:web_search`. A search that names a tool, in the path or in
`search_tool_name`, runs on that tool with its own key. A
Brave key localizes results to the request's `country`; Tavily takes a country by
its full name, so a Tavily key does not.

The request, the response and the errors are in the
[API reference](api-reference.md#search).

### Search providers

A search instance serves direct search and, as the [in-loop default](#web-search),
the in-loop tool, labeled `<provider>:<search-tool-name>` on both and priced as
[Pricing a gateway-run tool](#pricing-a-gateway-run-tool) says.

- **DeepSeek** (`deepseek`) needs an API key, read from the tool's `api_key`, such
  as `${DEEPSEEK_API_KEY}`, the key DeepSeek's models take too. It reports no
  cost, so a search is billed at the tool's own rate (its `price_per_call`, or a
  `search` price set on the dashboard's Web search page or through the
  API), else `otari:web_search`; a price for one of DeepSeek's models never prices
  it. A key restricted to some models runs it when
  [`allowed_tools`](access-control.md#users-and-identities), its own or its
  user's, names it; `deepseek:*` in `allowed_models` grants DeepSeek's models
  alone. Name it apart from a DeepSeek fetch instance, such as `deepseek` and
  `deepseek-fetch`. DeepSeek's
  [pricing](https://api-docs.deepseek.com/guides/search), read 2026-10-19, is
  $0.75 per 1,000 successful searches, whatever the number of results, with at
  most 100 searches a second per account, apart from fetches; its free period
  covers only calls through DeepSeek's own Harness, Claude Code or Codex, not
  Otari's. DeepSeek does not charge a search that returns no results, but Otari
  bills it at the tool's rate like any other successful search. A query may be up
  to 128 characters, 111 with a `time_range`. A search returns at most 50 results,
  and its hits carry a snippet; page text only with the `need_content` option.
- **Exa** (`exa`) needs an API key, read from the tool's `api_key`, such as
  `${EXA_API_KEY}`, and reports what each search costs (`costDollars.total`), which
  is what the search is billed at. Exa's [pricing](https://exa.ai/docs/admin/pricing),
  read 2026-10-13, is per search, covering up to 10 results, and every result past
  10 adds to it, so a workspace's `max_results` ceiling also bounds what one search
  can cost. A search with no results reported a cost of 0 when the adapter was
  written.
- **Mojeek** (`mojeek`) needs an API key, read from the tool's `api_key`, such as
  `${MOJEEK_API_KEY}`. Mojeek takes the key and the query in the request URL,
  which Otari keeps out of its logs. It reports no cost, so a search is billed at
  the tool's own rate (its `price_per_call`, or a `search` price set through the
  API). Mojeek's
  [pricing](https://www.mojeek.com/services/search/web-search-api/), read
  2026-10-13, is per search, in pounds before taxes: £2 per 1,000 on the Startup
  plan, £3 on Business, Enterprise by arrangement; set the rate in dollars at what
  your plan costs you. A search returns at most 10 results, the Startup plan's
  limit, whatever the plan, and its hits carry a snippet but no page text.
- **Serply** (`serply`) needs an API key, read from the tool's `api_key`, such as
  `${SERPLY_API_KEY}`. Serply returns Google's results; it takes the key in a
  header and the query in the request URL, which Otari keeps out of its logs. It
  reports no cost, so a search is billed at the tool's rate,
  `serply:<search-tool-name>`. Serply's [pricing](https://serply.io/pricing), read
  2026-10-15, is prepaid credits, one per search not served from its cache, from
  $2.00 per 1,000 (2,500 for $5) down to $0.75 per 1,000 in the largest packs; a
  new account gets 2,500 credits for its first 30 days. A search returns at most
  10 results, and its hits carry a snippet but no page text.
- **SearXNG** (`searxng`) is self-hosted and reports no cost, so a search is billed
  at the tool's rate.
