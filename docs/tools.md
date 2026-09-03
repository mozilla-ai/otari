# Built-in tools

Otari can run two tools during Chat Completions, Messages, and Responses:

- `otari_code_execution`, a sandboxed code session
- `otari_web_search`, a search backend

Each tool is optional and needs a separate backend. Requests cannot currently
combine these tools with each other or with MCP servers.

Inspect the tools available on a running deployment:

```bash
curl http://localhost:8000/v1/tools \
  -H "Authorization: Bearer $OTARI_API_KEY"
```

Unavailable but recognized tools remain in the response with
`"available": false`.

## Who runs a tool

An `otari_*` type is executed by Otari. Other tool declarations are forwarded
to the provider, including provider-native code interpreter and web-search
types. Function tools remain the caller's responsibility.

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
honored on every endpoint (`/v1/messages`, `/v1/chat/completions`,
`/v1/responses`) and on every declaration shape, including `otari_web_search`.
A failed search does not count against it, matching what is billed.

`max_uses: 0` is a cap of zero searches, not the absence of a cap, so every
search is refused. A negative or non-numeric value is a caller mistake and is
rejected with a 400 rather than guessed at.

Past the cap, the model is told the search was refused and can answer without it.
An Anthropic-native declaration is answered in its own vocabulary, a
`web_search_tool_result` carrying `error_code: max_uses_exceeded`; every other
caller gets the same `[tool error]` string a failed tool produces. A request
without `max_uses` is bounded only by `max_tool_iterations`.

### Who may read the settings

`GET /v1/tool-settings` answers any signed-in identity, so a member is told how
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
```

The dashboard accepts dollars per call. The pricing API stores the value in
`input_price_per_million`, so one cent per call is `10000`.

With `require_pricing: true`, an unpriced gateway-run tool is refused before the
model call. A failed tool invocation is recorded but not charged. Charges settle
on the final usage row alongside model tokens.

Direct `POST /v1/search` uses a different price key,
`<provider>:<search-tool-name>`, because it calls a configured search provider
without a model tool loop.

## What clients receive

Otari consumes its own tool calls, sends results back to the model, and returns
the final answer.

Responses and Messages can expose native server-tool result blocks when their
wire format expects them. Chat Completions returns only the final assistant
message. Calls for tools the client must execute are preserved.

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

The sandbox speaks the [code-execution protocol](code-execution-protocol.md).
A runnable example lives under `demo/code-exec/`.

### Per-workspace code policy

A workspace policy can disable code execution or narrow the deployment limits:

- `enabled`
- `max_iterations`
- `exec_timeout_s`
- `default_purpose_hint`
- allowed tool kinds
- an allowed sandbox image

Manage it under
`/v1/workspaces/{workspace_id}/code-execution-policy` or from Tools. A policy
cannot enable a missing deployment backend or exceed the deployment limits.
Workspace-selected images must come from
`sandbox_allowed_session_images` or the deployment's own session image.

The authenticating API key determines the workspace. With no policy, deployment
defaults apply. In hybrid mode, the control plane resolves the policy instead.

## Web search

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

Where the search key must not sit on the machine serving traffic, a deployment
can also serve the search itself at `GET /v1/web-search/search`. This is the
hosted shape: the control plane holds the key and runs the query, and its
**hybrid** data plane calls it by setting `web_search_url` to
`{control-plane}/v1/web-search`. That address has to be under the gateway's
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
`web_search_url` and points it at its control plane's `/v1/web-search` instead.
The adapters, and the `web-search-brave` and `web-search-tavily` compose
profiles that ran them, were removed. A `web_search_url` still pointing at one
keeps the deployment looking configured while every search fails, so change both
together.

A runnable example lives under `demo/web-search/`.

### Per-workspace search policy

A workspace search policy can:

- disable search
- lower `max_results`
- narrow allowed domains or add blocked domains
- provide a default purpose hint
- supply provider options

Manage it under `/v1/workspaces/{workspace_id}/web-search` or from Tools.
Workspace values can narrow deployment policy but cannot enable a missing
backend or relax an operator limit.

The policy also applies to direct search where relevant. In hybrid mode, the
connected control plane supplies workspace search configuration.

## Direct search

In standalone mode, `POST /v1/search` lets the caller submit a query directly
instead of waiting for a model to request one. Configure its named providers
through [`search_tools`](configuration.md#search-tools) or the Search tools API.

A SearXNG search tool can reuse `web_search_url`, so model-initiated and direct
search can share one backend. They remain distinct surfaces with separate
pricing keys.
