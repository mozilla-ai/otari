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

## Pricing a gateway-run tool

Gateway-run tools are priced per successful call:

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

Start the bundled SearXNG backend:

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

The bundled service is useful for evaluation, but public SearXNG engines may
rate-limit automated traffic. The repository includes Brave and Tavily adapters
under `scripts/`, or `web_search_url` can point at another compatible backend.

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

`POST /v1/search` lets the caller submit a query directly instead of waiting
for a model to request one. Configure its named providers through
[`search_tools`](configuration.md#search-tools) or the Search tools API.

A SearXNG search tool can reuse `web_search_url`, so model-initiated and direct
search can share one backend. They remain distinct surfaces with separate
pricing keys.
