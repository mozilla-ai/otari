# Built-in tools

Otari can run two tools itself so any model, including open-weight ones, gets parity with what frontier APIs expose as managed tools:

- **`otari_code_execution`**: a sandboxed Python REPL
- **`otari_web_search`**: a web search backend

Both are opt-in per request via the `tools` array and require bringing up additional services via Docker Compose profiles. Operators who don't use them don't pull the extra images.

Built-in tools work on `/v1/chat/completions`, `/v1/messages`, and `/v1/responses`.

**Current limitations:** `otari_code_execution` and `otari_web_search` cannot be used together in the same request, and neither can be combined with `mcp_servers` in the same request. These are planned to be lifted; for now, pick one per request.

To see what a given deployment exposes, ask it:

```bash
curl http://localhost:8000/v1/tools -H "Otari-Key: Bearer $OTARI_KEY"
```

Each entry reports the `tools[].type` values that deployment accepts, the argument schema the model is given, and a ready-to-send example. A tool with no backend configured is listed with `"available": false` rather than omitted, so a client can tell "not a thing" from "not set up here".

## How the keyword decides who runs it

An `otari_*` tool type is run by Otari in its own sandbox. Any other type, including provider-native keywords like `code_interpreter` or `web_search_<date>`, is forwarded to the provider's native sandbox untouched. Either way Otari handles routing and observability.

### Web-search interception

Some clients cannot be told to say `otari_web_search`. Claude Code, the Anthropic SDK, and Claude Desktop send Anthropic's own `{"type": "web_search_20250305", "name": "web_search"}`, so against a non-Anthropic model that declaration reaches a provider that cannot serve it. Setting `web_search_intercept` makes Otari claim the provider-named web-search keywords too, and run them against its own backend:

| Declared | Default | With `web_search_intercept` |
|---|---|---|
| `otari_web_search` | Otari | Otari |
| `web_search` | The provider | Otari |
| `web_search_<date>`, `web_search_preview` | The provider | Otari |
| A `function` named `web_search` | You dispatch it | You dispatch it |

Set it on the **Tools & Guardrails** page, or with `OTARI_WEB_SEARCH_INTERCEPT=true`. It needs `web_search_url` set; with no backend to intercept *to*, the keyword passes through as usual rather than failing the request.

It is off by default because turning it on takes a search away from a provider that would have run it: a deployment already relying on Anthropic's native web search would silently switch to Otari's backend on upgrade.

A `function` tool named `web_search` is never claimed, even with interception on. That is your own tool, and running it server-side would mean your handler never fires and you never get back a `tool_call` you can dispatch.

If the caller forces its declaration with `tool_choice` under a non-standard name, the choice is retargeted onto the backend's tool so the forced call still resolves.

`max_uses` on an Anthropic-native declaration is accepted but not enforced; `max_tool_iterations` bounds the loop instead.

Billing differs by who ran the tool:

| Who ran it | Billed by Otari? |
|---|---|
| Otari (`otari_*`) | Yes, per call, at the rate you set (see [Pricing a gateway-run tool](#pricing-a-gateway-run-tool)) |
| A provider's native tool (`code_interpreter`, `web_search_<date>`, …) | No. The provider bills you directly, per search or per session, and Otari records only the tokens the response reported |

## Pricing a gateway-run tool

A tool Otari runs itself costs you money at a search provider or a sandbox, so it
is priced per call under the key `otari:<tool>`, for example `otari:web_search`.

Set it on the **Tools & Guardrails** page in the dashboard, which asks for dollars
per call, or through the API. Over the API the stored convention is USD per
*million* calls (the same column model pricing uses), so a cent per search is
`10000`:

```bash
curl -X POST http://localhost:8000/v1/pricing \
  -H "Otari-Key: Bearer $OTARI_MASTER_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model_key": "otari:web_search", "input_price_per_million": 10000, "output_price_per_million": 0}'
```

The charge lands on the usage row of the request that triggered it, alongside the
token charge, and appears in the Activity detail as a `web_search_calls` line and
on the Usage page under "Gateway-run tools". A failed call is counted and never
billed.

When a request is routed through a [routing policy](routing.md), the whole
request's tool work is billed onto the row that served it, so a chain that failed
over does not split or double-count its searches.

**With `require_pricing` on (the default), an unpriced tool is refused with a 402**
before the provider is called, exactly as an unpriced model is. Otari warns at
startup when a configured tool has no price, so this surfaces before the first
rejected request. With `require_pricing` off, the calls run and are recorded at
zero cost.

Note the two search paths use different keys: the tool loop prices
`otari:web_search`, while [`POST /v1/search`](api-reference.md#search) prices
`<provider>:<tool>`, because that endpoint knows which commercial API it called.

## What your SDK sees

Otari consumes a gateway-run tool call itself: it runs the tool, feeds the result
back to the model, and returns the model's final answer. What the client observes
differs by API, because only one of the three has a vocabulary for "the server ran
a tool for you":

| API | Non-streaming | Streaming |
|---|---|---|
| `/v1/responses` | A native `web_search_call` output item per search, before the message | The same item, as `response.output_item.added` / `.done`. It is not repeated in `response.completed`'s `output`, so a client reading only the final response sees the answer without the calls |
| `/v1/messages` | A native `server_tool_use` + `web_search_tool_result` pair per search, before the message, for a caller that declared `web_search_<date>`. Nothing otherwise | The same pair, as `content_block_start` / `content_block_stop` events |
| `/v1/chat/completions` | Nothing. The final message only | Nothing. The gateway's own `tool_call` deltas are not forwarded |

The gateway's own tool calls are deliberately withheld from streaming clients: a
client shown a `tool_use` block can never be sent the matching `tool_result`,
because Otari consumed it, and an SDK accumulating that stream would be left with
an unanswered call. When a model asks for one of your tools *and* a gateway tool in
the same message, Otari runs its own, hides it, renumbers what is left so the
indices stay gapless for SDK accumulators, and hands you only the call you can
dispatch.

Billing is standalone-only. In hybrid mode the platform resolves the model and
receives the usage report, and that report carries no tool counts, so a
gateway-run tool call there is recorded upstream as tokens only.

On Messages, the native blocks are emitted only for a caller that asked in
Anthropic's own vocabulary (a `web_search_<date>` type). That is the declaration
which makes a client expect them and render a citations panel; `otari_web_search`
and the bare `web_search` short form do not, so those callers keep getting the
plain-text result they always have.

`web_search_tool_result` requires `encrypted_content`, an Anthropic-signed blob
only Anthropic can produce. Otari sends it **empty** rather than forging one, so
the block carries the URL, title, and page age a citations panel needs and nothing
it cannot legitimately provide.

That empty field doubles as provenance. Because clients echo the previous assistant
turn back to continue a conversation, Otari strips its own minted blocks off an
inbound `messages` array so an echoed turn never ships an unsignable block upstream.
Only blocks carrying gateway provenance are removed: a search a provider ran and
signed itself round-trips untouched, and the `server_tool_use` dropped is the one our
result answers, matched by `tool_use_id`, so a provider's pair is never split. A
client that echoes a minted block straight to Anthropic instead of through Otari
would be rejected there. Responses accepts a blunter version of the same trade-off
for its minted `web_search_call` items, which have no provenance marker available and
are stripped off an inbound `input` wholesale.

When a model asks for a gateway search *and* one of your own tools in the same
message, the search runs and its blocks are emitted alongside the call you have to
dispatch, so you get both the citations and your own `tool_use`.

## Code execution

Brings up a sandboxed Python REPL container Otari dispatches `otari_code_execution` calls to.

```bash
docker compose --profile code-exec up
```

Use in a request:

```json
{
  "model": "anthropic:claude-sonnet-4-6",
  "messages": [{"role": "user", "content": "Compute 23 factorial."}],
  "tools": [{"type": "otari_code_execution"}]
}
```

A runnable walkthrough is in `demo/code-exec/`.

## Web search

Brings up a SearXNG instance Otari dispatches `otari_web_search` calls to.

```bash
docker compose --profile web-search up
```

Use in a request:

```json
{
  "model": "anthropic:claude-sonnet-4-6",
  "messages": [{"role": "user", "content": "What's the latest stable Python release?"}],
  "tools": [{"type": "otari_web_search"}]
}
```

To let a client that only speaks a provider's vocabulary (Claude Code, the Anthropic SDK) reach this backend, turn on [web-search interception](#web-search-interception).

The bundled SearXNG backend is suitable for trying things out but rate-limited for sustained use. For production, point `OTARI_WEB_SEARCH_URL` at a licensed backend. Ready-to-run Brave and Tavily adapters ship in `scripts/` and are available as separate Compose profiles (`web-search-brave`, `web-search-tavily`).

A runnable walkthrough is in `demo/web-search/`.

To search directly rather than as part of a completion, use
[`POST /v1/search`](api-reference.md#search). It is billed and usage-logged the
same way, but the caller supplies the query instead of the model, it is
configured separately under
[`search_tools`](configuration.md#search-tools) (in the config file, or from the
dashboard's Tools page), and it is priced under its own
key (see [Pricing a gateway-run tool](#pricing-a-gateway-run-tool)). A tool with
`provider: searxng` runs against this same backend, so the endpoint needs no
commercial search key when `OTARI_WEB_SEARCH_URL` is already set.
