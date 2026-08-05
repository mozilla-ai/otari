# Built-in tools

Otari can run two tools itself so any model, including open-weight ones, gets parity with what frontier APIs expose as managed tools:

- **`otari_code_execution`**: a sandboxed Python REPL
- **`otari_web_search`**: a web search backend

Both are opt-in per request via the `tools` array and require bringing up additional services via Docker Compose profiles. Operators who don't use them don't pull the extra images.

Built-in tools work on `/v1/chat/completions`, `/v1/messages`, and `/v1/responses`.

**Current limitations:** `otari_code_execution` and `otari_web_search` cannot be used together in the same request, and neither can be combined with `mcp_servers` in the same request. These are planned to be lifted; for now, pick one per request.

## How the keyword decides who runs it

An `otari_*` tool type is run by Otari in its own sandbox. Any other type, including provider-native keywords like `code_interpreter` or `web_search_<date>`, is forwarded to the provider's native sandbox untouched. Either way Otari handles routing and observability.

Billing differs by who ran the tool:

| Who ran it | Billed by Otari? |
|---|---|
| Otari (`otari_*`) | Yes, per call, at the rate you set (see [Pricing a gateway-run tool](#pricing-a-gateway-run-tool)) |
| Anthropic's native `web_search_<date>` | Yes: the provider reports its search count, which Otari meters the same way |
| Any other provider's native tool | No. The provider bills you directly; Otari records the tokens only |

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
| `/v1/responses` | A native `web_search_call` output item per search, before the message | The same item, as `response.output_item.added` / `.done` |
| `/v1/messages` | Nothing. The final message only | Nothing. The gateway's own `tool_use` blocks are not forwarded |
| `/v1/chat/completions` | Nothing. The final message only | Nothing. The gateway's own `tool_call` deltas are not forwarded |

The gateway's own tool calls are deliberately withheld from streaming clients: a
client shown a `tool_use` block can never be sent the matching `tool_result`,
because Otari consumed it, and an SDK accumulating that stream would be left with
an unanswered call.

Anthropic Messages does not get the native treatment even though it has a
`server_tool_use` block, because the matching `web_search_tool_result` block
requires `encrypted_content`, an Anthropic-signed blob only Anthropic can produce
and which clients echo back upstream on the next turn. Minting one would either
be forged or break the follow-up request. On Responses, a gateway-minted
`web_search_call` is stripped back off an inbound `input`, so echoing a previous
turn cannot ship it to a provider that never declared the tool.

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

The bundled SearXNG backend is suitable for trying things out but rate-limited for sustained use. For production, point `OTARI_WEB_SEARCH_URL` at a licensed backend. Ready-to-run Brave and Tavily adapters ship in `scripts/` and are available as separate Compose profiles (`web-search-brave`, `web-search-tavily`).

A runnable walkthrough is in `demo/web-search/`.

To search directly rather than as part of a completion, use
[`POST /v1/search`](api-reference.md#search). It is billed and usage-logged the
same way, but the caller supplies the query instead of the model, it is
configured separately under
[`search_tools`](configuration.md#search-tools), and it is priced under its own
key (see [Pricing a gateway-run tool](#pricing-a-gateway-run-tool)).
