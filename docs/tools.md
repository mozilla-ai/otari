# Built-in tools

Otari can run two tools itself so any model, including open-weight ones, gets parity with what frontier APIs expose as managed tools:

- **`otari_code_execution`**: a sandboxed Python REPL
- **`otari_web_search`**: a web search backend

Both are opt-in per request via the `tools` array and require bringing up additional services via Docker Compose profiles. Operators who don't use them don't pull the extra images.

Built-in tools work on `/v1/chat/completions`, `/v1/messages`, and `/v1/responses`.

**Current limitations:** `otari_code_execution` and `otari_web_search` cannot be used together in the same request, and neither can be combined with `mcp_servers` in the same request. These are planned to be lifted; for now, pick one per request.

## How the keyword decides who runs it

An `otari_*` tool type is run by Otari in its own sandbox. Any other type, including provider-native keywords like `code_interpreter` or `web_search_<date>`, is forwarded to the provider's native sandbox untouched. Either way Otari handles routing, observability, and billing.

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
