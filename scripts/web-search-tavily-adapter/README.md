# Tavily Search adapter

A thin SearXNG-compatible front-end over the [Tavily Search API](https://tavily.com/),
so the gateway's `otari_web_search` tool can use a licensed, LLM-oriented
search API instead of the bundled SearXNG metasearch (whose free engines
rate-limit/CAPTCHA automated queries by IP).

It exposes `GET /search?q=…&format=json` returning
`{"results": [{"url", "title", "content", "extracted_content"?}]}` (the exact
shape `WebSearchBackend` expects) and translates that to/from Tavily's API.
The Tavily key lives in this service, never in the gateway.

Tavily can return the full extracted page text (`raw_content`); the adapter
maps it onto each result's `extracted_content`, so the gateway skips its own
per-URL fetch + trafilatura step.

## Provider options

The gateway forwards provider-specific knobs (set via `provider_options` on the
`otari_web_search` tool entry, or as workspace defaults in hybrid mode) as
extra query params. This adapter whitelists only the Tavily params it knows
about; anything else is ignored:

- `max_results` (int)
- `search_depth` (`basic` | `advanced`)
- `topic` (`general` | `news` | `finance`)
- `time_range`
- `include_answer` (bool)

## Run it via docker-compose

1. Get a key from <https://tavily.com/>. Provide it via the environment:
   either `export TAVILY_API_KEY=...`, or (cleaner) drop it in a gitignored
   `.env` in the repo root, which `docker compose` auto-loads. You also need to
   point the gateway at the adapter, so set both:

   ```bash
   # ./.env  (gitignored)
   TAVILY_API_KEY=tvly-xxxxxxxx
   GATEWAY_WEB_SEARCH_URL=http://tavily-adapter:8080
   ```

   (If you pass an explicit `--env-file PATH`, compose reads that instead of
   the root `.env`, so put both vars in the file you actually pass.)

2. Start the adapter and the gateway:

   ```bash
   docker compose --profile web-search-tavily up -d --build tavily-adapter gateway
   ```

   (The `web-search-tavily` profile brings up the adapter; the `searxng`
   container is not needed.)

3. `otari_web_search` requests now resolve through Tavily. Verify directly:

   ```bash
   curl -s "http://localhost:8185/search?q=latest+python+version&format=json" | jq '.results | length'
   ```

## Swapping in a different provider

Any HTTP service exposing the same `/search?format=json` contract is a drop-in
replacement (Brave, Exa, Linkup, Serper, …). Copy this adapter and change the
upstream call in `app.py`. This adapter sets `extracted_content` (from Tavily's
`raw_content`), so the gateway uses that text directly and skips its own per-URL
fetch + trafilatura. Omit `extracted_content` instead if you want the gateway to
fetch and extract each page itself.
