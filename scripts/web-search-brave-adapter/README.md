# Brave Search adapter

A thin SearXNG-compatible front-end over the [Brave Search API](https://brave.com/search/api/),
so the gateway's `otari_web_search` tool can use a licensed API instead of the
bundled SearXNG metasearch (whose free engines rate-limit/CAPTCHA automated
queries by IP).

It exposes `GET /search?q=…&format=json` returning
`{"results": [{"url", "title", "content"}]}` — the exact shape
`WebSearchBackend` expects — and translates that to/from Brave's API. The Brave
key lives in this service, never in the gateway.

## Run it via docker-compose

1. Get a key from <https://brave.com/search/api/> and export it:

   ```bash
   export BRAVE_API_KEY=...
   ```

2. Start the adapter and point the gateway at it:

   ```bash
   export GATEWAY_WEB_SEARCH_URL=http://brave-adapter:8080
   docker compose --profile web-search-brave up -d --build brave-adapter gateway
   ```

   (The `web-search-brave` profile brings up the adapter; the `searxng`
   container is not needed.)

3. `otari_web_search` requests now resolve through Brave. Verify directly:

   ```bash
   curl -s "http://localhost:8182/search?q=latest+python+version&format=json" | jq '.results | length'
   ```

## Swapping in a different provider

Any HTTP service exposing the same `/search?format=json` contract is a drop-in
replacement (Tavily, Exa, Linkup, Serper, …). Copy this adapter and change the
upstream call in `app.py`. To skip the gateway's per-URL content extraction and
return snippet-only content, set `extracted_content` on each result.
