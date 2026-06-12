"""SearXNG-compatible adapter in front of the Tavily Search API.

The gateway's ``WebSearchBackend`` speaks one protocol: it issues
``GET {url}/search?q=…&format=json`` and expects

    {"results": [{"url": ..., "title": ..., "content": ...}, ...]}

Tavily's API has a different shape and its own auth header, so this thin
service translates between the two. Point the gateway at it with
``GATEWAY_WEB_SEARCH_URL=http://tavily-adapter:8080``; the Tavily key lives
here, never in the gateway.

The gateway may forward provider-specific knobs as extra query params
(``provider_options`` on the ``otari_web_search`` tool entry). This adapter
WHITELISTS only the Tavily params it knows about (``max_results``,
``search_depth``, ``topic``, ``time_range``, ``include_answer``) and ignores
anything else; arbitrary params are never forwarded to Tavily.

Tavily's ``raw_content`` (full extracted page text) is mapped onto each
result's ``extracted_content`` when present, so the gateway can skip its own
per-URL fetch + trafilatura step. ``content`` carries Tavily's snippet.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse

TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")
TAVILY_ENDPOINT = "https://api.tavily.com/search"

app = FastAPI(title="otari web-search Tavily adapter")


@app.get("/health")
async def health() -> JSONResponse:
    # Fail closed when misconfigured (missing key) so orchestrators don't treat
    # an unusable adapter as healthy.
    if not TAVILY_API_KEY:
        return JSONResponse(status_code=503, content={"status": "missing TAVILY_API_KEY"})
    return JSONResponse(status_code=200, content={"status": "healthy"})


@app.get("/search")
async def search(
    q: str = Query(..., min_length=1),
    max_results: int | None = Query(default=None, ge=1),
    search_depth: str | None = Query(default=None, pattern="^(basic|advanced)$"),
    topic: str | None = Query(default=None, pattern="^(general|news|finance)$"),
    time_range: str | None = Query(default=None, pattern="^(day|week|month|year|d|w|m|y)$"),
    include_answer: bool | None = Query(default=None),
) -> JSONResponse:
    """Translate a SearXNG-style query into a Tavily Search API call."""
    if not TAVILY_API_KEY:
        return JSONResponse(status_code=503, content={"error": "TAVILY_API_KEY is not set"})

    # Whitelist: only known Tavily params are forwarded, never arbitrary
    # query string keys.
    body: dict[str, Any] = {"query": q, "include_raw_content": True}
    if max_results is not None:
        body["max_results"] = max_results
    if search_depth is not None:
        body["search_depth"] = search_depth
    if topic is not None:
        body["topic"] = topic
    if time_range is not None:
        body["time_range"] = time_range
    if include_answer is not None:
        body["include_answer"] = include_answer

    headers = {"Authorization": f"Bearer {TAVILY_API_KEY}", "Content-Type": "application/json"}
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(TAVILY_ENDPOINT, json=body, headers=headers)
            resp.raise_for_status()
            payload = resp.json()
    except httpx.HTTPStatusError as exc:
        # Surface Tavily's status (e.g. 401 bad key, 429 quota) so the gateway
        # logs make the cause obvious instead of a generic "unreachable".
        return JSONResponse(
            status_code=502,
            content={"error": f"tavily search returned {exc.response.status_code}"},
        )
    except httpx.HTTPError as exc:
        return JSONResponse(status_code=502, content={"error": f"tavily search failed: {exc}"})
    except ValueError as exc:
        # Invalid / non-JSON body from Tavily. Translate to a 502 rather than
        # letting it bubble up as a 500, so the gateway sees a consistent signal.
        return JSONResponse(status_code=502, content={"error": f"tavily search returned invalid JSON: {exc}"})

    if not isinstance(payload, dict):
        return JSONResponse(status_code=502, content={"error": "tavily search returned an unexpected shape"})

    hits = payload.get("results")
    if not isinstance(hits, list):
        # A missing/non-list `results` is an upstream contract break, not "no
        # hits", so surface it as a 502 instead of silently returning empty.
        return JSONResponse(status_code=502, content={"error": "tavily search returned an unexpected shape"})
    results: list[dict[str, Any]] = []
    for h in hits:
        if not isinstance(h, dict) or not h.get("url"):
            continue
        result: dict[str, Any] = {
            "url": h["url"],
            "title": h.get("title", ""),
            "content": h.get("content", ""),
        }
        # Tavily can return the full extracted page text; pass it through as
        # extracted_content so the gateway skips its own fetch + trafilatura.
        raw_content = h.get("raw_content")
        if raw_content:
            result["extracted_content"] = raw_content
        results.append(result)
    return JSONResponse(content={"results": results})
