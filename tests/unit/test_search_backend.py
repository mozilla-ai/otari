"""Unit tests for the direct search endpoint's old SearXNG client.

Covers request translation from the LiteLLM-shaped request to SearXNG's query
params, and response translation back, until any-search has a SearXNG adapter.
The provider is stubbed with an ``httpx.MockTransport`` on the client the caller
passes, as the route passes the app's web provider client, so no network call is
made. Resolution and the LiteLLM mapping are tested with the resolver and with
``_litellm``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any

import httpx
import pytest

from gateway.services.search_backend import (
    SearchOutcome,
    SearchProviderError,
    SearchTool,
    build_searxng_params,
    run_search,
)
from gateway.services.tools import DEFAULT_MAX_TOKENS_PER_PAGE, SearchQuery

_SEARXNG_TOOL = SearchTool(
    name="local",
    provider="searxng",
    api_base="http://searxng:8080",
    timeout_s=30.0,
    options={},
)


async def _run_search(
    handler: Callable[[httpx.Request], httpx.Response], tool: SearchTool, query: SearchQuery
) -> SearchOutcome:
    """Run one search on a client whose provider answers through ``handler``."""
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        return await run_search(tool, query, client=client)


def _replying(response: httpx.Response) -> Callable[[httpx.Request], httpx.Response]:
    return lambda _request: response


# --------------------------------------------------------------------------- #
# SearXNG request translation
# --------------------------------------------------------------------------- #


def test_searxng_params_defaults() -> None:
    params = build_searxng_params(_SEARXNG_TOOL, SearchQuery(query="otari gateway"))
    assert params == {"q": "otari gateway", "format": "json", "max_results": 10}


def test_searxng_params_forward_scalar_tool_options() -> None:
    tool = replace(
        _SEARXNG_TOOL,
        options={"engines": "duckduckgo,mojeek", "safesearch": 1, "image_proxy": True, "extra": {"nope": 1}},
    )
    params = build_searxng_params(tool, SearchQuery(query="q"))
    assert params["engines"] == "duckduckgo,mojeek"
    assert params["safesearch"] == 1
    assert params["image_proxy"] == "true"
    # Complex values cannot ride in a query string, so they are dropped.
    assert "extra" not in params


def test_searxng_params_never_let_options_displace_the_query() -> None:
    tool = replace(_SEARXNG_TOOL, options={"q": "operator-injected", "format": "html"})
    params = build_searxng_params(tool, SearchQuery(query="caller wins"))
    assert params["q"] == "caller wins"
    assert params["format"] == "json"


def test_searxng_params_forward_the_country() -> None:
    params = build_searxng_params(_SEARXNG_TOOL, SearchQuery(query="q", country="US"))
    assert params["country"] == "US"


def test_searxng_params_forward_the_result_count() -> None:
    """An adapter that reads it would otherwise cap the page at its own default."""
    assert build_searxng_params(_SEARXNG_TOOL, SearchQuery(query="q", max_results=7))["max_results"] == 7
    assert build_searxng_params(_SEARXNG_TOOL, SearchQuery(query="q"))["max_results"] == 10


def test_searxng_params_result_count_is_not_overridable_by_options() -> None:
    """The count is the gateway's: an option cannot ask upstream for more."""
    tool = replace(_SEARXNG_TOOL, options={"max_results": 500})
    assert build_searxng_params(tool, SearchQuery(query="q", max_results=5))["max_results"] == 5


# --------------------------------------------------------------------------- #
# SearXNG response translation
# --------------------------------------------------------------------------- #


_SEARXNG_BODY: dict[str, Any] = {
    "query": "otari",
    "results": [
        {
            "url": "https://example.com/otari",
            "title": "Otari",
            "content": "An OpenAI-compatible LLM gateway.",
            "publishedDate": "2026-01-02T00:00:00.000Z",
        },
        {"title": "No URL", "content": "dropped"},
        {
            "url": "https://docs.example.com/guide",
            "title": "Guide",
            "content": "snippet",
            "extracted_content": "full page text",
            "published_date": "2026-02-03",
        },
    ],
}


@pytest.mark.asyncio
async def test_searxng_search_normalizes_results() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["token"] = request.headers.get("x-gateway-token")
        return httpx.Response(200, json=_SEARXNG_BODY)

    outcome = await _run_search(handler, _SEARXNG_TOOL, SearchQuery(query="otari"))

    assert seen["url"] == "http://searxng:8080/search?max_results=10&q=otari&format=json"
    # A self-hosted SearXNG has no credential to send.
    assert seen["token"] is None
    # SearXNG reports no charge, so the tool's configured flat rate bills.
    assert outcome.cost_usd is None
    # The result without a URL is dropped: it cannot be cited or followed.
    assert [hit.url for hit in outcome.results] == [
        "https://example.com/otari",
        "https://docs.example.com/guide",
    ]
    assert outcome.results[0].title == "Otari"
    assert outcome.results[0].snippet == "An OpenAI-compatible LLM gateway."
    assert outcome.results[0].date == "2026-01-02T00:00:00.000Z"
    # An adapter that already had the page text wins over the engine snippet.
    assert outcome.results[1].snippet == "full page text"
    assert outcome.results[1].date == "2026-02-03"


@pytest.mark.asyncio
async def test_searxng_search_forwards_a_configured_key_as_the_gateway_token() -> None:
    """A tool with a key fronts something that authenticates the gateway."""
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["token"] = request.headers.get("x-gateway-token")
        return httpx.Response(200, json={"results": []})

    tool = replace(_SEARXNG_TOOL, api_base="https://searxng.example.com", api_key="adapter-secret")
    await _run_search(handler, tool, SearchQuery(query="otari"))
    assert seen["token"] == "adapter-secret"


@pytest.mark.asyncio
async def test_run_search_never_sends_a_credential_over_http() -> None:
    called = False

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200, json={"results": []})

    tool = replace(_SEARXNG_TOOL, api_key="adapter-secret")
    with pytest.raises(SearchProviderError, match="api_base must use https when api_key is set"):
        await _run_search(handler, tool, SearchQuery(query="otari"))
    assert called is False


@pytest.mark.asyncio
async def test_searxng_search_caps_the_result_count() -> None:
    """SearXNG has no result-count param, so max_results is applied to the hits."""
    body = {"results": [{"url": f"https://example.com/{i}"} for i in range(12)]}
    answer = _replying(httpx.Response(200, json=body))

    capped = await _run_search(answer, _SEARXNG_TOOL, SearchQuery(query="q", max_results=3))
    assert len(capped.results) == 3
    # With no max_results the module default applies rather than the whole page.
    defaulted = await _run_search(answer, _SEARXNG_TOOL, SearchQuery(query="q"))
    assert len(defaulted.results) == 10


@pytest.mark.asyncio
async def test_searxng_search_caps_extracted_page_text() -> None:
    """An adapter can return a whole page, so the caller's per-page cap applies."""
    body = {"results": [{"url": "https://example.com/1", "extracted_content": "x" * 50_000}]}
    answer = _replying(httpx.Response(200, json=body))

    requested = await _run_search(answer, _SEARXNG_TOOL, SearchQuery(query="q", max_tokens_per_page=100))
    assert requested.results[0].snippet == "x" * 400
    # With no request field the module default bounds it rather than nothing at all.
    defaulted = await _run_search(answer, _SEARXNG_TOOL, SearchQuery(query="q"))
    assert len(defaulted.results[0].snippet or "") == DEFAULT_MAX_TOKENS_PER_PAGE * 4


@pytest.mark.asyncio
async def test_searxng_search_applies_the_domain_filter() -> None:
    """No SearXNG-side filter exists, so dropping it would widen the search."""
    body = {
        "results": [
            {"url": "https://arxiv.org/abs/1"},
            {"url": "https://export.arxiv.org/abs/2"},
            {"url": "https://spam.example/3"},
        ]
    }
    answer = _replying(httpx.Response(200, json=body))

    restricted = await _run_search(answer, _SEARXNG_TOOL, SearchQuery(query="q", domain_filter=("arxiv.org",)))
    assert [hit.url for hit in restricted.results] == [
        "https://arxiv.org/abs/1",
        "https://export.arxiv.org/abs/2",
    ]

    excluded = await _run_search(answer, _SEARXNG_TOOL, SearchQuery(query="q", domain_filter=("-spam.example",)))
    assert [hit.url for hit in excluded.results] == [
        "https://arxiv.org/abs/1",
        "https://export.arxiv.org/abs/2",
    ]


@pytest.mark.asyncio
async def test_searxng_search_raises_on_upstream_error_status() -> None:
    answer = _replying(httpx.Response(429, text="too many requests"))
    with pytest.raises(SearchProviderError, match="searxng search returned HTTP 429"):
        await _run_search(answer, _SEARXNG_TOOL, SearchQuery(query="otari"))


@pytest.mark.asyncio
async def test_searxng_search_raises_on_malformed_body() -> None:
    answer = _replying(httpx.Response(200, json={"unexpected": True}))
    with pytest.raises(SearchProviderError, match="no 'results' list"):
        await _run_search(answer, _SEARXNG_TOOL, SearchQuery(query="otari"))
