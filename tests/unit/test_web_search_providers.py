"""Unit tests for the first-party Tavily and Brave web-search providers.

Covers the translation both ways (a query onto each provider's native request,
its answer back onto SearXNG-shaped hits) and the failure modes that used to be
an adapter container's job, with the HTTP layer mocked so the suite needs no
provider key and no network.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from gateway.services.web_search_providers import WebSearchProviderError, provider_search

TAVILY_HOST = "api.tavily.com"
BRAVE_HOST = "api.search.brave.com"


class _Recorder(httpx.AsyncBaseTransport):
    """Answers every request with one canned response, keeping what was asked."""

    def __init__(self, response: httpx.Response | Exception) -> None:
        self._response = response
        self.requests: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


def _client(response: httpx.Response | Exception) -> tuple[httpx.AsyncClient, _Recorder]:
    recorder = _Recorder(response)
    return httpx.AsyncClient(transport=recorder), recorder


TAVILY_OK = {
    "results": [
        {
            "url": "https://example.com/a",
            "title": "A",
            "content": "snippet a",
            "raw_content": "the whole page of a",
        },
        {"url": "https://example.org/b", "title": "B", "content": "snippet b"},
        {"title": "no url", "content": "dropped"},
    ]
}

BRAVE_OK = {
    "web": {
        "results": [
            {"url": "https://example.com/a", "title": "A", "description": "snippet a"},
            {"title": "no url", "description": "dropped"},
        ]
    }
}

BRAVE_DATED = {
    "web": {
        "results": [
            {"url": "https://a.example", "title": "A", "description": "a", "page_age": "2026-08-30T00:00:00"},
            {"url": "https://b.example", "title": "B", "description": "b", "age": "3 days ago"},
            {"url": "https://c.example", "title": "C", "description": "c"},
        ]
    }
}


@pytest.mark.asyncio
async def test_tavily_maps_results_and_passes_raw_content_through() -> None:
    client, recorder = _client(httpx.Response(200, json=TAVILY_OK))
    async with client:
        results = await provider_search(provider="tavily", api_key="tvly-x", query="claude code", client=client)

    assert results == [
        {
            "url": "https://example.com/a",
            "title": "A",
            "content": "snippet a",
            "extracted_content": "the whole page of a",
        },
        {"url": "https://example.org/b", "title": "B", "content": "snippet b"},
    ]
    request = recorder.requests[0]
    assert request.url.host == TAVILY_HOST
    assert request.headers["authorization"] == "Bearer tvly-x"


@pytest.mark.asyncio
async def test_tavily_forwards_only_the_options_it_understands() -> None:
    client, recorder = _client(httpx.Response(200, json=TAVILY_OK))
    async with client:
        await provider_search(
            provider="tavily",
            api_key="tvly-x",
            query="q",
            options={"search_depth": "advanced", "max_results": 3, "engines": "google", "not_a_knob": 1},
            client=client,
        )

    body: dict[str, Any] = json.loads(recorder.requests[0].content)
    assert body["search_depth"] == "advanced"
    assert body["max_results"] == 3
    assert "engines" not in body
    assert "not_a_knob" not in body


@pytest.mark.asyncio
async def test_brave_maps_results_and_leaves_extraction_to_the_caller() -> None:
    client, recorder = _client(httpx.Response(200, json=BRAVE_OK))
    async with client:
        results = await provider_search(provider="brave", api_key="brv-x", query="claude code", client=client)

    assert results == [{"url": "https://example.com/a", "title": "A", "content": "snippet a"}]
    request = recorder.requests[0]
    assert request.url.host == BRAVE_HOST
    assert request.headers["x-subscription-token"] == "brv-x"


@pytest.mark.asyncio
async def test_brave_clamps_max_results_to_the_documented_ceiling() -> None:
    client, recorder = _client(httpx.Response(200, json=BRAVE_OK))
    async with client:
        await provider_search(provider="brave", api_key="brv-x", query="q", options={"max_results": 500}, client=client)

    assert recorder.requests[0].url.params["count"] == "20"


@pytest.mark.asyncio
async def test_brave_reads_a_missing_web_block_as_no_hits() -> None:
    client, _ = _client(httpx.Response(200, json={}))
    async with client:
        assert await provider_search(provider="brave", api_key="brv-x", query="q", client=client) == []


@pytest.mark.asyncio
async def test_tavily_missing_results_list_is_an_error_not_an_empty_search() -> None:
    client, _ = _client(httpx.Response(200, json={"answer": "no results key"}))
    async with client:
        with pytest.raises(WebSearchProviderError):
            await provider_search(provider="tavily", api_key="tvly-x", query="q", client=client)


@pytest.mark.asyncio
async def test_upstream_error_status_becomes_a_provider_error() -> None:
    client, _ = _client(httpx.Response(429, text="slow down"))
    async with client:
        with pytest.raises(WebSearchProviderError, match="429"):
            await provider_search(provider="tavily", api_key="tvly-x", query="q", client=client)


@pytest.mark.asyncio
async def test_transport_failure_becomes_a_provider_error() -> None:
    client, _ = _client(httpx.ConnectError("no route"))
    async with client:
        with pytest.raises(WebSearchProviderError, match="could not be reached"):
            await provider_search(provider="brave", api_key="brv-x", query="q", client=client)


@pytest.mark.asyncio
async def test_non_json_body_becomes_a_provider_error() -> None:
    client, _ = _client(httpx.Response(200, text="<html>maintenance</html>"))
    async with client:
        with pytest.raises(WebSearchProviderError, match="not JSON"):
            await provider_search(provider="tavily", api_key="tvly-x", query="q", client=client)


@pytest.mark.asyncio
async def test_unknown_provider_is_refused() -> None:
    client, _ = _client(httpx.Response(200, json={}))
    async with client:
        with pytest.raises(ValueError, match="web_search_provider"):
            await provider_search(provider="exa", api_key="k", query="q", client=client)


@pytest.mark.parametrize(
    ("time_range", "expected"),
    [("day", "pd"), ("d", "pd"), ("week", "pw"), ("w", "pw"), ("month", "pm"), ("year", "py"), ("Y", "py")],
)
@pytest.mark.asyncio
async def test_brave_sends_time_range_as_freshness(time_range: str, expected: str) -> None:
    """The adapter this replaces mapped the same vocabulary, so a stored
    ``provider_options`` keeps filtering by recency instead of silently going wide."""
    client, recorder = _client(httpx.Response(200, json=BRAVE_OK))
    async with client:
        await provider_search(
            provider="brave", api_key="brv-x", query="q", options={"time_range": time_range}, client=client
        )

    assert recorder.requests[0].url.params["freshness"] == expected


@pytest.mark.asyncio
async def test_brave_sends_no_freshness_without_a_time_range() -> None:
    client, recorder = _client(httpx.Response(200, json=BRAVE_OK))
    async with client:
        await provider_search(provider="brave", api_key="brv-x", query="q", client=client)

    assert "freshness" not in recorder.requests[0].url.params


@pytest.mark.asyncio
async def test_brave_carries_the_recency_signal_back() -> None:
    """``published_date`` is what the model-facing result block renders as a date.

    ``page_age`` is preferred over ``age``: they are different formats, not
    alternatives, and only the first is parseable.
    """
    client, _ = _client(httpx.Response(200, json=BRAVE_DATED))
    async with client:
        results = await provider_search(provider="brave", api_key="brv-x", query="q", client=client)

    assert [hit.get("published_date") for hit in results] == ["2026-08-30T00:00:00", "3 days ago", None]


@pytest.mark.asyncio
async def test_an_upstream_error_body_never_reaches_the_message() -> None:
    """A provider's error text can echo back what was sent to it, and the message
    reaches the request log and an OTel span. The status is what identifies the fault."""
    client, _ = _client(httpx.Response(401, text="invalid api key tvly-secret for account acme"))
    async with client:
        with pytest.raises(WebSearchProviderError) as raised:
            await provider_search(provider="tavily", api_key="tvly-secret", query="q", client=client)

    assert "401" in str(raised.value)
    assert "tvly-secret" not in str(raised.value)
    assert "acme" not in str(raised.value)


@pytest.mark.asyncio
async def test_tavily_clamps_max_results_to_the_documented_ceiling() -> None:
    """``options`` is the opaque ``provider_options`` bag and can name any number,
    which Tavily answers with a 400 rather than a clamp."""
    client, recorder = _client(httpx.Response(200, json=TAVILY_OK))
    async with client:
        await provider_search(
            provider="tavily", api_key="tvly-x", query="q", options={"max_results": 500}, client=client
        )

    assert json.loads(recorder.requests[0].content)["max_results"] == 20


@pytest.mark.parametrize("provider", ["tavily", "brave"])
@pytest.mark.asyncio
async def test_a_caller_ceiling_reaches_the_provider(provider: str) -> None:
    """``WebSearchBackend`` passes its resolved ceiling as the default ``max_results``.

    Without it the provider serves its own default and the ceiling is only a
    post-hoc slice, so raising ``web_search_max_results`` returned no more hits.
    """
    payload = TAVILY_OK if provider == "tavily" else BRAVE_OK
    client, recorder = _client(httpx.Response(200, json=payload))
    async with client:
        await provider_search(
            provider=provider, api_key="k", query="q", options={"max_results": 15}, client=client
        )

    request = recorder.requests[0]
    asked = json.loads(request.content)["max_results"] if provider == "tavily" else request.url.params["count"]
    assert int(asked) == 15
