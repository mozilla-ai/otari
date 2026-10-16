"""The direct endpoint's response against LiteLLM's search types (design J).

A caller moving off the LiteLLM proxy parses the response with LiteLLM's
``SearchResponse``, so the default response must validate against it: every
result carries ``title``, ``url`` and ``snippet`` as strings, which a provider
hit without a title or a snippet used to break with a null. The two models
below mirror LiteLLM's, from ``litellm/llms/base_llm/search/transformation.py``;
LiteLLM allows extra fields, and otari adds only ``search_tool``, and ``raw``
when asked.
"""

from typing import Any

import httpx
import pytest
from pydantic import BaseModel, ConfigDict

from gateway.core.config import GatewayConfig
from gateway.schemas.tools import SearchRequest
from gateway.services.tools import pick_direct_search_instance, resolve_direct_search, run_direct_search


class LiteLLMSearchResult(BaseModel):
    """LiteLLM's ``SearchResult``."""

    model_config = ConfigDict(extra="allow")

    title: str
    url: str
    snippet: str
    date: str | None = None
    last_updated: str | None = None


class LiteLLMSearchResponse(BaseModel):
    """LiteLLM's ``SearchResponse``."""

    model_config = ConfigDict(extra="allow")

    results: list[LiteLLMSearchResult]
    object: str = "search"


# A hit with no title, no snippet and no date, and one with all of them.
_HITS = [
    {"url": "https://example.com/bare"},
    {"url": "https://example.com/full", "title": "Full", "snippet": "s", "published": "2026-01-02T00:00:00Z"},
]


async def _response(**request: Any) -> dict[str, Any]:
    config = GatewayConfig(search_tools={"fake-search": {"provider": "fake", "options": {"hits": _HITS}}})
    instance = pick_direct_search_instance(config, None)
    search = resolve_direct_search(config, instance, None, SearchRequest(query="q", **request), named=False)
    async with httpx.AsyncClient() as client:
        result = await run_direct_search(search, client)
    response: dict[str, Any] = result.response.model_dump(mode="json")
    return response


@pytest.mark.asyncio
async def test_the_default_response_is_litellms() -> None:
    response = await _response()

    parsed = LiteLLMSearchResponse.model_validate(response, strict=True)

    assert parsed.object == "search"
    assert [result.title for result in parsed.results] == ["", "Full"]
    assert [result.snippet for result in parsed.results] == ["", "s"]
    assert [result.date for result in parsed.results] == [None, "2026-01-02T00:00:00+00:00"]
    # Nothing beyond LiteLLM's own fields on a result, and only the tool's name beside them.
    assert {key for result in response["results"] for key in result} == set(LiteLLMSearchResult.model_fields)
    assert set(response) == {"object", "search_tool", "results"}


@pytest.mark.asyncio
async def test_raw_rides_beside_litellms_fields_only_when_asked() -> None:
    response = await _response(include_raw=True)

    LiteLLMSearchResponse.model_validate(response, strict=True)
    assert set(response) == {"object", "search_tool", "results", "raw"}
    assert response["raw"]["results"] == _HITS
