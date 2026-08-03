"""Unit tests for the ``POST /v1/search`` backend.

Covers tool resolution against ``config.search_tools`` and the Exa adapter:
request translation from the LiteLLM-shaped request to Exa's native body, and
response translation back. The provider is stubbed with an
``httpx.MockTransport`` so no network call is made.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from dataclasses import replace
from typing import Any

import httpx
import pytest
import pytest_asyncio

from gateway.core.config import GatewayConfig
from gateway.services.search_backend import (
    SearchProviderError,
    SearchQuery,
    SearchTool,
    SearchToolError,
    build_exa_payload,
    close_search_client,
    get_search_client,
    resolve_search_tool,
    run_search,
)

_EXA_TOOL = SearchTool(
    name="exa-search",
    provider="exa",
    api_key="exa-secret",
    api_base="https://api.exa.ai",
    timeout_s=30.0,
    options={},
)


def _config(**search_tools: dict[str, Any]) -> GatewayConfig:
    return GatewayConfig(search_tools=dict(search_tools))


def _patch_transport(monkeypatch: pytest.MonkeyPatch, handler: Callable[[httpx.Request], httpx.Response]) -> None:
    """Replace the module's ``httpx.AsyncClient`` with one backed by ``handler``."""
    transport = httpx.MockTransport(handler)
    real_async_client = httpx.AsyncClient  # capture before patching to avoid recursion

    def factory(*_args: object, **_kwargs: object) -> httpx.AsyncClient:
        return real_async_client(transport=transport)

    monkeypatch.setattr("gateway.services.search_backend.httpx.AsyncClient", factory)


@pytest_asyncio.fixture(autouse=True)
async def _fresh_pooled_client() -> AsyncIterator[None]:
    """Drop the module's pooled client around each test.

    The client is process-wide by design, so without this a mock transport (and
    the event loop it was built on) would leak into the next test.
    """
    await close_search_client()
    yield
    await close_search_client()


# --------------------------------------------------------------------------- #
# Tool resolution
# --------------------------------------------------------------------------- #


def test_sole_tool_resolves_without_a_name() -> None:
    tool = resolve_search_tool(_config(**{"exa-search": {"provider": "exa", "api_key": "k"}}), None)
    assert tool.name == "exa-search"
    assert tool.provider == "exa"
    assert tool.api_base == "https://api.exa.ai"


def test_provider_defaults_to_the_tool_name() -> None:
    tool = resolve_search_tool(_config(exa={"api_key": "k"}), "exa")
    assert tool.provider == "exa"


def test_api_base_override_is_normalized() -> None:
    tool = resolve_search_tool(_config(exa={"api_key": "k", "api_base": "https://proxy.internal/"}), "exa")
    assert tool.api_base == "https://proxy.internal"


def test_no_configured_tools_is_a_tool_error() -> None:
    with pytest.raises(SearchToolError, match="No search tools are configured"):
        resolve_search_tool(_config(), None)


def test_unknown_tool_name_is_a_tool_error() -> None:
    with pytest.raises(SearchToolError, match="Unknown search tool 'nope'"):
        resolve_search_tool(_config(exa={"api_key": "k"}), "nope")


def test_omitted_name_with_several_tools_is_a_tool_error() -> None:
    config = _config(exa={"api_key": "k"}, **{"exa-fast": {"provider": "exa", "api_key": "k"}})
    with pytest.raises(SearchToolError, match="Several search tools are configured"):
        resolve_search_tool(config, None)


# --------------------------------------------------------------------------- #
# Exa request translation
# --------------------------------------------------------------------------- #


def test_payload_defaults() -> None:
    payload = build_exa_payload(_EXA_TOOL, SearchQuery(query="otari gateway"))
    assert payload["query"] == "otari gateway"
    assert payload["numResults"] == 10
    assert payload["contents"] == {"text": {"maxCharacters": 4096}}
    assert "includeDomains" not in payload
    assert "userLocation" not in payload


def test_payload_maps_request_fields() -> None:
    payload = build_exa_payload(
        _EXA_TOOL,
        SearchQuery(
            query="q",
            max_results=3,
            domain_filter=("arxiv.org", "-spam.example"),
            country="US",
            max_tokens_per_page=100,
        ),
    )
    assert payload["numResults"] == 3
    assert payload["includeDomains"] == ["arxiv.org"]
    assert payload["excludeDomains"] == ["spam.example"]
    assert payload["userLocation"] == "US"
    assert payload["contents"]["text"]["maxCharacters"] == 400


def test_payload_clamps_result_count_from_tool_options() -> None:
    """A tool's own numResults cannot escape the cap the request field enforces."""
    tool = replace(_EXA_TOOL, options={"numResults": 500})
    assert build_exa_payload(tool, SearchQuery(query="q"))["numResults"] == 20


def test_payload_clamps_page_size_to_exas_ceiling() -> None:
    payload = build_exa_payload(_EXA_TOOL, SearchQuery(query="q", max_tokens_per_page=100_000))
    assert payload["contents"]["text"]["maxCharacters"] == 10_000


def test_tool_options_are_defaults_and_never_displace_the_query() -> None:
    tool = replace(_EXA_TOOL, options={"type": "fast", "query": "operator-injected"})
    payload = build_exa_payload(tool, SearchQuery(query="caller wins"))
    assert payload["type"] == "fast"
    assert payload["query"] == "caller wins"


def test_tool_pinned_contents_survive_when_the_caller_asks_for_no_page_size() -> None:
    tool = replace(_EXA_TOOL, options={"contents": {"text": {"maxCharacters": 800}}})
    payload = build_exa_payload(tool, SearchQuery(query="q"))
    assert payload["contents"]["text"]["maxCharacters"] == 800


def test_caller_page_size_overrides_tool_pinned_contents() -> None:
    """The caller's size wins; 800 would survive if the override were a no-op."""
    tool = replace(_EXA_TOOL, options={"contents": {"text": {"maxCharacters": 800}}})
    payload = build_exa_payload(tool, SearchQuery(query="q", max_tokens_per_page=50))
    assert payload["contents"]["text"]["maxCharacters"] == 200


@pytest.mark.parametrize("pinned", [None, False])
def test_tool_can_opt_out_of_page_content(pinned: object) -> None:
    """Pinning contents.text to null/false suppresses the per-page content charge."""
    tool = replace(_EXA_TOOL, options={"contents": {"text": pinned, "highlights": True}})
    payload = build_exa_payload(tool, SearchQuery(query="q"))
    assert payload["contents"] == {"highlights": True}


def test_opting_out_of_page_content_drops_an_empty_contents_block() -> None:
    tool = replace(_EXA_TOOL, options={"contents": {"text": None}})
    assert "contents" not in build_exa_payload(tool, SearchQuery(query="q"))


def test_page_content_opt_out_survives_a_caller_page_size() -> None:
    """The opt-out is the operator's: a request cannot re-enable the charge."""
    tool = replace(_EXA_TOOL, options={"contents": {"text": None, "highlights": True}})
    payload = build_exa_payload(tool, SearchQuery(query="q", max_tokens_per_page=256))
    assert payload["contents"] == {"highlights": True}


def test_caller_page_size_keeps_the_tools_other_text_options() -> None:
    """Only the size is the caller's to set; pinned siblings are merged, not replaced."""
    tool = replace(
        _EXA_TOOL,
        options={"contents": {"text": {"maxCharacters": 800, "verbosity": "full"}, "highlights": True}},
    )
    payload = build_exa_payload(tool, SearchQuery(query="q", max_tokens_per_page=50))
    assert payload["contents"]["text"] == {"maxCharacters": 200, "verbosity": "full"}
    assert payload["contents"]["highlights"] is True


# --------------------------------------------------------------------------- #
# Exa response translation
# --------------------------------------------------------------------------- #


_EXA_BODY: dict[str, Any] = {
    "requestId": "req-1",
    "results": [
        {
            "title": "Otari",
            "url": "https://example.com/otari",
            "text": "An OpenAI-compatible LLM gateway.",
            "publishedDate": "2026-01-02T00:00:00.000Z",
        },
        {"title": "No URL", "text": "dropped"},
        {"url": "https://example.com/highlights", "highlights": ["first hit. ", "second hit."]},
    ],
    "costDollars": {"total": 0.007, "search": {"neural": 0.007}},
}


@pytest.mark.asyncio
async def test_run_search_normalizes_results_and_cost(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["api_key"] = request.headers.get("x-api-key")
        return httpx.Response(200, json=_EXA_BODY)

    _patch_transport(monkeypatch, handler)
    outcome = await run_search(_EXA_TOOL, SearchQuery(query="otari"))

    assert seen["url"] == "https://api.exa.ai/search"
    assert seen["api_key"] == "exa-secret"
    assert outcome.cost_usd == 0.007
    # The result without a URL is dropped: it cannot be cited or followed.
    assert [hit.url for hit in outcome.results] == [
        "https://example.com/otari",
        "https://example.com/highlights",
    ]
    assert outcome.results[0].title == "Otari"
    assert outcome.results[0].snippet == "An OpenAI-compatible LLM gateway."
    assert outcome.results[0].date == "2026-01-02T00:00:00.000Z"
    # Highlights stand in as the snippet when no page text was returned.
    assert outcome.results[1].snippet == "first hit. second hit."


@pytest.mark.asyncio
async def test_run_search_reports_no_cost_when_the_provider_omits_it(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_transport(monkeypatch, lambda _r: httpx.Response(200, json={"results": []}))
    outcome = await run_search(_EXA_TOOL, SearchQuery(query="otari"))
    assert outcome.results == []
    assert outcome.cost_usd is None


@pytest.mark.asyncio
async def test_run_search_raises_on_upstream_error_status(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_transport(monkeypatch, lambda _r: httpx.Response(401, text="invalid api key"))
    with pytest.raises(SearchProviderError, match="HTTP 401"):
        await run_search(_EXA_TOOL, SearchQuery(query="otari"))


@pytest.mark.asyncio
async def test_run_search_raises_on_malformed_body(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_transport(monkeypatch, lambda _r: httpx.Response(200, json={"unexpected": True}))
    with pytest.raises(SearchProviderError, match="no 'results' list"):
        await run_search(_EXA_TOOL, SearchQuery(query="otari"))


@pytest.mark.asyncio
async def test_run_search_raises_when_the_provider_is_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    _patch_transport(monkeypatch, handler)
    with pytest.raises(SearchProviderError, match="request failed"):
        await run_search(_EXA_TOOL, SearchQuery(query="otari"))


# --------------------------------------------------------------------------- #
# Pooled client
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_searches_share_one_pooled_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """Connections survive between searches instead of a handshake per request."""
    timeouts: list[object] = []

    def handler(request: httpx.Request) -> httpx.Response:
        timeouts.append(request.extensions.get("timeout", {}).get("connect"))
        return httpx.Response(200, json={"results": []})

    _patch_transport(monkeypatch, handler)
    await run_search(_EXA_TOOL, SearchQuery(query="one"))
    first = get_search_client()
    await run_search(replace(_EXA_TOOL, timeout_s=5.0), SearchQuery(query="two"))

    assert get_search_client() is first
    # The client is shared, so the per-tool timeout has to ride on the request.
    assert timeouts == [30.0, 5.0]


@pytest.mark.asyncio
async def test_closing_the_pooled_client_lets_the_next_search_rebuild_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shutdown closes the client; a later search must not reuse the closed one."""
    _patch_transport(monkeypatch, lambda _r: httpx.Response(200, json={"results": []}))
    await run_search(_EXA_TOOL, SearchQuery(query="one"))
    closed = get_search_client()
    await close_search_client()
    assert closed.is_closed

    await run_search(_EXA_TOOL, SearchQuery(query="two"))
    assert get_search_client() is not closed


# --------------------------------------------------------------------------- #
# Startup validation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "search_tools,expected",
    [
        ({"exa": {"api_key": "k"}}, None),
        ({"web": {"provider": "exa", "api_key": "k"}}, None),
        ({"web": {"api_key": "k"}}, "is not a supported search provider"),
        ({"exa": {}}, "api_key is required"),
        ({"exa": {"api_key": "k", "options": "nope"}}, "options must be a mapping"),
        ({"exa": {"api_key": "k", "timeout": "soon"}}, "timeout must be a number"),
        ({"exa": {"api_key": "k", "timeout": 0}}, "timeout must be greater than 0"),
        ({"exa": {"api_key": "k", "timeout": -5}}, "timeout must be greater than 0"),
        ({"exa": {"api_key": "k", "timeout": 0.5}}, None),
        ({"ex/a": {"provider": "exa", "api_key": "k"}}, "must not contain '/'"),
    ],
)
def test_validate_search_tools(search_tools: dict[str, Any], expected: str | None) -> None:
    config = GatewayConfig(search_tools=search_tools)
    if expected is None:
        config.validate_search_tools()
        return
    with pytest.raises(ValueError, match=expected):
        config.validate_search_tools()
