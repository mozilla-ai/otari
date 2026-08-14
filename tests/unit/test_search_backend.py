"""Unit tests for the ``POST /v1/search`` backend.

Covers tool resolution against ``config.search_tools`` and the Exa and SearXNG
adapters: request translation from the LiteLLM-shaped request to each
provider's native shape, and response translation back. The provider is stubbed
with an ``httpx.MockTransport`` so no network call is made.
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
    DEFAULT_MAX_TOKENS_PER_PAGE,
    SearchProviderError,
    SearchQuery,
    SearchTool,
    SearchToolError,
    build_exa_payload,
    build_searxng_params,
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

_SEARXNG_TOOL = SearchTool(
    name="local",
    provider="searxng",
    api_base="http://searxng:8080",
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


def test_searxng_tool_resolves_without_an_api_key() -> None:
    """A self-hosted SearXNG authenticates with nothing, so no key is required."""
    tool = resolve_search_tool(_config(local={"provider": "searxng", "api_base": "http://searxng:8080/"}), "local")
    assert tool.provider == "searxng"
    assert tool.api_key is None
    assert tool.api_base == "http://searxng:8080"


def test_searxng_tool_inherits_the_web_search_url() -> None:
    """The in-loop backend's URL is the default, so one config line is enough."""
    config = GatewayConfig(
        search_tools={"local": {"provider": "searxng"}},
        web_search_url="http://searxng:8080/",
    )
    assert resolve_search_tool(config, "local").api_base == "http://searxng:8080"


def test_searxng_tool_api_base_wins_over_the_web_search_url() -> None:
    config = GatewayConfig(
        search_tools={"local": {"provider": "searxng", "api_base": "http://adapter:9000"}},
        web_search_url="http://searxng:8080",
    )
    assert resolve_search_tool(config, "local").api_base == "http://adapter:9000"


def test_searxng_tool_inherits_the_configured_engines() -> None:
    """Both surfaces query the same engines on the same backend."""
    config = GatewayConfig(
        search_tools={"local": {"provider": "searxng"}},
        web_search_url="http://searxng:8080",
        web_search_engines="duckduckgo,mojeek",
    )
    assert resolve_search_tool(config, "local").options["engines"] == "duckduckgo,mojeek"


def test_searxng_tool_engines_option_wins_over_the_configured_engines() -> None:
    config = GatewayConfig(
        search_tools={"local": {"provider": "searxng", "options": {"engines": "wikipedia"}}},
        web_search_url="http://searxng:8080",
        web_search_engines="duckduckgo,mojeek",
    )
    assert resolve_search_tool(config, "local").options["engines"] == "wikipedia"


def test_configured_engines_do_not_leak_into_an_exa_tool() -> None:
    config = GatewayConfig(
        search_tools={"exa": {"api_key": "k"}},
        web_search_engines="duckduckgo,mojeek",
    )
    assert "engines" not in resolve_search_tool(config, "exa").options


def test_searxng_tool_without_any_base_url_is_a_tool_error() -> None:
    """Nothing says where the backend is, so refuse rather than call nowhere."""
    with pytest.raises(SearchToolError, match="not configured correctly"):
        resolve_search_tool(_config(local={"provider": "searxng"}), "local")


def test_exa_tool_without_an_api_key_is_a_tool_error() -> None:
    """Exa still authenticates with a key: keyless is per-provider, not universal."""
    with pytest.raises(SearchToolError, match="not configured correctly"):
        resolve_search_tool(_config(exa={}), "exa")


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
async def test_searxng_search_normalizes_results(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["token"] = request.headers.get("x-gateway-token")
        return httpx.Response(200, json=_SEARXNG_BODY)

    _patch_transport(monkeypatch, handler)
    outcome = await run_search(_SEARXNG_TOOL, SearchQuery(query="otari"))

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
async def test_searxng_search_forwards_a_configured_key_as_the_gateway_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tool with a key fronts something that authenticates the gateway."""
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["token"] = request.headers.get("x-gateway-token")
        return httpx.Response(200, json={"results": []})

    _patch_transport(monkeypatch, handler)
    await run_search(replace(_SEARXNG_TOOL, api_key="adapter-secret"), SearchQuery(query="otari"))
    assert seen["token"] == "adapter-secret"


@pytest.mark.asyncio
async def test_searxng_search_caps_the_result_count(monkeypatch: pytest.MonkeyPatch) -> None:
    """SearXNG has no result-count param, so max_results is applied to the hits."""
    body = {"results": [{"url": f"https://example.com/{i}"} for i in range(12)]}
    _patch_transport(monkeypatch, lambda _r: httpx.Response(200, json=body))

    capped = await run_search(_SEARXNG_TOOL, SearchQuery(query="q", max_results=3))
    assert len(capped.results) == 3
    # With no max_results the module default applies rather than the whole page.
    defaulted = await run_search(_SEARXNG_TOOL, SearchQuery(query="q"))
    assert len(defaulted.results) == 10


@pytest.mark.asyncio
async def test_searxng_search_caps_extracted_page_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """An adapter can return a whole page, so the caller's per-page cap applies."""
    body = {"results": [{"url": "https://example.com/1", "extracted_content": "x" * 50_000}]}
    _patch_transport(monkeypatch, lambda _r: httpx.Response(200, json=body))

    requested = await run_search(_SEARXNG_TOOL, SearchQuery(query="q", max_tokens_per_page=100))
    assert requested.results[0].snippet == "x" * 400
    # With no request field the module default bounds it rather than nothing at all.
    defaulted = await run_search(_SEARXNG_TOOL, SearchQuery(query="q"))
    assert len(defaulted.results[0].snippet or "") == DEFAULT_MAX_TOKENS_PER_PAGE * 4


@pytest.mark.asyncio
async def test_searxng_search_applies_the_domain_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    """No SearXNG-side filter exists, so dropping it would widen the search."""
    body = {
        "results": [
            {"url": "https://arxiv.org/abs/1"},
            {"url": "https://export.arxiv.org/abs/2"},
            {"url": "https://spam.example/3"},
        ]
    }
    _patch_transport(monkeypatch, lambda _r: httpx.Response(200, json=body))

    restricted = await run_search(_SEARXNG_TOOL, SearchQuery(query="q", domain_filter=("arxiv.org",)))
    assert [hit.url for hit in restricted.results] == [
        "https://arxiv.org/abs/1",
        "https://export.arxiv.org/abs/2",
    ]

    excluded = await run_search(_SEARXNG_TOOL, SearchQuery(query="q", domain_filter=("-spam.example",)))
    assert [hit.url for hit in excluded.results] == [
        "https://arxiv.org/abs/1",
        "https://export.arxiv.org/abs/2",
    ]


@pytest.mark.asyncio
async def test_searxng_search_raises_on_upstream_error_status(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_transport(monkeypatch, lambda _r: httpx.Response(429, text="too many requests"))
    with pytest.raises(SearchProviderError, match="searxng search returned HTTP 429"):
        await run_search(_SEARXNG_TOOL, SearchQuery(query="otari"))


@pytest.mark.asyncio
async def test_searxng_search_raises_on_malformed_body(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_transport(monkeypatch, lambda _r: httpx.Response(200, json={"unexpected": True}))
    with pytest.raises(SearchProviderError, match="no 'results' list"):
        await run_search(_SEARXNG_TOOL, SearchQuery(query="otari"))


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
        ({"exa": {}}, "api_key is required for provider 'exa'"),
        ({"searxng": {"api_base": "http://searxng:8080"}}, None),
        ({"local": {"provider": "searxng", "api_base": "http://searxng:8080"}}, None),
        # A missing backend URL is reported as a startup warning, not raised: see
        # the search_tools_without_backend_url tests below.
        ({"searxng": {}}, None),
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


def test_validate_accepts_a_searxng_tool_that_inherits_the_web_search_url() -> None:
    """The api_base a searxng tool omits is the one the in-loop tool already uses."""
    config = GatewayConfig(
        search_tools={"local": {"provider": "searxng"}},
        web_search_url="http://searxng:8080",
    )
    config.validate_search_tools()
    assert config.search_tools_without_backend_url() == []


def test_a_searxng_tool_with_no_backend_url_anywhere_is_reported() -> None:
    """Startup warns instead of failing: a dashboard-stored URL lands after load."""
    config = GatewayConfig(search_tools={"local": {"provider": "searxng"}, "exa": {"api_key": "k"}})
    assert config.search_tools_without_backend_url() == ["local"]


def test_a_searxng_tool_with_its_own_api_base_is_not_reported() -> None:
    config = GatewayConfig(search_tools={"local": {"provider": "searxng", "api_base": "http://adapter:9000"}})
    assert config.search_tools_without_backend_url() == []
