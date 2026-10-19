"""The web search tool on any-search: the search backend, its enrichment, and the cap it shares with fetch.

The searches run on any-search's ``fake`` provider and the provider enrichments
on any-fetch's. The built-in fetcher is stubbed at ``WebFetchService.fetch`` for
every case, so no case reaches the network.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from any_fetch.providers.fake import FakeProvider as FakeFetchProvider
from any_search.providers.fake import FakeProvider as FakeSearchProvider
from gateway.api.routes._pipeline import ToolContext
from gateway.core.config import GatewayConfig
from gateway.exceptions.tools_exceptions import WebSearchOptionRefusedError
from gateway.models.tools import ResolvedWebSearchConfig
from gateway.services.tool_usage import ToolUsageTally
from gateway.services.tools import (
    WEB_FETCH_TOOL_NAME,
    WEB_RETRIEVAL_RESULT_MAX_BYTES,
    WEB_SEARCH_PURPOSE_HINT,
    WEB_SEARCH_TOOL_NAME,
    InLoopSearch,
    WebFetchBackend,
    WebSearchBackend,
    WebToolCallCap,
    WebToolFailure,
    build_web_search_backends,
    resolve_in_loop_search,
)
from gateway.services.traces import RequestTrace
from gateway.services.web_fetch_service import WebFetchResult, WebFetchService
from gateway.services.web_retrieval_backend import WebRetrievalBackend
from gateway.services.web_retrieval_network import PinnedAsyncHTTPTransport, TrustedProxyAsyncHTTPTransport
from gateway.services.web_retrieval_policy import DomainPolicy, canonicalize_domain_rules, canonicalize_web_url
from gateway.types.tool_charges import LabelCalls

_SEARCH = "fake-search"
_FETCH = "fake-fetch"
_LABEL = f"fake:{_SEARCH}"
_HITS: list[dict[str, Any]] = [
    {"url": "https://example.com/a", "title": "A", "snippet": "Snippet A.", "published": "2026-01-15T09:30:00Z"},
    {"url": "https://example.org/b", "title": "B", "snippet": "Snippet B.", "published": "2026-02-01"},
    {"url": "https://example.net/c", "title": "", "snippet": "Snippet C.", "text": "Text the provider gave C."},
]


def _config(
    *,
    search: dict[str, Any] | None = None,
    fetch: dict[str, Any] | None = None,
    **settings: Any,
) -> GatewayConfig:
    """A deployment whose in-loop default is a ``fake`` search with these options, enriched by ``fetch`` when given."""
    search_entry: dict[str, Any] = {"provider": "fake", "options": {"hits": _HITS, **(search or {})}}
    fetch_tools = {_FETCH: {"provider": "fake", "options": fetch}} if fetch is not None else {}
    if fetch is not None:
        search_entry["fetch_tool"] = _FETCH
    return GatewayConfig(
        search_tools={_SEARCH: search_entry},
        fetch_tools=fetch_tools,
        web_search_default_tool=_SEARCH,
        **settings,
    )


def _search(config: GatewayConfig) -> InLoopSearch:
    search = resolve_in_loop_search(config, None)
    assert search is not None
    return search


def _backend(
    config: GatewayConfig | None = None,
    *,
    cap: WebToolCallCap | None = None,
    max_results: int = 5,
    policy: DomainPolicy | None = None,
    enrich: bool = True,
    **kwargs: Any,
) -> WebSearchBackend:
    return WebSearchBackend(
        _search(config or _config()),
        cap=cap or WebToolCallCap(10),
        max_results=max_results,
        policy=policy,
        enrich=enrich,
        **kwargs,
    )


def _allowing(*domains: str) -> DomainPolicy:
    return DomainPolicy(allowed=canonicalize_domain_rules(domains))


def _page(url: str, *, text: str | None = None, final: str | None = None) -> WebFetchResult:
    """What the built-in fetcher returns for a page."""
    return WebFetchResult(
        text=f"The page at {url}." if text is None else text,
        content_type="text/html",
        content_kind="html",
        requested_url=canonicalize_web_url(url),
        final_url=canonicalize_web_url(final or url),
        redirect_count=0,
    )


@pytest.fixture(autouse=True)
def fetcher() -> Iterator[AsyncMock]:
    """The built-in fetcher's network, stubbed for every case: each page reads as its own URL."""

    async def fetch(url: str, **_kwargs: Any) -> WebFetchResult:
        return _page(url)

    stub = AsyncMock(side_effect=fetch)
    with patch.object(WebFetchService, "fetch", new=stub):
        yield stub


async def _search_once(backend: WebSearchBackend, query: str = "otari") -> str:
    async with backend:
        return await backend.call_tool(WEB_SEARCH_TOOL_NAME, {"query": query})


# ---------------------------------------------------------------------------
# What the model reads (design H, step 6)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hits_read_as_numbered_blocks_with_their_iso_dates() -> None:
    text = await _search_once(_backend(enrich=False))

    assert text == (
        "[1] A (2026-01-15T09:30:00+00:00)\nhttps://example.com/a\nSnippet A.\n\n"
        "[2] B (2026-02-01)\nhttps://example.org/b\nSnippet B.\n\n"
        "[3] (untitled)\nhttps://example.net/c\nText the provider gave C."
    ), "a date with a time of day keeps it, a page's own text wins over its snippet"


@pytest.mark.asyncio
async def test_numbering_continues_across_a_requests_searches() -> None:
    backend = _backend(_config(search={"hits": _HITS[:2]}), enrich=False)
    async with backend:
        first = await backend.call_tool(WEB_SEARCH_TOOL_NAME, {"query": "one"})
        second = await backend.call_tool(WEB_SEARCH_TOOL_NAME, {"query": "two"})

    assert first.startswith("[1] A") and "\n\n[2] B" in first
    assert second.startswith("[3] A") and "\n\n[4] B" in second


@pytest.mark.asyncio
async def test_a_search_with_no_hits_says_so_and_is_billed() -> None:
    tally = ToolUsageTally()

    text = await _search_once(_backend(_config(search={"hits": []}), tally=tally), "nothing here")

    assert text == "No results for query: 'nothing here'"
    assert tally.meters() == {WEB_SEARCH_TOOL_NAME: {"billed": 1, "errors": 0}}


@pytest.mark.asyncio
async def test_a_long_body_is_cut_at_1500_characters_and_the_whole_result_at_50_kib() -> None:
    hits = [{"url": f"https://example.com/{index}", "title": "T" * 400, "text": "x" * 5000} for index in range(10)]
    long_titles = [{**hit, "title": "T" * 40000} for hit in hits]

    one = await _search_once(_backend(_config(search={"hits": hits[:1]}), enrich=False))
    many = await _search_once(_backend(_config(search={"hits": long_titles}), max_results=10, enrich=False))

    assert one.endswith("\n" + "x" * 1500 + "…")
    assert len(many.encode()) <= WEB_RETRIEVAL_RESULT_MAX_BYTES
    assert many.endswith("[Content truncated at the 50 KiB tool-result limit.]")


# ---------------------------------------------------------------------------
# The search itself (steps 2 to 4)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_count_is_asked_for_on_every_call_with_the_instances_options() -> None:
    config = _config(search={"cost": 0.01})
    with patch.object(FakeSearchProvider, "_search", autospec=True, side_effect=FakeSearchProvider._search) as spy:
        backend = _backend(config, max_results=2, enrich=False)
        async with backend:
            await backend.call_tool(WEB_SEARCH_TOOL_NAME, {"query": "one"})
            await backend.call_tool(WEB_SEARCH_TOOL_NAME, {"query": "two"})

    assert [call.kwargs["max_results"] for call in spy.call_args_list] == [2, 2]
    assert spy.call_args_list[0].kwargs["options"] == {"hits": _HITS, "cost": 0.01}


@pytest.mark.parametrize(
    ("entry", "options", "expected"),
    [
        pytest.param({}, {}, 5, id="the-deployments-default"),
        pytest.param({"max_results": 30}, {}, 30, id="no-ceiling-of-20"),
        pytest.param({"max_results": 8}, {"max_results": 3}, 3, id="the-instance-lowers-it"),
        pytest.param({}, {"max_results": 9}, 5, id="the-instance-never-raises-it"),
        pytest.param({"max_results": True}, {}, 5, id="a-boolean-is-no-count"),
    ],
)
def test_the_count_is_the_declarations_lowered_by_the_resolved_options(
    entry: dict[str, Any], options: dict[str, Any], expected: int
) -> None:
    config = _config(search=options)

    (backend,) = build_web_search_backends(
        config,
        _search(config),
        search_tool_entry=entry,
        fetch_declared=False,
        fetch_policy=DomainPolicy(),
        cap=WebToolCallCap(10),
        purpose_hint=None,
        web_provider_client=None,
        tally=None,
        trace=None,
    )

    assert isinstance(backend, WebSearchBackend)
    assert backend._max_results == expected


@pytest.mark.asyncio
async def test_hits_off_the_searchs_domain_rules_are_dropped() -> None:
    backend = _backend(policy=_allowing("example.org", "example.net"), enrich=False)

    text = await _search_once(backend)

    assert text.startswith("[1] B (2026-02-01)\nhttps://example.org/b\nSnippet B.\n\n[2] (untitled)")
    assert "example.com" not in text


@pytest.mark.parametrize(
    ("options", "text", "failure"),
    [
        pytest.param(
            {"error": "rate_limited", "error_status": 429},
            "[tool error] fake search failed (rate_limited, HTTP 429)",
            WebToolFailure(provider="fake", status=429, tag="rate_limited", retryable=True),
            id="raised-with-a-status",
        ),
        pytest.param(
            {"error": "access_denied"},
            "[tool error] fake search failed (access_denied)",
            WebToolFailure(provider="fake", status=None, tag="access_denied", retryable=False),
            id="raised-without-one",
        ),
        pytest.param(
            {"in_body_error": "daily_limit_reached", "error_status": 503},
            "[tool error] fake search failed (daily_limit_reached, HTTP 503)",
            WebToolFailure(provider="fake", status=503, tag="daily_limit_reached", retryable=True),
            id="inside-a-successful-answer",
        ),
        pytest.param(
            {"leak_query": True},
            "[tool error] fake search failed (internal_error)",
            WebToolFailure(provider="fake", status=None, tag="internal_error", retryable=False),
            id="any-other-exception",
        ),
    ],
)
@pytest.mark.asyncio
async def test_a_provider_failure_is_returned_with_its_tag_and_never_billed(
    options: dict[str, Any], text: str, failure: WebToolFailure
) -> None:
    tally = ToolUsageTally()
    backend = _backend(_config(search=options), tally=tally)

    assert await _search_once(backend, "a private question") == text
    assert backend.take_last_failure() == failure
    assert tally.meters() == {WEB_SEARCH_TOOL_NAME: {"billed": 0, "errors": 1}}
    assert tally.label_calls() == []


@pytest.mark.parametrize("arguments", [{}, {"query": "  "}, {"query": 7}], ids=["none", "blank", "not-a-string"])
@pytest.mark.asyncio
async def test_a_call_without_a_query_is_refused_before_the_provider(arguments: dict[str, Any]) -> None:
    backend = _backend(_config(search={"error": "should_not_run"}))
    async with backend:
        text = await backend.call_tool(WEB_SEARCH_TOOL_NAME, arguments)

    assert text == "[tool error] empty query"
    assert backend.take_last_failure() == WebToolFailure(provider=None, status=None, tag="empty_query", retryable=False)


@pytest.mark.asyncio
async def test_the_hits_and_the_failure_are_kept_for_one_rendering_each() -> None:
    backend = _backend(enrich=False)
    async with backend:
        await backend.call_tool(WEB_SEARCH_TOOL_NAME, {"query": "otari"})
        hits = backend.take_last_hits()
        assert [hit.url for hit in hits] == [hit["url"] for hit in _HITS]
        assert backend.take_last_hits() == []
        await backend.call_tool(WEB_SEARCH_TOOL_NAME, {})
        assert backend.take_last_hits() == []
        assert backend.take_last_failure() is not None
        assert backend.take_last_failure() is None


@pytest.mark.asyncio
async def test_the_backend_is_the_search_tool_alone_with_its_hint() -> None:
    backend = _backend()

    assert [tool["function"]["name"] for tool in backend.openai_tools] == [WEB_SEARCH_TOOL_NAME]
    assert backend.owns_tool(WEB_SEARCH_TOOL_NAME) and not backend.owns_tool(WEB_FETCH_TOOL_NAME)
    assert backend.purpose_hints() == [(WEB_SEARCH_TOOL_NAME, WEB_SEARCH_PURPOSE_HINT)]
    with pytest.raises(KeyError):
        await backend.call_tool(WEB_FETCH_TOOL_NAME, {"url": "https://example.com/"})
    with pytest.raises(RuntimeError, match="not entered"):
        await backend.call_tool(WEB_SEARCH_TOOL_NAME, {"query": "otari"})


# ---------------------------------------------------------------------------
# The shared cap (step 1)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_search_past_the_cap_is_refused_and_counted_as_an_error() -> None:
    tally = ToolUsageTally()
    backend = _backend(cap=WebToolCallCap(1), tally=tally, enrich=False)
    async with backend:
        await backend.call_tool(WEB_SEARCH_TOOL_NAME, {"query": "one"})
        refused = await backend.call_tool(WEB_SEARCH_TOOL_NAME, {"query": "two"})

    assert refused == "[tool error] web search call limit reached for this request"
    assert backend.take_last_failure() == WebToolFailure(provider=None, status=None, tag="call_limit", retryable=False)
    assert tally.meters() == {WEB_SEARCH_TOOL_NAME: {"billed": 1, "errors": 1}}


def _search_and_fetch(
    config: GatewayConfig, cap: WebToolCallCap, **kwargs: Any
) -> tuple[WebSearchBackend, WebFetchBackend]:
    search, fetch = build_web_search_backends(
        config,
        _search(config),
        search_tool_entry={},
        fetch_declared=True,
        fetch_policy=DomainPolicy(),
        cap=cap,
        purpose_hint=None,
        web_provider_client=None,
        tally=kwargs.get("tally"),
        trace=None,
    )
    assert isinstance(search, WebSearchBackend) and isinstance(fetch, WebFetchBackend)
    return search, fetch


@pytest.mark.asyncio
async def test_search_and_fetch_calls_share_the_cap_and_an_invalid_fetch_takes_a_slot() -> None:
    config = _config(web_search_extract=False)
    cap = WebToolCallCap(3)
    search, fetch = _search_and_fetch(config, cap)
    async with search, fetch:
        assert not (await fetch.call_tool(WEB_FETCH_TOOL_NAME, {"url": 7})).endswith("limit reached for this request")
        assert not (await search.call_tool(WEB_SEARCH_TOOL_NAME, {"query": "one"})).startswith("[tool error]")
        assert not (await fetch.call_tool(WEB_FETCH_TOOL_NAME, {"url": "https://example.com/"})).startswith(
            "[tool error]"
        )
        refused_search = await search.call_tool(WEB_SEARCH_TOOL_NAME, {"query": "two"})
        refused_fetch = await fetch.call_tool(WEB_FETCH_TOOL_NAME, {"url": "https://example.com/"})

    assert refused_search == "[tool error] web search call limit reached for this request"
    assert refused_fetch == "[tool error] web fetch call limit reached for this request"
    assert fetch.take_last_failure() == WebToolFailure(provider=None, status=None, tag="call_limit", retryable=False)
    assert cap.used == 3


@pytest.mark.asyncio
async def test_the_cap_holds_across_backends_rebuilt_for_each_routing_attempt() -> None:
    """A routing plan opens the backends again for each attempt, on the request's one cap."""
    config = _config(web_search_extract=False)
    cap = WebToolCallCap(2)
    texts = [await _search_once(_backend(config, cap=cap)) for _attempt in range(3)]

    assert [text.startswith("[tool error] web search call limit") for text in texts] == [False, False, True]


# ---------------------------------------------------------------------------
# Enrichment (step 5)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_built_in_fetcher_enriches_a_hit_without_text_in_its_snippets_place(fetcher: AsyncMock) -> None:
    text = await _search_once(_backend(policy=_allowing("example.com", "example.org", "example.net")))

    assert "https://example.com/a\nThe page at https://example.com/a." in text
    assert "https://example.org/b\nThe page at https://example.org/b." in text
    assert "https://example.net/c\nText the provider gave C." in text, "a hit with text of its own is not fetched"
    assert sorted(call.args[0] for call in fetcher.call_args_list) == ["https://example.com/a", "https://example.org/b"]


@pytest.mark.parametrize("allowed", [False, True])
@pytest.mark.asyncio
async def test_the_built_in_fetcher_gets_the_searchs_rules_and_private_hosts_only_when_allowed(
    fetcher: AsyncMock, allowed: bool
) -> None:
    policy = _allowing("example.com")
    await _search_once(_backend(policy=policy, allow_private=allowed))

    assert fetcher.call_args.kwargs == {"policy": policy, "allow_private": allowed}


@pytest.mark.asyncio
async def test_a_page_the_built_in_fetcher_refuses_leaves_the_hit_its_snippet(fetcher: AsyncMock) -> None:
    """A redirect off the search's allowed domains is refused by the fetcher, which checks every hop."""
    from gateway.services.web_retrieval_network import RedirectValidationError

    fetcher.side_effect = RedirectValidationError("redirect off the allowed domains")

    text = await _search_once(_backend(policy=_allowing("example.com"), max_results=1))

    assert text == "[1] A (2026-01-15T09:30:00+00:00)\nhttps://example.com/a\nSnippet A."


@pytest.mark.parametrize("trusted", [False, True])
@pytest.mark.asyncio
async def test_the_built_in_fetchers_client_is_pinned_unless_the_proxy_is_trusted(trusted: bool) -> None:
    backend = _backend(trust_env_proxy=trusted)
    async with backend:
        client = backend._retrieval_client
        assert client is not None
        transport = client._transport
    assert isinstance(transport, TrustedProxyAsyncHTTPTransport if trusted else PinnedAsyncHTTPTransport)


@pytest.mark.asyncio
async def test_no_retrieval_client_is_opened_without_built_in_enrichment() -> None:
    for backend in (_backend(enrich=False), _backend(_config(fetch={}))):
        async with backend:
            assert backend._retrieval_client is None


@pytest.mark.asyncio
async def test_a_provider_instances_page_follows_the_hits_snippet() -> None:
    """A provider may answer a site's error page as the page's text, so the snippet is never lost."""
    text = await _search_once(_backend(_config(fetch={"text": "Not Found"}), max_results=1))

    assert text == "[1] A (2026-01-15T09:30:00+00:00)\nhttps://example.com/a\nSnippet A.\n\nNot Found"


@pytest.mark.asyncio
async def test_a_page_a_provider_cannot_fetch_leaves_the_hit_its_snippet() -> None:
    text = await _search_once(_backend(_config(fetch={"error": "http_error", "error_status": 500}), max_results=1))

    assert text.endswith("https://example.com/a\nSnippet A.")


@pytest.mark.parametrize(
    "url", ["http://127.0.0.1/a", "http://127.1/a", "http://localhost/a", "http://[::1]/a"], ids=str
)
@pytest.mark.asyncio
async def test_a_provider_instance_is_never_asked_for_an_address_that_is_not_public(url: str) -> None:
    config = _config(search={"hits": [{"url": url, "title": "Local", "snippet": "Here."}]}, fetch={"cost": 0.002})
    tally = ToolUsageTally()
    with patch.object(FakeFetchProvider, "_fetch", autospec=True, side_effect=FakeFetchProvider._fetch) as spy:
        text = await _search_once(_backend(config, tally=tally))

    assert spy.call_count == 0
    assert text.endswith("\nHere.")
    assert tally.label_calls()[0].enrichment_cost == Decimal(0)


@pytest.mark.asyncio
async def test_a_page_a_provider_found_off_the_rules_is_refused_and_not_counted() -> None:
    config = _config(fetch={"final_url": "https://elsewhere.example/landing", "cost": 0.002})
    tally = ToolUsageTally()

    text = await _search_once(_backend(config, max_results=1, policy=_allowing("example.com"), tally=tally))

    assert text.endswith("https://example.com/a\nSnippet A.")
    assert tally.label_calls()[0].enrichment_cost == Decimal(0)


@pytest.mark.parametrize(("own", "asked"), [(None, 1501), (300, 300), (9000, 1501)])
@pytest.mark.asyncio
async def test_enrichment_asks_for_no_more_than_a_hit_shows(own: int | None, asked: int) -> None:
    fetch_options = {} if own is None else {"max_chars": own}
    with patch.object(FakeFetchProvider, "_fetch", autospec=True, side_effect=FakeFetchProvider._fetch) as spy:
        await _search_once(_backend(_config(fetch=fetch_options), max_results=1))

    assert spy.call_args.kwargs["max_chars"] == asked


@pytest.mark.asyncio
async def test_a_search_enriches_its_first_results_up_to_the_bound_and_shows_the_rest_by_their_snippets(
    fetcher: AsyncMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bound is 20; the fake provider answers at most 10 results, so the case lowers it to 3."""
    import gateway.services.tools._web_search._backend as backend_module

    assert backend_module._ENRICHMENT_MAX_HITS == 20
    monkeypatch.setattr(backend_module, "_ENRICHMENT_MAX_HITS", 3)
    hits = [{"url": f"https://example.com/{index}", "snippet": f"Snippet {index}."} for index in range(5)]

    text = await _search_once(_backend(_config(search={"hits": hits})))

    assert fetcher.await_count == 3
    assert "https://example.com/2\nThe page at https://example.com/2." in text
    assert "https://example.com/3\nSnippet 3." in text


@pytest.mark.asyncio
async def test_enrichment_takes_no_slot_and_records_no_fetch() -> None:
    cap, tally = WebToolCallCap(1), ToolUsageTally()

    text = await _search_once(_backend(_config(fetch={"text": "Page."}), cap=cap, tally=tally))

    assert text.count("\n\nPage.") == 2
    assert cap.used == 1
    assert set(tally.meters()) == {WEB_SEARCH_TOOL_NAME}


@pytest.mark.asyncio
async def test_an_enrichment_instance_whose_key_cannot_be_read_leaves_the_snippets_and_the_search_billed() -> None:
    """The fetcher is built inside the fetch's own failure handling, so a missing key costs only the enrichment."""
    search = _search(_config(search={"cost": 0.005}, fetch={}))
    unreadable = replace(search, enrichment=replace(search.enrichment, provider="tabstack", api_key=""))
    cap, tally = WebToolCallCap(10), ToolUsageTally()
    backend = WebSearchBackend(unreadable, cap=cap, max_results=1, tally=tally)

    text = await _search_once(backend)

    assert text == "[1] A (2026-01-15T09:30:00+00:00)\nhttps://example.com/a\nSnippet A."
    assert tally.meters() == {WEB_SEARCH_TOOL_NAME: {"billed": 1, "errors": 0}}
    assert tally.label_calls()[0].reported_cost == Decimal("0.005")


@pytest.mark.asyncio
async def test_a_hit_whose_page_text_is_blank_is_enriched(fetcher: AsyncMock) -> None:
    config = _config(search={"hits": [{"url": "https://example.com/a", "snippet": "Snippet A.", "text": " \n "}]})

    text = await _search_once(_backend(config))

    assert text.endswith("https://example.com/a\nThe page at https://example.com/a.")


@pytest.mark.asyncio
async def test_a_failure_the_backend_does_not_know_is_counted_and_traced_before_it_is_raised() -> None:
    tally = ToolUsageTally()
    trace = RequestTrace(
        request_id="req-1",
        workspace_id=uuid.uuid4(),
        user_id="alice",
        api_key_id="key-1",
        endpoint="/v1/chat/completions",
        started_at=datetime.now(UTC),
        max_spans=10,
    )
    backend = _backend(tally=tally, trace=trace)
    with patch.object(WebSearchBackend, "_enriched", side_effect=RuntimeError("unexpected")):
        async with backend:
            with pytest.raises(RuntimeError):
                await backend.call_tool(WEB_SEARCH_TOOL_NAME, {"query": "otari"})

    assert tally.meters() == {WEB_SEARCH_TOOL_NAME: {"billed": 0, "errors": 1}}
    assert [span.outcome for span in trace.spans if span.kind == "tool"] == ["error"]


# ---------------------------------------------------------------------------
# What is recorded (step 7)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_search_and_its_enrichment_are_recorded_under_the_instance_with_their_costs_apart() -> None:
    tally = ToolUsageTally()
    priced = _config(search={"cost": 0.005}, fetch={"cost": 0.002})
    unpriced = _config(fetch={})

    await _search_once(_backend(priced, tally=tally))
    await _search_once(_backend(unpriced, tally=tally))

    assert tally.label_calls() == [
        LabelCalls(
            tool=WEB_SEARCH_TOOL_NAME,
            label=_LABEL,
            rate_keys=(_LABEL, "otari:web_search"),
            calls=2,
            unreported_calls=1,
            reported_cost=Decimal("0.005"),
            enrichment_cost=Decimal("0.004"),
        )
    ], "two hits enriched at 0.002; an enrichment fetcher that reports nothing adds nothing"


@pytest.mark.asyncio
async def test_each_search_is_one_span_on_the_requests_trace() -> None:
    trace = RequestTrace(
        request_id="req-1",
        workspace_id=uuid.uuid4(),
        user_id="alice",
        api_key_id="key-1",
        endpoint="/v1/chat/completions",
        started_at=datetime.now(UTC),
        max_spans=10,
    )
    backend = _backend(_config(search={"error": "boom"}), trace=trace)
    async with backend:
        await backend.call_tool(WEB_SEARCH_TOOL_NAME, {"query": "otari"})

    assert [(span.tool_name, span.tool_type, span.outcome) for span in trace.spans if span.kind == "tool"] == [
        (WEB_SEARCH_TOOL_NAME, "otari_web_search", "error")
    ]


@pytest.mark.asyncio
async def test_each_search_emits_an_otel_span_with_counts_and_no_query(monkeypatch: pytest.MonkeyPatch) -> None:
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    import gateway.services.tools._web_search._backend as backend_module

    exporter, provider = InMemorySpanExporter(), TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(backend_module, "_tracer", provider.get_tracer(backend_module.__name__))

    await _search_once(_backend(enrich=False), "sentinel-query-31c")

    (span,) = exporter.get_finished_spans()
    assert span.attributes is not None
    assert dict(span.attributes) == {
        "tool.name": WEB_SEARCH_TOOL_NAME,
        "tool.type": "otari_web_search",
        "web_search.provider": "fake",
        "web_search.result_count": 3,
    }


# ---------------------------------------------------------------------------
# Which backends a request gets
# ---------------------------------------------------------------------------


def test_the_in_loop_search_carries_the_fetch_that_enriches_it() -> None:
    by_instance = _search(_config(fetch={}))
    by_default = _search(GatewayConfig(search_tools={_SEARCH: {"provider": "fake"}}))

    assert (by_instance.search.label, by_instance.enrichment.name) == (_LABEL, _FETCH)
    assert by_default.enrichment.name == "builtin_fetch"


def test_a_request_option_the_provider_does_not_take_is_refused_and_a_workspace_one_dropped() -> None:
    config = _config()
    workspace = ResolvedWebSearchConfig(
        enabled=True,
        max_results=None,
        purpose_hint=None,
        allowed_domains=None,
        blocked_domains=None,
        provider_options={"no_such_option": 1},
        authorized_tools=None,
    )

    with pytest.raises(WebSearchOptionRefusedError, match="no_such_option"):
        resolve_in_loop_search(config, None, request_options={"no_such_option": 1})
    search = resolve_in_loop_search(config, workspace, request_options={"cost": 0.01})
    assert search is not None
    assert dict(search.search.options) == {"hits": _HITS, "cost": 0.01}


def test_the_legacy_settings_resolve_to_no_in_loop_search_on_the_libraries() -> None:
    assert resolve_in_loop_search(GatewayConfig(web_search_url="http://searxng:8080"), None) is None
    assert resolve_in_loop_search(GatewayConfig(), None) is None


def _context(config: GatewayConfig, *, search: bool, fetch: bool, entry: dict[str, Any] | None = None) -> ToolContext:
    search_entry = (entry or {"type": "otari_web_search"}) if search else None
    in_loop = resolve_in_loop_search(config, None) if search else None
    return ToolContext(
        config=config,
        mcp_server_configs=None,
        use_sandbox=False,
        sandbox_tool_entry=None,
        code_execution_port=None,
        use_web_search=search,
        web_search_tool_entry=search_entry,
        web_search_url=None,
        web_search_auth_token=None,
        web_search=in_loop,
        use_web_fetch=fetch,
        web_fetch_tool_entry={"type": "otari_web_fetch"} if fetch else None,
        remaining_user_tools=None,
        max_tool_iterations=10,
        tools_header=None,
    )


def test_a_request_on_the_libraries_gets_its_search_and_fetch_as_two_backends_with_one_cap() -> None:
    config = _config(web_search_max_calls=4)

    search_only = _context(config, search=True, fetch=False).build_web_backends()
    context = _context(config, search=True, fetch=True)
    both = context.build_web_backends()
    fetch_only = _context(config, search=False, fetch=True).build_web_backends()

    assert [type(backend) for backend in search_only] == [WebSearchBackend]
    assert [type(backend) for backend in both] == [WebSearchBackend, WebFetchBackend]
    assert [type(backend) for backend in fetch_only] == [WebRetrievalBackend], "a fetch alone keeps the old backend"
    search, fetch = both
    assert isinstance(search, WebSearchBackend) and isinstance(fetch, WebFetchBackend)
    assert search._cap is context.web_call_cap is fetch._cap
    assert [tool["function"]["name"] for backend in both for tool in backend.openai_tools] == [
        WEB_SEARCH_TOOL_NAME,
        WEB_FETCH_TOOL_NAME,
    ]
    assert [name for backend in both for name, _hint in backend.purpose_hints()] == [
        WEB_SEARCH_TOOL_NAME,
        WEB_FETCH_TOOL_NAME,
    ]


def test_the_requests_cap_is_the_deployments_call_limit() -> None:
    assert _context(_config(web_search_max_calls=4), search=True, fetch=False).web_call_cap.claim()
    context = _context(_config(web_search_max_calls=1), search=True, fetch=False)

    assert context.web_call_cap.claim() and not context.web_call_cap.claim()


@pytest.mark.parametrize(
    ("entry_hint", "setting", "environment", "expected"),
    [
        pytest.param(
            "From the declaration.", "From the setting.", "From the env.", "From the declaration.", id="entry"
        ),
        pytest.param(None, "From the setting.", "From the env.", "From the setting.", id="setting"),
        pytest.param(None, None, "From the env.", "From the env.", id="environment"),
        pytest.param(None, None, None, WEB_SEARCH_PURPOSE_HINT, id="default"),
    ],
)
def test_the_search_hint_comes_from_the_declaration_the_setting_the_environment_then_the_default(
    monkeypatch: pytest.MonkeyPatch, entry_hint: str | None, setting: str | None, environment: str | None, expected: str
) -> None:
    if environment is None:
        monkeypatch.delenv("OTARI_WEB_SEARCH_PURPOSE_HINT", raising=False)
    else:
        monkeypatch.setenv("OTARI_WEB_SEARCH_PURPOSE_HINT", environment)
    entry = {"type": "otari_web_search", **({"purpose_hint": entry_hint} if entry_hint else {})}
    context = _context(_config(web_search_purpose_hint=setting), search=True, fetch=False, entry=entry)

    (backend,) = context.build_web_backends()

    assert backend.purpose_hints() == [(WEB_SEARCH_TOOL_NAME, expected)]
