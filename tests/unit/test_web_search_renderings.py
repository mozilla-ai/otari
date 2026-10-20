"""Web search's native server-tool renderings, one per wire dialect."""

from __future__ import annotations

import json
from contextlib import AsyncExitStack
from dataclasses import replace
from typing import Any

import pytest

from gateway.core.config import GatewayConfig
from gateway.services.tool_usage import is_tool_error
from gateway.services.tools import (
    WEB_RETRIEVAL_RESULT_MAX_BYTES,
    WEB_SEARCH_CALL_ID_PREFIX,
    WEB_SEARCH_TOOL_NAME,
    CombinedToolBackend,
    Dialect,
    NativeCall,
    WebToolCallCap,
    build_web_search_backends,
    native_rendering,
    owning_backend,
    resolve_in_loop_search,
)
from gateway.services.web_retrieval_policy import DomainPolicy


class _SearchBackendLike:
    """The gateway's own retrieval backend, as a rendering reads it."""

    def __init__(self, results: list[dict[str, Any]] | None = None) -> None:
        self._results = results or []

    def take_last_results(self) -> list[dict[str, Any]]:
        results, self._results = self._results, []
        return results


class _McpPoolLike:
    """An MCP pool, which keeps no structured search results."""


def _rendering(dialect: Dialect) -> Any:
    rendering = native_rendering(WEB_SEARCH_TOOL_NAME, dialect)
    assert rendering is not None
    return rendering


def _call(query: str = "what is the latest python?", *, failed: bool = False) -> NativeCall:
    return NativeCall(WEB_SEARCH_TOOL_NAME, "toolu_1", {"query": query}, failed=failed)


@pytest.mark.parametrize(
    ("type_value", "expected"),
    [
        ("web_search_20250305", True),
        ("web_search_preview", True),
        ("web_search", False),
        ("otari_web_search", False),
    ],
)
def test_only_a_dated_keyword_asks_for_the_anthropic_pair(type_value: str, expected: bool) -> None:
    """The bare and canonical shapes imply no response shape, so those callers keep plain text."""
    assert _rendering(Dialect.MESSAGES).declared({"type": type_value}) is expected


@pytest.mark.parametrize("tool_entry", [None, {}])
def test_a_missing_declaration_asks_for_no_anthropic_pair(tool_entry: dict[str, Any] | None) -> None:
    assert _rendering(Dialect.MESSAGES).declared(tool_entry) is False


@pytest.mark.parametrize("tool_entry", [None, {}, {"type": "otari_web_search"}, {"type": "web_search_preview"}])
def test_every_caller_is_told_a_search_ran_in_the_responses_vocabulary(tool_entry: dict[str, Any] | None) -> None:
    """The item forges no provider-signed content, so no caller has to opt in."""
    assert _rendering(Dialect.RESPONSES).declared(tool_entry) is True


def test_the_anthropic_pair_cites_the_hits_the_backend_kept() -> None:
    backend = _SearchBackendLike([{"url": "https://example.com/a", "title": "A", "published_date": "2026-01-02"}])

    use, result = _rendering(Dialect.MESSAGES).ran(_call(), backend)

    assert use.name == WEB_SEARCH_TOOL_NAME
    assert use.input == {"query": "what is the latest python?"}
    assert result.tool_use_id == use.id
    assert [citation.url for citation in result.content] == ["https://example.com/a"]


def test_a_failed_search_has_nothing_to_cite() -> None:
    backend = _SearchBackendLike([{"url": "https://example.com/a", "title": "A"}])

    assert _rendering(Dialect.MESSAGES).ran(_call(failed=True), backend) == []


def test_a_pool_that_kept_no_results_is_not_the_gateway_search_backend() -> None:
    """An MCP server may expose a tool of the same name; its call is not a gateway search."""
    assert _rendering(Dialect.MESSAGES).ran(_call(), _McpPoolLike()) == []


def test_a_refused_search_is_reported_in_the_anthropic_error_shape() -> None:
    use, result = _rendering(Dialect.MESSAGES).refused(_call())

    assert use.input == {"query": "what is the latest python?"}
    assert result.tool_use_id == use.id
    assert result.content.error_code == "max_uses_exceeded"


def test_the_responses_item_carries_the_query_and_an_id_the_gateway_minted() -> None:
    """The prefix is what lets an echoed item be told apart from a search the provider ran."""
    (item,) = _rendering(Dialect.RESPONSES).ran(_call(), _SearchBackendLike())

    assert item.type == "web_search_call"
    assert item.id.startswith(WEB_SEARCH_CALL_ID_PREFIX)
    assert item.action.query == "what is the latest python?"
    assert item.status == "completed"


def test_a_refused_search_is_invisible_in_the_responses_vocabulary() -> None:
    """No search ran, so there is nothing to announce."""
    assert _rendering(Dialect.RESPONSES).refused(_call()) == []


@pytest.mark.parametrize("tool_entry", [{"type": "otari_web_search"}, {"type": "web_search_20250305"}, None])
def test_chat_reports_every_declaration_of_the_search(tool_entry: dict[str, Any] | None) -> None:
    assert _rendering(Dialect.CHAT).declared(tool_entry) is True


def test_a_chat_search_is_announced_with_its_query_and_ended_with_its_sources_under_the_same_id() -> None:
    call = _call()
    backend = _SearchBackendLike(
        [
            {
                "url": "https://python.org",
                "title": "Python",
                "content": "page text only the model reads",
                "published_date": "2026-10-01",
            },
            {"url": "https://docs.python.org", "title": "", "content": "more page text"},
        ]
    )

    [started] = _rendering(Dialect.CHAT).started(call)
    [ended] = _rendering(Dialect.CHAT).ran(call, backend)

    assert started == {
        "id": started["id"],
        "type": "web_search",
        "status": "in_progress",
        "query": "what is the latest python?",
    }
    assert started["id"].startswith(WEB_SEARCH_CALL_ID_PREFIX)
    assert ended == {
        **started,
        "status": "completed",
        "sources": [
            {"url": "https://python.org", "title": "Python", "published_date": "2026-10-01"},
            {"url": "https://docs.python.org", "title": "https://docs.python.org", "published_date": None},
        ],
    }
    assert "page text" not in str(ended)


def test_a_chat_search_with_no_hits_is_completed_with_no_sources() -> None:
    [ended] = _rendering(Dialect.CHAT).ran(_call(), _SearchBackendLike([{"url": "", "title": "no url"}]))

    assert ended["status"] == "completed"
    assert ended["sources"] == []


def test_a_chat_source_bounds_its_title_and_date() -> None:
    hit = {"url": "https://example.com", "title": "t" * 2000, "published_date": "Mon,\n  " + "1" * 500}

    [ended] = _rendering(Dialect.CHAT).ran(_call(), _SearchBackendLike([hit]))

    [source] = ended["sources"]
    assert len(source["title"].encode()) <= 512
    assert source["published_date"].startswith("Mon, 111")
    assert len(source["published_date"]) == 128


def test_a_failed_chat_search_on_the_old_backend_has_no_error() -> None:
    [ended] = _rendering(Dialect.CHAT).ran(_call(failed=True), _SearchBackendLike([{"url": "https://stale.example"}]))

    assert ended["status"] == "failed"
    assert "error" not in ended
    assert "sources" not in ended


def test_a_refused_chat_search_fails_with_the_call_limit_tag() -> None:
    [ended] = _rendering(Dialect.CHAT).refused(_call())

    assert ended["status"] == "failed"
    assert ended["query"] == "what is the latest python?"
    assert ended["error"] == {"provider": None, "status": None, "tag": "call_limit", "retryable": False}


@pytest.mark.parametrize("dialect", [Dialect.MESSAGES, Dialect.RESPONSES])
def test_the_other_dialects_announce_nothing_when_a_search_starts(dialect: Dialect) -> None:
    assert _rendering(dialect).started(_call()) == []


def test_chat_sources_stay_within_the_model_facing_results_byte_limit() -> None:
    """Titles and dates are capped, so the URL is what the limit bounds; a source that would cross it is left out."""
    long_url = "https://example.com/" + "a" * (WEB_RETRIEVAL_RESULT_MAX_BYTES // 3)
    hits = [{"url": f"{long_url}{index}", "title": "t"} for index in range(5)] + [{"url": "https://short.example"}]

    [ended] = _rendering(Dialect.CHAT).ran(_call(), _SearchBackendLike(hits))

    assert len(json.dumps(ended["sources"]).encode()) <= WEB_RETRIEVAL_RESULT_MAX_BYTES
    assert len(ended["sources"]) == 3
    assert ended["sources"][-1]["url"] == "https://short.example"


# ---------------------------------------------------------------------------
# The search backend on any-search, as the loop hands it to a rendering: the
# member of the request's pool that ran the call
# ---------------------------------------------------------------------------

_HITS = [
    {"url": "https://example.com/a", "title": "A", "snippet": "Snippet A.", "published": "2026-01-15T09:30:00Z"},
    {"url": "https://example.org/b", "title": "", "snippet": "Snippet B.", "published": "2026-02-01"},
]


async def _ran_on_the_libraries(
    dialect: Dialect, options: dict[str, Any] | None = None, *, cap: int = 10, calls: int = 1
) -> list[Any]:
    """The rendering of the last of ``calls`` searches on a ``fake`` instance, in a pool beside the fetch backend."""
    config = GatewayConfig(
        search_tools={"fake-search": {"provider": "fake", "options": {"hits": _HITS, **(options or {})}}},
        web_search_default_tool="fake-search",
        web_search_extract=False,
    )
    search = resolve_in_loop_search(config, None)
    assert search is not None
    backends = build_web_search_backends(
        config,
        search,
        search_tool_entry={},
        fetch_declared=True,
        fetch_policy=DomainPolicy(),
        cap=WebToolCallCap(cap),
        purpose_hint=None,
        web_provider_client=None,
        tally=None,
        trace=None,
    )
    pool = CombinedToolBackend(backends)
    call = _call()
    async with AsyncExitStack() as stack:
        for backend in backends:
            await stack.enter_async_context(backend)
        owner = owning_backend(pool, WEB_SEARCH_TOOL_NAME)
        for _ in range(calls):
            text = await owner.call_tool(WEB_SEARCH_TOOL_NAME, dict(call.arguments))
        ended: list[Any] = _rendering(dialect).ran(replace(call, failed=is_tool_error(text)), owner)
        return ended


@pytest.mark.asyncio
async def test_the_anthropic_pair_cites_a_library_searchs_hits_with_iso_dates() -> None:
    use, result = await _ran_on_the_libraries(Dialect.MESSAGES)

    assert use.input == {"query": "what is the latest python?"}
    assert [(citation.url, citation.title, citation.page_age) for citation in result.content] == [
        ("https://example.com/a", "A", "2026-01-15T09:30:00+00:00"),
        ("https://example.org/b", "https://example.org/b", "2026-02-01"),
    ]


@pytest.mark.asyncio
async def test_a_failed_library_search_has_nothing_to_cite() -> None:
    assert await _ran_on_the_libraries(Dialect.MESSAGES, {"error": "timeout"}) == []


@pytest.mark.asyncio
async def test_a_chat_search_on_the_libraries_ends_with_its_sources_and_iso_dates() -> None:
    [ended] = await _ran_on_the_libraries(Dialect.CHAT)

    assert ended["status"] == "completed"
    assert ended["sources"] == [
        {"url": "https://example.com/a", "title": "A", "published_date": "2026-01-15T09:30:00+00:00"},
        {"url": "https://example.org/b", "title": "https://example.org/b", "published_date": "2026-02-01"},
    ]
    assert "Snippet" not in str(ended)


@pytest.mark.asyncio
async def test_a_failed_chat_search_on_the_libraries_carries_the_providers_failure() -> None:
    [ended] = await _ran_on_the_libraries(Dialect.CHAT, {"error": "upstream_error", "error_status": 503})

    assert ended["status"] == "failed"
    assert ended["error"] == {"provider": "fake", "status": 503, "tag": "upstream_error", "retryable": True}
    assert "sources" not in ended


@pytest.mark.asyncio
async def test_a_chat_search_past_the_requests_call_limit_fails_with_the_call_limit_tag() -> None:
    [ended] = await _ran_on_the_libraries(Dialect.CHAT, cap=1, calls=2)

    assert ended["status"] == "failed"
    assert ended["error"] == {"provider": None, "status": None, "tag": "call_limit", "retryable": False}
