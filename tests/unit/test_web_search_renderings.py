"""Web search's native server-tool renderings, one per wire dialect."""

from __future__ import annotations

from typing import Any

import pytest

from gateway.services.tools import Dialect, NativeCall, native_rendering
from gateway.services.web_retrieval_backend import WEB_SEARCH_TOOL_NAME


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


def test_the_responses_item_carries_the_query_and_the_caller_s_call_id() -> None:
    (item,) = _rendering(Dialect.RESPONSES).ran(_call(), _SearchBackendLike())

    assert item.type == "web_search_call"
    assert item.id == "toolu_1"
    assert item.action.query == "what is the latest python?"
    assert item.status == "completed"


def test_a_refused_search_is_invisible_in_the_responses_vocabulary() -> None:
    """No search ran, so there is nothing to announce."""
    assert _rendering(Dialect.RESPONSES).refused(_call()) == []
