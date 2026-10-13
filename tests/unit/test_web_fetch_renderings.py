"""Web fetch's native server-tool rendering on Messages."""

from __future__ import annotations

from typing import Any

import pytest

from gateway.services.tools import (
    SERVER_TOOL_USE_ID_PREFIX,
    WEB_FETCH_TOOL_NAME,
    WEB_RETRIEVAL_RESULT_MAX_BYTES,
    Dialect,
    NativeCall,
    native_rendering,
)
from gateway.services.web_fetch_service import WebFetchResult
from gateway.services.web_retrieval_policy import canonicalize_web_url


def _page(text: str = "page text", url: str = "https://example.com/a?token=secret#frag") -> WebFetchResult:
    canonical = canonicalize_web_url(url)
    return WebFetchResult(
        text=text,
        content_type="text/html",
        content_kind="html",
        requested_url=canonical,
        final_url=canonical,
        redirect_count=0,
    )


class _FetchBackendLike:
    """The gateway's own retrieval backend, as a rendering reads it."""

    def __init__(self, page: WebFetchResult | None = None) -> None:
        self._page = page

    def take_last_fetch(self) -> WebFetchResult | None:
        page, self._page = self._page, None
        return page


class _McpPoolLike:
    """An MCP pool, which keeps no fetched page."""


def _rendering() -> Any:
    rendering = native_rendering(WEB_FETCH_TOOL_NAME, Dialect.MESSAGES)
    assert rendering is not None
    return rendering


def _call(*, failed: bool = False) -> NativeCall:
    return NativeCall(WEB_FETCH_TOOL_NAME, "toolu_1", {"url": "https://example.com/a?token=secret"}, failed=failed)


@pytest.mark.parametrize(
    ("tool_entry", "expected"),
    [
        ({"type": "web_fetch_20250910"}, True),
        ({"type": "web_fetch_preview"}, False),
        ({"type": "web_fetch_2025091"}, False),
        ({"type": "otari_web_fetch"}, False),
        ({}, False),
        (None, False),
    ],
)
def test_only_the_dated_keyword_asks_for_the_anthropic_pair(tool_entry: dict[str, Any] | None, expected: bool) -> None:
    assert _rendering().declared(tool_entry) is expected


def test_no_other_dialect_announces_a_fetch() -> None:
    assert native_rendering(WEB_FETCH_TOOL_NAME, Dialect.RESPONSES) is None
    assert native_rendering(WEB_FETCH_TOOL_NAME, Dialect.CHAT) is None


def test_the_pair_carries_the_page_the_backend_kept_without_its_query() -> None:
    use, result = _rendering().ran(_call(), _FetchBackendLike(_page()))

    assert use.type == "server_tool_use"
    assert use.name == WEB_FETCH_TOOL_NAME
    assert use.id.startswith(SERVER_TOOL_USE_ID_PREFIX)
    assert use.input == {"url": "https://example.com/a"}
    assert result.type == "web_fetch_tool_result"
    assert result.tool_use_id == use.id
    assert result.content.type == "web_fetch_result"
    assert result.content.url == "https://example.com/a"
    assert result.content.content.source.data == "page text"


def test_the_document_is_bounded_like_the_model_facing_result() -> None:
    _, result = _rendering().ran(_call(), _FetchBackendLike(_page("x" * (WEB_RETRIEVAL_RESULT_MAX_BYTES * 2))))

    assert len(result.content.content.source.data.encode()) <= WEB_RETRIEVAL_RESULT_MAX_BYTES


def test_a_failed_fetch_is_reported_in_the_error_shape() -> None:
    use, result = _rendering().ran(_call(failed=True), _FetchBackendLike())

    assert result.tool_use_id == use.id
    assert result.content.type == "web_fetch_tool_result_error"
    assert result.content.error_code == "unavailable"


def test_an_mcp_tool_named_web_fetch_is_not_announced() -> None:
    assert _rendering().ran(_call(), _McpPoolLike()) == []


def test_a_refused_fetch_reports_max_uses_exceeded() -> None:
    use, result = _rendering().refused(_call())

    assert result.tool_use_id == use.id
    assert result.content.error_code == "max_uses_exceeded"
