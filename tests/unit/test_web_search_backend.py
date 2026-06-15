"""Unit tests for `WebSearchBackend`.

Mocks the HTTP layer so the suite needs neither a SearXNG container nor
network access for trafilatura's per-URL fetches.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import httpx
import pytest

from gateway.services.web_search_backend import (
    WEB_SEARCH_TOOL_NAME,
    WebSearchBackend,
    WebSearchNotReachableError,
)


class _MockTransport(httpx.AsyncBaseTransport):
    """Tiny in-process httpx transport routing requests to a (host, path) handler dict."""

    def __init__(self, handlers: dict[tuple[str, str], httpx.Response | Exception]) -> None:
        self._handlers = handlers
        self.captured: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.captured.append(request)
        key = (request.url.host or "", request.url.path)
        handler = self._handlers.get(key)
        if handler is None:
            return httpx.Response(404, json={"error": f"no handler for {key}"})
        if isinstance(handler, Exception):
            raise handler
        return handler


def _patched_async_client(handlers: dict[tuple[str, str], Any], monkeypatch: pytest.MonkeyPatch) -> _MockTransport:
    transport = _MockTransport(handlers)
    original_init = httpx.AsyncClient.__init__

    def patched_init(self: httpx.AsyncClient, *args: Any, **kwargs: Any) -> None:
        kwargs["transport"] = transport
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", patched_init)
    return transport


SEARXNG_OK_BODY = {
    "query": "claude code",
    "number_of_results": 2,
    "results": [
        {
            "url": "https://example.com/post-a",
            "title": "Post A",
            "content": "snippet about A",
            "engine": "duckduckgo",
        },
        {
            "url": "https://example.org/post-b",
            "title": "Post B",
            "content": "snippet about B",
            "engine": "mojeek",
        },
    ],
}


@pytest.mark.asyncio
async def test_owns_only_web_search() -> None:
    backend = WebSearchBackend(base_url="http://searxng:8080")
    assert backend.owns_tool(WEB_SEARCH_TOOL_NAME)
    assert not backend.owns_tool("code_execution")
    assert not backend.owns_tool("anything_else")


@pytest.mark.asyncio
async def test_openai_tools_advertises_web_search() -> None:
    backend = WebSearchBackend(base_url="http://searxng:8080")
    tools = backend.openai_tools
    assert len(tools) == 1
    assert tools[0]["function"]["name"] == WEB_SEARCH_TOOL_NAME
    assert "query" in tools[0]["function"]["parameters"]["properties"]


@pytest.mark.asyncio
async def test_purpose_hint_is_emitted() -> None:
    backend = WebSearchBackend(base_url="http://searxng:8080")
    hints = backend.purpose_hints()
    assert len(hints) == 1
    assert hints[0][0] == WEB_SEARCH_TOOL_NAME


@pytest.mark.asyncio
async def test_call_tool_returns_formatted_results_without_extraction(monkeypatch: pytest.MonkeyPatch) -> None:
    _patched_async_client(
        {("searxng", "/search"): httpx.Response(200, json=SEARXNG_OK_BODY)},
        monkeypatch,
    )

    async with WebSearchBackend(base_url="http://searxng:8080", extract_content=False) as backend:
        result = await backend.call_tool(WEB_SEARCH_TOOL_NAME, {"query": "claude code"})

    # Falls back to engine-supplied snippets when extraction is disabled.
    assert "[1] Post A" in result
    assert "https://example.com/post-a" in result
    assert "snippet about A" in result
    assert "[2] Post B" in result


@pytest.mark.asyncio
async def test_call_tool_extracts_content_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    _patched_async_client(
        {
            ("searxng", "/search"): httpx.Response(200, json=SEARXNG_OK_BODY),
            ("example.com", "/post-a"): httpx.Response(200, text="<html><body><p>Article A body</p></body></html>"),
            ("example.org", "/post-b"): httpx.Response(200, text="<html><body><p>Article B body</p></body></html>"),
        },
        monkeypatch,
    )

    # Stub trafilatura to return deterministic markdown without invoking the
    # heavier real extractor — keeps the unit test fast and free of HTML
    # quirks. The contract under test is "extracted content wins over
    # snippet", not trafilatura's specific output.
    with patch("gateway.services.web_search_backend.trafilatura.extract") as mock_extract:
        mock_extract.side_effect = lambda html, **_: f"extracted: {html[-30:]}"
        async with WebSearchBackend(base_url="http://searxng:8080", extract_content=True) as backend:
            result = await backend.call_tool(WEB_SEARCH_TOOL_NAME, {"query": "claude code"})

    # Stub returns the last 30 chars of the fetched HTML — verify the extracted
    # content (not just the snippet) reached the formatter.
    assert "extracted:" in result
    # Original snippets should NOT be present since extracted content wins.
    assert "snippet about A" not in result
    assert "snippet about B" not in result


@pytest.mark.asyncio
async def test_extraction_failure_falls_back_to_snippet(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed per-URL fetch must not break the whole search — degrade silently to snippet."""
    _patched_async_client(
        {
            ("searxng", "/search"): httpx.Response(200, json=SEARXNG_OK_BODY),
            ("example.com", "/post-a"): httpx.ConnectError("dns failure"),
            ("example.org", "/post-b"): httpx.ConnectError("dns failure"),
        },
        monkeypatch,
    )

    async with WebSearchBackend(base_url="http://searxng:8080", extract_content=True) as backend:
        result = await backend.call_tool(WEB_SEARCH_TOOL_NAME, {"query": "claude code"})

    assert "snippet about A" in result
    assert "snippet about B" in result


@pytest.mark.asyncio
async def test_allowed_domains_filter_in_gateway(monkeypatch: pytest.MonkeyPatch) -> None:
    _patched_async_client(
        {("searxng", "/search"): httpx.Response(200, json=SEARXNG_OK_BODY)},
        monkeypatch,
    )

    async with WebSearchBackend(
        base_url="http://searxng:8080",
        extract_content=False,
        allowed_domains=("example.org",),
    ) as backend:
        result = await backend.call_tool(WEB_SEARCH_TOOL_NAME, {"query": "claude code"})

    assert "example.org" in result
    assert "example.com" not in result


@pytest.mark.asyncio
async def test_blocked_domains_filter_in_gateway(monkeypatch: pytest.MonkeyPatch) -> None:
    _patched_async_client(
        {("searxng", "/search"): httpx.Response(200, json=SEARXNG_OK_BODY)},
        monkeypatch,
    )

    async with WebSearchBackend(
        base_url="http://searxng:8080",
        extract_content=False,
        blocked_domains=("example.com",),
    ) as backend:
        result = await backend.call_tool(WEB_SEARCH_TOOL_NAME, {"query": "claude code"})

    assert "example.com" not in result
    assert "example.org" in result


@pytest.mark.asyncio
async def test_max_results_truncates(monkeypatch: pytest.MonkeyPatch) -> None:
    body = {"results": [{"url": f"https://ex.com/{i}", "title": f"T{i}", "content": f"s{i}"} for i in range(10)]}
    _patched_async_client(
        {("searxng", "/search"): httpx.Response(200, json=body)},
        monkeypatch,
    )

    async with WebSearchBackend(base_url="http://searxng:8080", extract_content=False, max_results=3) as backend:
        result = await backend.call_tool(WEB_SEARCH_TOOL_NAME, {"query": "x"})

    assert "[1] T0" in result
    assert "[3] T2" in result
    assert "[4]" not in result


@pytest.mark.asyncio
async def test_provider_options_forwarded_as_query_params(monkeypatch: pytest.MonkeyPatch) -> None:
    """Scalar provider_options become extra /search query params; bools
    serialise as lowercase strings and None/complex values are dropped."""
    transport = _patched_async_client(
        {("searxng", "/search"): httpx.Response(200, json=SEARXNG_OK_BODY)},
        monkeypatch,
    )

    async with WebSearchBackend(
        base_url="http://searxng:8080",
        extract_content=False,
        provider_options={
            "search_depth": "advanced",
            "max_results": 7,
            "include_answer": True,
            "exclude_news": False,
            "dropped_none": None,
            "dropped_list": ["a", "b"],
        },
    ) as backend:
        await backend.call_tool(WEB_SEARCH_TOOL_NAME, {"query": "x"})

    req = transport.captured[0]
    params = dict(req.url.params)
    assert params["q"] == "x"
    assert params["format"] == "json"
    assert params["search_depth"] == "advanced"
    assert params["max_results"] == "7"
    assert params["include_answer"] == "true"
    assert params["exclude_news"] == "false"
    assert "dropped_none" not in params
    assert "dropped_list" not in params


@pytest.mark.asyncio
async def test_provider_options_cannot_override_reserved_params(monkeypatch: pytest.MonkeyPatch) -> None:
    """provider_options must never override the gateway-controlled q/format/engines."""
    transport = _patched_async_client(
        {("searxng", "/search"): httpx.Response(200, json=SEARXNG_OK_BODY)},
        monkeypatch,
    )

    async with WebSearchBackend(
        base_url="http://searxng:8080",
        engines=("duckduckgo",),
        extract_content=False,
        provider_options={"q": "evil", "format": "xml", "engines": "google", "topic": "news"},
    ) as backend:
        await backend.call_tool(WEB_SEARCH_TOOL_NAME, {"query": "real query"})

    params = dict(transport.captured[0].url.params)
    assert params["q"] == "real query"
    assert params["format"] == "json"
    assert params["engines"] == "duckduckgo"
    assert params["topic"] == "news"


@pytest.mark.asyncio
async def test_backend_unreachable_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _patched_async_client(
        {("searxng", "/search"): httpx.ConnectError("connection refused")},
        monkeypatch,
    )

    async with WebSearchBackend(base_url="http://searxng:8080") as backend:
        with pytest.raises(WebSearchNotReachableError, match="web_search failed"):
            await backend.call_tool(WEB_SEARCH_TOOL_NAME, {"query": "x"})


@pytest.mark.asyncio
async def test_empty_query_returns_error_message(monkeypatch: pytest.MonkeyPatch) -> None:
    # No HTTP call should be made for an empty query — backend short-circuits.
    transport = _patched_async_client(
        {("searxng", "/search"): httpx.Response(200, json=SEARXNG_OK_BODY)},
        monkeypatch,
    )

    async with WebSearchBackend(base_url="http://searxng:8080") as backend:
        result = await backend.call_tool(WEB_SEARCH_TOOL_NAME, {"query": "   "})

    assert "[tool error]" in result
    # Should not have hit /search.
    assert not any(r.url.path == "/search" for r in transport.captured)


@pytest.mark.asyncio
async def test_extracted_content_from_backend_bypasses_local_extraction(monkeypatch: pytest.MonkeyPatch) -> None:
    """Adapter-supplied ``extracted_content`` (e.g. a Tavily-fronting service)
    must take precedence — the gateway should not refetch+extract those URLs.
    """
    body = {
        "results": [
            {
                "url": "https://example.com/a",
                "title": "A",
                "content": "snippet",
                "extracted_content": "PRE-EXTRACTED BY ADAPTER",
            }
        ]
    }
    transport = _patched_async_client(
        {("searxng", "/search"): httpx.Response(200, json=body)},
        monkeypatch,
    )

    async with WebSearchBackend(base_url="http://searxng:8080", extract_content=True) as backend:
        result = await backend.call_tool(WEB_SEARCH_TOOL_NAME, {"query": "x"})

    assert "PRE-EXTRACTED BY ADAPTER" in result
    # No follow-up fetch to the result URL.
    assert not any(r.url.host == "example.com" for r in transport.captured)


@pytest.mark.asyncio
async def test_does_not_own_unknown_tool() -> None:
    backend = WebSearchBackend(base_url="http://searxng:8080")
    with pytest.raises(KeyError):
        # call_tool requires entering as a context manager first; without
        # owning the tool, however, it short-circuits with KeyError before
        # the client check.
        await backend.call_tool("not_web_search", {"query": "x"})


# --- SSRF guard --------------------------------------------------------------


@pytest.mark.asyncio
async def test_ssrf_guard_blocks_cloud_metadata_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """A search result pointing at AWS IMDS must not be fetched."""
    body = {
        "results": [
            {"url": "http://169.254.169.254/latest/meta-data/", "title": "metadata", "content": "snippet"},
        ]
    }
    transport = _patched_async_client(
        {("searxng", "/search"): httpx.Response(200, json=body)},
        monkeypatch,
    )

    async with WebSearchBackend(base_url="http://searxng:8080", extract_content=True) as backend:
        result = await backend.call_tool(WEB_SEARCH_TOOL_NAME, {"query": "x"})

    # The result is still surfaced (with the snippet) but no fetch happened.
    assert "snippet" in result
    assert not any(r.url.host == "169.254.169.254" for r in transport.captured)


@pytest.mark.asyncio
async def test_ssrf_guard_blocks_rfc1918_url(monkeypatch: pytest.MonkeyPatch) -> None:
    body = {
        "results": [
            {"url": "http://10.0.0.5/internal", "title": "internal", "content": "snippet"},
        ]
    }
    transport = _patched_async_client(
        {("searxng", "/search"): httpx.Response(200, json=body)},
        monkeypatch,
    )

    async with WebSearchBackend(base_url="http://searxng:8080", extract_content=True) as backend:
        await backend.call_tool(WEB_SEARCH_TOOL_NAME, {"query": "x"})

    assert not any(r.url.host == "10.0.0.5" for r in transport.captured)


@pytest.mark.asyncio
async def test_ssrf_guard_blocks_loopback(monkeypatch: pytest.MonkeyPatch) -> None:
    body = {
        "results": [
            {"url": "http://127.0.0.1:9000/secrets", "title": "local", "content": "snippet"},
        ]
    }
    transport = _patched_async_client(
        {("searxng", "/search"): httpx.Response(200, json=body)},
        monkeypatch,
    )

    async with WebSearchBackend(base_url="http://searxng:8080", extract_content=True) as backend:
        await backend.call_tool(WEB_SEARCH_TOOL_NAME, {"query": "x"})

    assert not any(r.url.host == "127.0.0.1" for r in transport.captured)


@pytest.mark.asyncio
async def test_ssrf_guard_env_override_allows_private(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GATEWAY_WEB_SEARCH_ALLOW_PRIVATE_HOSTS", "true")
    body = {
        "results": [
            {"url": "http://10.0.0.5/page", "title": "internal", "content": "snippet"},
        ]
    }
    transport = _patched_async_client(
        {
            ("searxng", "/search"): httpx.Response(200, json=body),
            ("10.0.0.5", "/page"): httpx.Response(200, text="<html><body>hi</body></html>"),
        },
        monkeypatch,
    )

    async with WebSearchBackend(base_url="http://searxng:8080", extract_content=True) as backend:
        await backend.call_tool(WEB_SEARCH_TOOL_NAME, {"query": "x"})

    # Override flips the gate — fetch is allowed.
    assert any(r.url.host == "10.0.0.5" for r in transport.captured)


# --- Redirect-chain SSRF -----------------------------------------------------


@pytest.mark.asyncio
async def test_ssrf_guard_blocks_redirect_to_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    """A publicly-resolving result that 302-redirects to AWS IMDS must not be followed.

    The classic SSRF bypass: the initial URL passes the safety check (public
    DNS), then the response 302-redirects to ``169.254.169.254`` and the
    naive client follows. We walk redirects manually and re-validate each
    hop — the metadata IP must be rejected here too.
    """
    body = {"results": [{"url": "https://attacker.example.com/start", "title": "trap", "content": "snippet"}]}
    transport = _patched_async_client(
        {
            ("searxng", "/search"): httpx.Response(200, json=body),
            ("attacker.example.com", "/start"): httpx.Response(
                302, headers={"location": "http://169.254.169.254/latest/meta-data/iam"}
            ),
            # If the bug were still present, this would be the leaked target.
            ("169.254.169.254", "/latest/meta-data/iam"): httpx.Response(
                200, text="<html><body>SECRET-KEY</body></html>"
            ),
        },
        monkeypatch,
    )

    async with WebSearchBackend(base_url="http://searxng:8080", extract_content=True) as backend:
        result = await backend.call_tool(WEB_SEARCH_TOOL_NAME, {"query": "x"})

    # The gateway must NOT have fetched the metadata IP and must NOT surface its body.
    assert "SECRET-KEY" not in result
    assert not any(r.url.host == "169.254.169.254" for r in transport.captured)


# --- Fetch size cap ----------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_capped_truncates_huge_response(monkeypatch: pytest.MonkeyPatch) -> None:
    """A page bigger than _FETCH_MAX_BYTES must not blow memory.

    We stub the response with 10 MB of HTML and verify that the formatted
    output still surfaces (extraction may yield empty for filler bytes —
    that's fine; the contract is "don't OOM", not "always extract").
    """
    huge_html = "<html><body>" + "A" * (10 * 1024 * 1024) + "</body></html>"
    body = {"results": [{"url": "https://example.com/huge", "title": "Huge", "content": "snippet"}]}
    _patched_async_client(
        {
            ("searxng", "/search"): httpx.Response(200, json=body),
            ("example.com", "/huge"): httpx.Response(200, text=huge_html),
        },
        monkeypatch,
    )

    async with WebSearchBackend(base_url="http://searxng:8080", extract_content=True) as backend:
        result = await backend.call_tool(WEB_SEARCH_TOOL_NAME, {"query": "x"})

    # Either extracted (truncated) content or snippet fallback — both fine,
    # as long as we returned in finite time and finite memory.
    assert "[1] Huge" in result
    assert "https://example.com/huge" in result


@pytest.mark.asyncio
async def test_fetch_capped_buffer_never_exceeds_max_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    """The internal buffer is capped strictly at ``_FETCH_MAX_BYTES``.

    Earlier the loop appended each chunk in full before checking the cap,
    so the buffer could overshoot by up to one chunk-size and then peak
    at ~2x while ``b"".join(...)`` materialised the final bytestring.
    Now the overshooting chunk is truncated to the remaining budget.
    """
    from gateway.services import web_search_backend as wsb_module

    monkeypatch.setattr(wsb_module, "_FETCH_MAX_BYTES", 1024)
    payload = b"<html>" + b"A" * 4096 + b"</html>"
    _patched_async_client(
        {
            ("searxng", "/search"): httpx.Response(
                200,
                json={"results": [{"url": "https://example.com/p", "title": "T", "content": "snip"}]},
            ),
            ("example.com", "/p"): httpx.Response(200, content=payload),
        },
        monkeypatch,
    )

    captured_buffers: list[bytearray] = []
    original_fetch = wsb_module.WebSearchBackend._fetch_capped

    async def spy_fetch(self: wsb_module.WebSearchBackend, url: str) -> str | None:
        out = await original_fetch(self, url)
        # _fetch_capped returns a decoded str; we cannot inspect the bytes
        # buffer post-decode, so assert via the str length cap (utf-8 is a
        # superset of ASCII so 1 byte = 1 char here, giving us a precise
        # bound).
        if out is not None:
            captured_buffers.append(bytearray(out, "utf-8"))
        return out

    monkeypatch.setattr(wsb_module.WebSearchBackend, "_fetch_capped", spy_fetch)

    async with WebSearchBackend(base_url="http://searxng:8080", extract_content=True) as backend:
        await backend.call_tool(WEB_SEARCH_TOOL_NAME, {"query": "x"})

    assert captured_buffers, "fetch was not invoked"
    assert all(len(b) <= 1024 for b in captured_buffers), (
        f"_fetch_capped buffer exceeded cap: sizes={[len(b) for b in captured_buffers]}"
    )


@pytest.mark.parametrize("bad_value", [0, -1, -100])
def test_max_results_clamps_subone_to_one(bad_value: int) -> None:
    """``WebSearchBackend.__init__`` clamps non-positive ``max_results`` to 1.

    Reaches via env var (``GATEWAY_WEB_SEARCH_MAX_RESULTS=-1``) or a
    malformed per-tool entry; downstream the value reaches
    ``results[: self._max_results]`` and would otherwise produce silently-
    wrong slicing (empty list for 0, drop-last for -1).
    """
    backend = WebSearchBackend(base_url="http://searxng:8080", max_results=bad_value)
    assert backend._max_results == 1  # noqa: SLF001 — invariant under test


def test_max_results_clamps_above_cap_to_max_results_cap() -> None:
    """Values above the cap clamp down (existing behaviour, regression guard)."""
    from gateway.services.web_search_backend import _MAX_RESULTS_CAP

    backend = WebSearchBackend(base_url="http://searxng:8080", max_results=10_000)
    assert backend._max_results == _MAX_RESULTS_CAP  # noqa: SLF001


@pytest.mark.asyncio
async def test_auth_token_forwarded_as_gateway_header(monkeypatch: pytest.MonkeyPatch) -> None:
    """When ``auth_token`` is set, the /search request carries X-Gateway-Token.

    This lets the platform-hosted web-search backend authenticate the gateway.
    """
    transport = _patched_async_client(
        {("searxng", "/search"): httpx.Response(200, json=SEARXNG_OK_BODY)},
        monkeypatch,
    )

    async with WebSearchBackend(
        base_url="http://searxng:8080",
        extract_content=False,
        auth_token="gw-secret-token",  # noqa: S106 — test fixture, not a real secret
    ) as backend:
        await backend.call_tool(WEB_SEARCH_TOOL_NAME, {"query": "x"})

    assert transport.captured[0].headers["X-Gateway-Token"] == "gw-secret-token"


@pytest.mark.asyncio
async def test_no_gateway_header_when_auth_token_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """A standalone backend (no auth_token) sends no X-Gateway-Token header."""
    transport = _patched_async_client(
        {("searxng", "/search"): httpx.Response(200, json=SEARXNG_OK_BODY)},
        monkeypatch,
    )

    async with WebSearchBackend(base_url="http://searxng:8080", extract_content=False) as backend:
        await backend.call_tool(WEB_SEARCH_TOOL_NAME, {"query": "x"})

    assert "X-Gateway-Token" not in transport.captured[0].headers


@pytest.mark.asyncio
async def test_auth_token_not_leaked_to_result_page_fetches(monkeypatch: pytest.MonkeyPatch) -> None:
    """With extraction enabled, X-Gateway-Token rides only the /search call to the
    backend — never the per-result page fetches (headers are per-request). Locks
    in the no-leak behaviour against future refactors."""
    transport = _patched_async_client(
        {
            ("searxng", "/search"): httpx.Response(200, json=SEARXNG_OK_BODY),
            ("example.com", "/post-a"): httpx.Response(200, text="<html><body><p>A</p></body></html>"),
            ("example.org", "/post-b"): httpx.Response(200, text="<html><body><p>B</p></body></html>"),
        },
        monkeypatch,
    )

    with patch("gateway.services.web_search_backend.trafilatura.extract") as mock_extract:
        mock_extract.side_effect = lambda html, **_: "extracted"
        async with WebSearchBackend(
            base_url="http://searxng:8080",
            extract_content=True,
            auth_token="gw-secret-token",  # noqa: S106 — test fixture, not a real secret
        ) as backend:
            await backend.call_tool(WEB_SEARCH_TOOL_NAME, {"query": "claude code"})

    search_reqs = [r for r in transport.captured if r.url.path == "/search"]
    fetch_reqs = [r for r in transport.captured if r.url.path in ("/post-a", "/post-b")]
    assert search_reqs and all(r.headers.get("X-Gateway-Token") == "gw-secret-token" for r in search_reqs)
    assert fetch_reqs and all("X-Gateway-Token" not in r.headers for r in fetch_reqs)
