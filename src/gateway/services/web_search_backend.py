"""Dispatch `web_search` tool calls to an external search service.

A backend the tool-use loop in :mod:`gateway.services.mcp_loop` dispatches
to whenever the model emits a ``web_search(query=…)`` call. The search
service is operator-configurable via ``OTARI_WEB_SEARCH_URL`` and is
expected to speak a SearXNG-compatible JSON API:

* ``GET {base_url}/search?q=…&format=json&engines=…``
    → returns ``{"results": [{"url", "title", "content", ...}]}``

The default deployment points this at a bundled SearXNG container (see
``docker-compose.yml``, ``web-search`` profile). Any other container
that exposes the same JSON shape on ``/search`` is a drop-in
replacement — including commercial-API-fronting adapters whose response
sets the optional ``extracted_content`` field to bypass the
gateway-side trafilatura step.

After search, the backend optionally fetches the top results' URLs and
runs trafilatura in-process to produce LLM-ready Markdown. Fetch +
extract failures degrade silently to the engine-supplied snippet — the
search itself never fails because one page didn't render.

This backend satisfies the same duck-typed protocol the MCP loop uses
for tool dispatch (``openai_tools``, ``owns_tool``, ``purpose_hints``,
``call_tool``), so the loop accepts it as a ``pool`` without any
refactor to :func:`gateway.services.mcp_loop.mcp_tool_loop`.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import AsyncExitStack
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import httpx
import trafilatura

from gateway.services.url_safety import UnsafeURLError, validate_outbound_fetch_url

if TYPE_CHECKING:
    from types import TracebackType

logger = logging.getLogger(__name__)

WEB_SEARCH_TOOL_NAME = "web_search"

# Gateway-controlled /search query params that provider_options must never override.
_RESERVED_SEARCH_PARAMS = frozenset({"q", "format", "engines"})

_DEFAULT_SEARCH_TIMEOUT_S = 15.0
_DEFAULT_FETCH_TIMEOUT_S = 5.0
_DEFAULT_MAX_RESULTS = 5
_MAX_RESULTS_CAP = 20
_DEFAULT_EXTRACT_CONCURRENCY = 5
# Hard cap on bytes we'll read from a single fetched page before passing to
# trafilatura. A huge response (compromised host, content-bomb, or just a
# legitimately massive page) would otherwise blow memory across N parallel
# fetches. 5 MB of HTML is generous — any well-formed article fits.
_FETCH_MAX_BYTES = 5 * 1024 * 1024
# Bounded redirect walk. httpx's default is 20; we re-validate every hop
# against the SSRF guard (Location headers are attacker-influenced content
# from the upstream page), so the count below limits how many round trips
# we make per fetch attempt.
_FETCH_MAX_REDIRECTS = 5
# Default engine list deliberately excludes Google/Bing/Yahoo (which forbid
# automated querying in their ToS) and Brave (whose paid Search API is the
# licensed path; scraping their public SERP is not what Brave wants).
# duckduckgo/mojeek/qwant/wikipedia is the most defensible OSS default.
# Operators who enable scraping-of-major-engines do so consciously.
# Commercial/production deployments should swap the bundled SearXNG container
# for a licensed-API backend (Tavily, Brave API, Exa, Linkup, Serper) by
# pointing OTARI_WEB_SEARCH_URL at any service exposing the same
# /search?format=json shape.
_DEFAULT_ENGINES = ("duckduckgo", "mojeek", "qwant", "wikipedia")
_CONTENT_TRUNCATE_CHARS = 1500

_DEFAULT_PURPOSE_HINT = (
    "Prefer `web_search` for current information, news, recent events, "
    "documentation lookups, or any question whose answer changes over time. "
    "Returns ranked results with extracted page content where available."
)


class WebSearchNotReachableError(RuntimeError):
    """Raised when the search backend can't be reached or returns malformed data."""


class WebSearchBackend:
    """Async context manager that owns an HTTP client for the search backend's lifetime.

    Usage::

        async with WebSearchBackend(base_url="http://searxng:8080") as backend:
            # backend duck-types as the MCP loop's `pool` parameter
            result = await mcp_tool_loop(
                completion_kwargs=kwargs, pool=backend, max_iterations=N,
            )
    """

    def __init__(
        self,
        *,
        base_url: str,
        engines: tuple[str, ...] = _DEFAULT_ENGINES,
        max_results: int = _DEFAULT_MAX_RESULTS,
        allowed_domains: tuple[str, ...] = (),
        blocked_domains: tuple[str, ...] = (),
        extract_content: bool = True,
        extract_timeout_s: float = _DEFAULT_FETCH_TIMEOUT_S,
        extract_concurrency: int = _DEFAULT_EXTRACT_CONCURRENCY,
        search_timeout_s: float = _DEFAULT_SEARCH_TIMEOUT_S,
        purpose_hint: str | None = None,
        provider_options: dict[str, Any] | None = None,
        auth_token: str | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._engines = engines
        # Clamp to [1, _MAX_RESULTS_CAP]. Sub-1 values (e.g. ``0`` or ``-1``
        # from a misconfigured env var) would otherwise reach
        # ``results[: self._max_results]`` and produce surprising slicing
        # behaviour (empty list or "drop the last hit") instead of a useful
        # bound.
        self._max_results = max(1, min(max_results, _MAX_RESULTS_CAP))
        self._allowed_domains = tuple(d.lower() for d in allowed_domains)
        self._blocked_domains = tuple(d.lower() for d in blocked_domains)
        self._extract_content = extract_content
        self._extract_timeout_s = extract_timeout_s
        self._extract_concurrency = extract_concurrency
        self._search_timeout_s = search_timeout_s
        self._purpose_hint = purpose_hint or _DEFAULT_PURPOSE_HINT
        # Sanitised copy of provider-specific knobs forwarded to the search
        # backend as extra `/search` query params. Only scalar values survive
        # (see `_search`); complex / None values are dropped so a misconfigured
        # entry can't smuggle structured payloads into the GET.
        self._provider_options = dict(provider_options) if provider_options else {}
        # Optional bearer-style credential forwarded as `X-Gateway-Token` on the
        # `/search` request. Set when the search backend is the platform-hosted
        # endpoint (which authenticates the gateway); unset for a standalone
        # SearXNG / self-hosted adapter, which ignores the header.
        self._auth_token = auth_token
        self._client: httpx.AsyncClient | None = None
        self._stack: AsyncExitStack = AsyncExitStack()

    async def __aenter__(self) -> WebSearchBackend:
        # The search call has its own short timeout; per-page fetches use a
        # separate (also short) timeout. Set the client default to the longer
        # of the two so neither path is pre-empted by the client timeout.
        client_timeout = max(self._search_timeout_s, self._extract_timeout_s)
        self._client = await self._stack.enter_async_context(httpx.AsyncClient(timeout=client_timeout))
        return self

    async def __aexit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _tb: TracebackType | None,
    ) -> None:
        await self._stack.aclose()

    # ----- duck-typed protocol the MCP loop uses on `pool` -----

    @property
    def openai_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": WEB_SEARCH_TOOL_NAME,
                    "description": (
                        "Search the web for current information. Returns a ranked list "
                        "of results with URLs, titles, and extracted page content."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query. Use natural language.",
                            },
                        },
                        "required": ["query"],
                    },
                },
            }
        ]

    def owns_tool(self, name: str) -> bool:
        return name == WEB_SEARCH_TOOL_NAME

    def purpose_hints(self) -> list[tuple[str, str]]:
        return [(WEB_SEARCH_TOOL_NAME, self._purpose_hint)]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        if name != WEB_SEARCH_TOOL_NAME:
            raise KeyError(f"WebSearchBackend does not own tool {name!r}")
        if self._client is None:
            raise RuntimeError("WebSearchBackend not entered as an async context manager")

        query = (arguments.get("query") or "").strip()
        if not query:
            return "[tool error] empty query"

        try:
            raw_results = await self._search(query)
        except (httpx.HTTPError, ValueError, KeyError) as exc:
            raise WebSearchNotReachableError(f"web_search failed against {self._base_url}: {exc}") from exc

        filtered = self._apply_domain_filters(raw_results)[: self._max_results]
        if self._extract_content:
            await self._enrich_with_extracted_content(filtered)

        return _format_results_for_model(query, filtered)

    # ----- internals -----

    async def _search(self, query: str) -> list[dict[str, Any]]:
        """Issue the backend's ``/search`` GET.

        ``q`` / ``format`` / ``engines`` are the fixed SearXNG params. Any
        configured ``provider_options`` are forwarded as additional query
        params so the backend (the adapter) can interpret provider-specific
        knobs; the gateway does not interpret these keys itself. Only scalar
        values (str / int / float / bool) are forwarded — bools serialise as
        lowercase ``"true"`` / ``"false"`` — and None / complex values are
        skipped. Reserved gateway-controlled params (``q`` / ``format`` /
        ``engines``) are never overridable by ``provider_options``.
        """
        assert self._client is not None
        params: dict[str, str | int | float] = {
            "q": query,
            "format": "json",
            "engines": ",".join(self._engines),
        }
        for key, value in self._provider_options.items():
            if key in _RESERVED_SEARCH_PARAMS or value is None:
                continue
            if isinstance(value, bool):
                params[key] = "true" if value else "false"
            elif isinstance(value, (str, int, float)):
                params[key] = value
        headers = {"X-Gateway-Token": self._auth_token} if self._auth_token else None
        response = await self._client.get(
            f"{self._base_url}/search",
            params=params,
            headers=headers,
            timeout=self._search_timeout_s,
        )
        response.raise_for_status()
        body = response.json()
        results = body.get("results")
        if not isinstance(results, list):
            raise ValueError(f"backend returned non-list results: {body!r}")
        return [r for r in results if isinstance(r, dict) and r.get("url")]

    def _apply_domain_filters(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not self._allowed_domains and not self._blocked_domains:
            return results
        kept: list[dict[str, Any]] = []
        for r in results:
            host = (urlparse(str(r.get("url"))).hostname or "").lower()
            if self._blocked_domains and any(host == d or host.endswith("." + d) for d in self._blocked_domains):
                continue
            if self._allowed_domains and not any(host == d or host.endswith("." + d) for d in self._allowed_domains):
                continue
            kept.append(r)
        return kept

    async def _enrich_with_extracted_content(self, results: list[dict[str, Any]]) -> None:
        """Populate ``extracted_content`` on each result, in place.

        Skips results whose backend already supplied ``extracted_content``
        (e.g. a Tavily-fronting adapter). Failures degrade silently.
        """
        sem = asyncio.Semaphore(self._extract_concurrency)

        async def one(result: dict[str, Any]) -> None:
            if result.get("extracted_content"):
                return
            url = str(result["url"])
            async with sem:
                content = await self._fetch_and_extract(url)
            if content:
                result["extracted_content"] = content

        await asyncio.gather(*(one(r) for r in results), return_exceptions=False)

    async def _fetch_and_extract(self, url: str) -> str | None:
        assert self._client is not None

        # SSRF guard: search-engine results are user-influenced content. A
        # malicious or compromised engine could return URLs pointing at
        # internal services (cloud metadata, RFC1918, loopback). Validate
        # before any network round trip.
        try:
            await validate_outbound_fetch_url(url)
        except UnsafeURLError as exc:
            logger.warning("web_search: refusing to fetch %s: %s", url, exc)
            return None

        try:
            html = await self._fetch_capped(url)
        except httpx.HTTPError as exc:
            logger.debug("web_search: fetch failed for %s: %s", url, exc)
            return None
        if html is None:
            return None

        # trafilatura.extract is synchronous and CPU-bound on large inputs; run
        # in a thread so the event loop stays responsive while many pages are
        # extracted in parallel.
        try:
            extracted = await asyncio.to_thread(
                trafilatura.extract,
                html,
                output_format="markdown",
                include_comments=False,
                include_tables=True,
                favor_recall=True,
            )
        except Exception as exc:  # noqa: BLE001 — trafilatura raises broad
            logger.debug("web_search: extract failed for %s: %s", url, exc)
            return None
        return extracted or None

    async def _fetch_capped(self, url: str) -> str | None:
        """Fetch ``url`` as text, refusing to read past ``_FETCH_MAX_BYTES``.

        Streams the response and stops once the cap is reached — never
        allocates a >5MB buffer regardless of what the upstream sends.

        Redirects are walked manually so every hop's URL can be re-validated
        against the SSRF guard. httpx's ``follow_redirects=True`` would
        bypass the per-URL check (a publicly-resolving result page can 302
        the gateway to ``169.254.169.254`` / loopback / RFC1918), so we
        explicitly disable it and bound the walk via
        :data:`_FETCH_MAX_REDIRECTS`. Returns ``None`` on non-2xx, redirect
        without a Location, blocked Location, or too many hops; raises
        ``httpx.HTTPError`` on transport failure.
        """
        assert self._client is not None
        current_url = url
        for _ in range(_FETCH_MAX_REDIRECTS + 1):
            async with self._client.stream(
                "GET",
                current_url,
                timeout=self._extract_timeout_s,
                follow_redirects=False,
                headers={"User-Agent": "Mozilla/5.0 (compatible; otari-web-search)"},
            ) as response:
                if response.is_redirect:
                    location = response.headers.get("location")
                    if not location:
                        return None
                    next_url = str(response.url.join(location))
                    try:
                        await validate_outbound_fetch_url(next_url)
                    except UnsafeURLError as exc:
                        logger.warning(
                            "web_search: refusing to follow redirect %s -> %s: %s",
                            current_url,
                            next_url,
                            exc,
                        )
                        return None
                    current_url = next_url
                    continue
                if response.status_code >= 400:
                    return None
                # Accumulate directly into a bytearray and truncate the
                # chunk that crosses the cap to the remaining budget. The
                # previous list-of-chunks + ``b"".join(...)`` pattern (a)
                # overshot by up to one chunk-size and (b) briefly held
                # two copies during join — under fetch concurrency, peak
                # memory was ~2× the cap. bytearray + decode is one copy.
                buf = bytearray()
                async for chunk in response.aiter_bytes(chunk_size=65536):
                    remaining = _FETCH_MAX_BYTES - len(buf)
                    if remaining <= 0:
                        break
                    buf.extend(chunk if len(chunk) <= remaining else chunk[:remaining])
                    if len(buf) >= _FETCH_MAX_BYTES:
                        break
                # Best-effort decode: trafilatura tolerates partial / encoding-noisy HTML.
                return buf.decode(response.encoding or "utf-8", errors="replace")
        # Exceeded redirect budget without reaching a terminal response.
        logger.debug("web_search: redirect budget exhausted for %s", url)
        return None


def _format_results_for_model(query: str, results: list[dict[str, Any]]) -> str:
    """Render results as compact Markdown for tool-message consumption.

    Numbered so the model can refer to ``[1]``, ``[2]`` in its answer — gives
    us a clean v2 path to extract structured citations later without changing
    the v1 wire format.
    """
    if not results:
        return f"No results for query: {query!r}"

    parts: list[str] = []
    for i, r in enumerate(results, start=1):
        title = str(r.get("title") or "(untitled)").strip()
        url = str(r.get("url") or "").strip()
        snippet = str(r.get("content") or "").strip()
        extracted = str(r.get("extracted_content") or "").strip()
        body = extracted or snippet
        if len(body) > _CONTENT_TRUNCATE_CHARS:
            body = body[:_CONTENT_TRUNCATE_CHARS].rstrip() + "…"
        parts.append(f"[{i}] {title}\n{url}\n{body}".rstrip())
    return "\n\n".join(parts)
