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
replacement, including commercial-API-fronting adapters whose response
sets the optional ``extracted_content`` field to bypass the
gateway-side trafilatura step, or the optional ``published_date`` field
(e.g. the Brave adapter) to surface a result's recency to the model.

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
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import AsyncExitStack
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import httpx
import trafilatura
from opentelemetry import trace

from gateway.heap import release_free_heap
from gateway.services.tool_usage import ToolUsageTally
from gateway.services.url_safety import UnsafeURLError, validate_outbound_fetch_url

if TYPE_CHECKING:
    from types import TracebackType

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

# trafilatura is not safe to call from several threads at once: it parses through
# lxml parsers held in its own module globals (``trafilatura.utils.HTML_PARSER``,
# ``trafilatura.xml.CONTROL_PARSER``), and sharing an lxml parser across threads
# corrupts the allocator's bookkeeping. Extracting a search's pages concurrently
# aborted the process outright with a glibc "double free or corruption" in a
# minority of runs over real pages, so every extraction is confined to one
# dedicated worker.
#
# A dedicated executor rather than a lock around ``asyncio.to_thread``: the
# default executor is shared with file extraction, PDF rasterizing and OCR
# (:mod:`gateway.services.file_extractors`), and threads blocked waiting for a
# lock still occupy their slot in that pool. Queued searches would then stall
# unrelated uploads behind them for seconds. One worker gives the same
# serialization while holding one thread total. Module-level, because the unsafe
# state is trafilatura's own module globals and is shared across event loops.
#
# Serializing does cost throughput, since lxml drops the GIL and extraction
# genuinely ran in parallel. Aborting the process takes every in-flight request
# with it, so that is the better trade.
_extract_executor: ThreadPoolExecutor | None = None
_extract_executor_lock = threading.Lock()


def _get_extract_executor() -> ThreadPoolExecutor:
    """The single worker every page extraction runs on, created on first search."""
    global _extract_executor
    with _extract_executor_lock:
        if _extract_executor is None:
            _extract_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="otari-extract")
    return _extract_executor


def _extract_markdown(html: str) -> str | None:
    """Extract Markdown from one page. Only ever called on the extraction worker."""
    result = trafilatura.extract(
        html,
        output_format="markdown",
        include_comments=False,
        include_tables=True,
        favor_recall=True,
    )
    return str(result) if result else None


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
# published_date is backend-controlled (whatever a search-API-fronting
# adapter forwards from the provider); bound it as basic rendering hygiene, a
# single overlong or multiline value here shouldn't be the thing that makes
# the result block hard to read.
_PUBLISHED_DATE_MAX_CHARS = 128

_DEFAULT_PURPOSE_HINT = (
    "Prefer `web_search` for current information, news, recent events, "
    "documentation lookups, or any question whose answer changes over time. "
    "Returns ranked results with extracted page content where available."
)


def web_search_tool_definition() -> dict[str, Any]:
    """The OpenAI-shaped function definition the model is given for web search.

    Module-level, and returning a fresh dict per call, so the ``/v1/tools``
    discovery endpoint can advertise the same schema the tool loop injects without
    constructing a backend (or risking a shared mutable constant).
    """
    return {
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
        tally: ToolUsageTally | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        # Per-request accounting, owned by the route and passed in. None when the
        # backend runs outside a billed request (tests, direct use).
        self._tally = tally
        self._engines = engines
        # Clamp to [1, _MAX_RESULTS_CAP]. Sub-1 values (e.g. ``0`` or ``-1``
        # from a misconfigured env var) would otherwise reach
        # ``results[: self._max_results]`` and produce surprising slicing
        # behavior (empty list or "drop the last hit") instead of a useful
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
        # Structured hits from the most recent ``call_tool``, kept so a caller that
        # speaks a native server-tool vocabulary can turn them into citation blocks
        # (``take_last_results``). The formatted string the model consumes has
        # already flattened them. Safe as single-slot state because every tool loop
        # awaits its calls one at a time.
        self._last_results: list[dict[str, Any]] = []

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
        return [web_search_tool_definition()]

    def owns_tool(self, name: str) -> bool:
        return name == WEB_SEARCH_TOOL_NAME

    def purpose_hints(self) -> list[tuple[str, str]]:
        return [(WEB_SEARCH_TOOL_NAME, self._purpose_hint)]

    def take_last_results(self) -> list[dict[str, Any]]:
        """Structured hits from the last ``call_tool``, clearing them.

        Consumed right after each awaited call by a loop building native
        server-tool result blocks. Clearing means a later call that fails, or one
        whose loop doesn't ask, cannot attribute the previous call's hits to itself.
        """
        results = self._last_results
        self._last_results = []
        return results

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Run a search and record it on the request's tally.

        Recording happens here, not in the tool loop, because this is the only
        place that knows a call actually reached the backend. The loop converts
        every failure to a ``[tool error]`` string for the model, which cannot
        distinguish a search that failed from one that never ran.
        """
        if name != WEB_SEARCH_TOOL_NAME:
            raise KeyError(f"WebSearchBackend does not own tool {name!r}")
        try:
            result = await self._search_tool(arguments)
        except Exception:
            if self._tally is not None:
                self._tally.record_failure(WEB_SEARCH_TOOL_NAME)
            raise
        if self._tally is not None:
            self._tally.record_result(WEB_SEARCH_TOOL_NAME, result)
        return result

    async def _search_tool(self, arguments: dict[str, Any]) -> str:
        if self._client is None:
            raise RuntimeError("WebSearchBackend not entered as an async context manager")

        # Cleared up front so an early return or a raise below cannot leave the
        # previous call's hits behind for this one to claim.
        self._last_results = []

        with tracer.start_as_current_span(
            WEB_SEARCH_TOOL_NAME,
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            span.set_attribute("tool.name", WEB_SEARCH_TOOL_NAME)
            span.set_attribute("tool.type", "otari_web_search")
            query = (arguments.get("query") or "").strip()
            span.set_attribute("web_search.query", query)
            span.set_attribute("web_search.provider", ",".join(self._engines))
            span.set_attribute("web_search.backend_url", self._base_url)
            if not query:
                span.set_status(trace.StatusCode.ERROR, "empty query")
                return "[tool error] empty query"
            try:
                raw_results = await self._search(query)
            except (httpx.HTTPError, ValueError, KeyError) as exc:
                span.record_exception(exc)
                span.set_status(trace.StatusCode.ERROR, str(exc))
                raise WebSearchNotReachableError(f"web_search failed against {self._base_url}: {exc}") from exc

            filtered = self._apply_domain_filters(raw_results)[: self._max_results]
            if self._extract_content:
                await self._enrich_with_extracted_content(filtered)

            span.set_attribute("web_search.result_count", len(filtered))
            self._last_results = filtered
            return _format_results_for_model(query, filtered)

    # ----- internals -----

    async def _search(self, query: str) -> list[dict[str, Any]]:
        """Issue the backend's ``/search`` GET.

        ``q`` / ``format`` / ``engines`` are the fixed SearXNG params. Any
        configured ``provider_options`` are forwarded as additional query
        params so the backend (the adapter) can interpret provider-specific
        knobs; the gateway does not interpret these keys itself. Only scalar
        values (str / int / float / bool) are forwarded — bools serialize as
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

        try:
            await asyncio.gather(*(one(r) for r in results), return_exceptions=False)
        finally:
            # Parsing several pages of HTML strands tens of megabytes per search
            # in glibc's arenas; without this the resident set plateaus at its
            # high-water mark for the life of the process. Runs on the failure
            # path too, which allocated just the same.
            release_free_heap()

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
        # it off the loop so the loop stays responsive while the other pages in
        # this search are still being fetched.
        try:
            loop = asyncio.get_running_loop()
            extracted = await loop.run_in_executor(_get_extract_executor(), _extract_markdown, html)
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
        # Optional: only backends that supply a recency signal (e.g. the
        # Brave adapter's provider_options time_range support) set this. The
        # model can't judge how current a result is from the snippet alone,
        # so surface it when present instead of silently dropping it.
        # Collapsed to one line and length-capped as rendering hygiene, not a
        # security boundary: title/content above aren't normalized the same
        # way, so this alone doesn't stop a compromised backend from
        # injecting newlines into the block; it just keeps this one new field
        # from being the messiest part of it.
        published_date = " ".join(str(r.get("published_date") or "").split())[:_PUBLISHED_DATE_MAX_CHARS]
        header = f"[{i}] {title}" + (f" ({published_date})" if published_date else "")
        parts.append(f"{header}\n{url}\n{body}".rstrip())
    return "\n\n".join(parts)
