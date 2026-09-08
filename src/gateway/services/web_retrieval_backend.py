"""Search dispatch backed by the shared bounded web-retrieval service.

Search execution uses its configured trusted backend. Public result-page
retrieval is separate: every destination is canonicalized, resolved, validated,
and dialed through the pinned transport before bounded extraction. Per-page
failures remain best effort and fall back to provider snippets.
"""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from typing import TYPE_CHECKING, Any

import httpx
from opentelemetry import trace

from gateway.services.tool_usage import ToolUsageTally
from gateway.services.web_fetch_service import WebFetchService
from gateway.services.web_retrieval_network import PinnedAsyncHTTPTransport, truncate_utf8
from gateway.services.web_retrieval_policy import (
    DomainPolicy,
    WebURLValidationError,
    canonicalize_domain_rules,
    canonicalize_web_url,
)
from gateway.services.web_search_providers import WebSearchProviderError, provider_search

if TYPE_CHECKING:
    from types import TracebackType

tracer = trace.get_tracer(__name__)


WEB_SEARCH_TOOL_NAME = "web_search"

# Gateway-controlled /search query params that provider_options must never override.
_RESERVED_SEARCH_PARAMS = frozenset({"q", "format", "engines"})

_DEFAULT_SEARCH_TIMEOUT_S = 15.0
# Public alongside the cap below: `routes/_tools.web_search_max_results_baseline`
# needs the value a request gets when neither the deployment nor the request
# names one, because that is what a workspace ceiling is floored against.
DEFAULT_MAX_RESULTS = 5
# Public because a workspace's stored web-search ceiling is validated against it
# (`services/tenancy/workspace_web_search_service.py`): a value above what this
# backend will honor could never take effect, so it is refused at the write.
MAX_RESULTS_CAP = 20
_DEFAULT_EXTRACT_CONCURRENCY = 5
WEB_RETRIEVAL_RESULT_MAX_BYTES = 50 * 1024
_RESULT_TRUNCATION_NOTICE = "\n\n[Content truncated at the 50 KiB tool-result limit.]"
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


class WebRetrievalBackend:
    """Own the trusted Search client and pinned public retrieval client."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        provider: str | None = None,
        provider_api_key: str | None = None,
        engines: tuple[str, ...] = _DEFAULT_ENGINES,
        max_results: int = DEFAULT_MAX_RESULTS,
        allowed_domains: tuple[str, ...] = (),
        blocked_domains: tuple[str, ...] = (),
        extract_content: bool = True,
        extract_concurrency: int = _DEFAULT_EXTRACT_CONCURRENCY,
        search_timeout_s: float = _DEFAULT_SEARCH_TIMEOUT_S,
        purpose_hint: str | None = None,
        provider_options: dict[str, Any] | None = None,
        auth_token: str | None = None,
        tally: ToolUsageTally | None = None,
        retrieval_service: WebFetchService | None = None,
    ) -> None:
        # Exactly one of the two search paths, checked here rather than at the
        # first query: a backend with neither would raise mid-completion, after
        # the request has already been admitted and billed for its first turn.
        if not base_url and not (provider and provider_api_key):
            msg = "WebRetrievalBackend needs either a base_url or a provider with its api key"
            raise ValueError(msg)
        self._base_url = base_url.rstrip("/") if base_url else None
        # A licensed search API this process calls itself, instead of the
        # SearXNG-shaped service at ``base_url``. Set when the deployment
        # configured one and holds its key, which on a hosted deployment is the
        # control plane and never the data plane.
        self._provider = provider
        self._provider_api_key = provider_api_key
        # Per-request accounting, owned by the route and passed in. None when the
        # backend runs outside a billed request (tests, direct use).
        self._tally = tally
        self._engines = engines
        # Clamp to [1, MAX_RESULTS_CAP]. Sub-1 values (e.g. ``0`` or ``-1``
        # from a misconfigured env var) would otherwise reach
        # ``results[: self._max_results]`` and produce surprising slicing
        # behavior (empty list or "drop the last hit") instead of a useful
        # bound.
        self._max_results = max(1, min(max_results, MAX_RESULTS_CAP))
        self._domain_policy = DomainPolicy(
            allowed=canonicalize_domain_rules(allowed_domains),
            blocked=canonicalize_domain_rules(blocked_domains),
        )
        self._extract_content = extract_content
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
        self._retrieval_service = retrieval_service
        self._stack: AsyncExitStack = AsyncExitStack()
        # Structured hits from the most recent ``call_tool``, kept so a caller that
        # speaks a native server-tool vocabulary can turn them into citation blocks
        # (``take_last_results``). The formatted string the model consumes has
        # already flattened them. Safe as single-slot state because every tool loop
        # awaits its calls one at a time.
        self._last_results: list[dict[str, Any]] = []

    async def __aenter__(self) -> WebRetrievalBackend:
        self._client = await self._stack.enter_async_context(httpx.AsyncClient(timeout=self._search_timeout_s))
        if self._retrieval_service is None:
            retrieval_client = await self._stack.enter_async_context(
                httpx.AsyncClient(
                    transport=PinnedAsyncHTTPTransport(),
                    timeout=None,
                    follow_redirects=False,
                    trust_env=False,
                )
            )
            self._retrieval_service = WebFetchService(retrieval_client)
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
            raise KeyError(f"WebRetrievalBackend does not own tool {name!r}")
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
            raise RuntimeError("WebRetrievalBackend not entered as an async context manager")

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
            span.set_attribute("web_search.provider", self._provider or ",".join(self._engines))
            if not query:
                span.set_status(trace.StatusCode.ERROR, "empty_query")
                return "[tool error] empty query"
            try:
                raw_results = await self._search(query)
            except (httpx.HTTPError, WebSearchProviderError, ValueError, KeyError) as exc:
                span.set_status(trace.StatusCode.ERROR, "search_backend_failed")
                raise WebSearchNotReachableError("web_search backend could not be reached") from exc

            filtered = self._apply_domain_filters(raw_results)[: self._max_results]
            if self._extract_content:
                await self._enrich_with_extracted_content(filtered)

            span.set_attribute("web_search.result_count", len(filtered))
            self._last_results = filtered
            formatted = _format_results_for_model(query, filtered)
            return truncate_utf8(
                formatted,
                WEB_RETRIEVAL_RESULT_MAX_BYTES,
                suffix=_RESULT_TRUNCATION_NOTICE,
            ).text

    # ----- internals -----

    async def _search(self, query: str) -> list[dict[str, Any]]:
        """Run one search, through whichever backend this deployment configured.

        A configured licensed provider wins over ``base_url``: an operator who
        set both named the provider deliberately, and the URL is the fallback
        the bundled SearXNG container fills.
        """
        assert self._client is not None
        if self._provider and self._provider_api_key:
            return await provider_search(
                provider=self._provider,
                api_key=self._provider_api_key,
                query=query,
                # The same opaque bag the SearXNG path forwards as query params.
                # Each provider whitelists the keys it understands. The resolved
                # ceiling leads it as the default: asking the provider for fewer
                # hits than the deployment allows and then slicing to the same
                # number is how a raised ``web_search_max_results`` used to have
                # no effect. An explicit ``provider_options`` value still wins.
                options={"max_results": self._max_results, **self._provider_options},
                client=self._client,
                timeout_s=self._search_timeout_s,
            )
        return await self._search_over_http(query)

    async def _search_over_http(self, query: str) -> list[dict[str, Any]]:
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
        if not self._domain_policy.allowed and not self._domain_policy.blocked:
            return results
        kept: list[dict[str, Any]] = []
        for result in results:
            try:
                host = canonicalize_web_url(str(result.get("url"))).origin.host
            except WebURLValidationError:
                continue
            if self._domain_policy.permits(host):
                kept.append(result)
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

        await asyncio.gather(*(one(result) for result in results), return_exceptions=False)

    async def _fetch_and_extract(self, url: str) -> str | None:
        service = self._retrieval_service
        if service is None:
            raise RuntimeError("WebRetrievalBackend not entered as an async context manager")
        try:
            result = await service.fetch_for_search(url, policy=self._domain_policy)
        except Exception:
            return None
        return result.text or None


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
        # Optional: only backends that supply a recency signal set this, which
        # for the first-party providers means Brave answering a `time_range`.
        # The model can't judge how current a result is from the snippet alone,
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


# Compatibility for overlays and integrations that imported the old class name.
WebSearchBackend = WebRetrievalBackend
