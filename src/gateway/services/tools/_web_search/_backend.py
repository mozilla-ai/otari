"""The ``web_search`` tool on the search the resolver chose, through any-search (design H, steps 1 to 7).

Each call takes a slot of the request's shared search and fetch allowance
first; past it the call is refused with ``[tool error] web search call limit
reached for this request``. The search runs through the app's web provider
client with the resolved key, options and timeout, and always asks for the
request's count. Its hits are held to the search declaration's domain rules and
cut to the count, and a hit that came without page text is enriched through the
fetch instance the resolver chose for it, under the same domain rules, the
first 20 of a search at most; a page that cannot be fetched leaves the hit its
snippet. Enrichment takes no slot of the allowance. Its pages are billed as
fetches on the instance that read them: before any is read, their price at that
instance's rate is one hold on the request's budgets, and a hold that does not
fit leaves every hit its snippet; each page that comes back is recorded as a
``web_fetch`` call under the instance's label, at the cost the provider
reported or else the instance's rate.

``builtin`` enriches through the per-request retrieval client and checks every
redirect against the domain rules, reaching a private host only when
``web_search_allow_private_hosts`` is on. A provider fetch instance is checked
before the call and on the URL it reports after it, as the fetch tool checks
one. A provider that does not report its final URL, as Exa and Tabstack do
not, can follow a redirect off the domain rules. Such a provider may also
answer with a site's error page as the page's text, so its page follows the
hit's snippet rather than replacing it.

What the model reads is numbered blocks whose numbering continues across the
request's searches, each body cut at 1,500 characters, the whole capped at
50 KiB. A failure is returned rather than raised, since a raised error reaches
the model only as the loops' generic text: ``[tool error] <provider> search
failed (<tag>)``, with ``, HTTP <status>`` when there is one. None carries the
query, a URL, a body or an exception's text, and the failed call's safe fields
are kept for the Chat Completions rendering (decision 67).
"""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Self

from opentelemetry import trace

import any_search
from any_fetch import FetchedPage
from gateway.core.settings.tools import BUILTIN_FETCH_PROVIDER
from gateway.log_config import logger
from gateway.services.tool_usage import TOOL_BUDGET_EXCEEDED_TEXT, is_tool_error
from gateway.services.tools._web_call_cap import SEARCH_CALL_LIMIT_TEXT, WebToolCallCap
from gateway.services.tools._web_display import published_date
from gateway.services.tools._web_fetch import FetchFailure, fetch_page, retrieval_client
from gateway.services.tools._web_names import (
    WEB_FETCH_TOOL_NAME,
    WEB_RETRIEVAL_RESULT_MAX_BYTES,
    WEB_SEARCH_PURPOSE_HINT,
    WEB_SEARCH_TOOL_NAME,
    web_search_tool_definition,
)
from gateway.services.tools._web_provider_client import web_provider_client_for
from gateway.services.tools._web_tool_failure import (
    BUDGET_EXCEEDED_TAG,
    CALL_LIMIT_TAG,
    INTERNAL_ERROR_TAG,
    MISSING_CREDENTIAL_TAG,
    UNSUPPORTED_OPTION_TAG,
    WebToolFailure,
    failed_text,
)
from gateway.services.traces import ToolSpans
from gateway.services.web_retrieval_network import truncate_utf8
from gateway.services.web_retrieval_policy import DomainPolicy, WebURLValidationError, canonicalize_web_url

if TYPE_CHECKING:
    from types import TracebackType

    import httpx

    from gateway.services.tool_usage import ToolUsageTally
    from gateway.services.tools._web_search._resolve import InLoopSearch
    from gateway.services.traces import RequestTrace

_tracer = trace.get_tracer(__name__)

EMPTY_QUERY_TAG = "empty_query"

_EMPTY_QUERY_TEXT = "[tool error] empty query"
_RESULT_TRUNCATION_NOTICE = "\n\n[Content truncated at the 50 KiB tool-result limit.]"
# How much of each hit's text the model reads.
_BODY_MAX_CHARS = 1500
# One more than a hit's body shows, so an enriched page longer than that still ends in the ellipsis.
_ENRICHMENT_MAX_CHARS = _BODY_MAX_CHARS + 1
# How many of one search's pages are fetched at once, and at most in all: the older backend's
# ceiling of 20 results a search, so a count above it shows the rest by their snippets.
_ENRICHMENT_CONCURRENCY = 5
_ENRICHMENT_MAX_HITS = 20


@dataclass(frozen=True)
class _Searched:
    """One call's text for the model, and the cost its search reported."""

    text: str
    cost: Decimal | None = None


class WebSearchBackend:
    """One request's ``web_search`` tool, on one search and the fetch that enriches its hits.

    Opened around the tool loop as an async context manager. It keeps the last
    call's hits and failure for the renderings, each taken once.
    """

    def __init__(
        self,
        search: InLoopSearch,
        *,
        cap: WebToolCallCap,
        max_results: int,
        policy: DomainPolicy | None = None,
        enrich: bool = True,
        allow_private: bool = False,
        trust_env_proxy: bool = False,
        purpose_hint: str | None = None,
        web_provider_client: httpx.AsyncClient | None = None,
        tally: ToolUsageTally | None = None,
        trace: RequestTrace | None = None,
    ) -> None:
        self._search = search.search
        self._enrichment = search.enrichment
        self._builtin_enrichment = search.enrichment.provider == BUILTIN_FETCH_PROVIDER
        self._cap = cap
        self._max_results = max_results
        self._policy = policy or DomainPolicy()
        self._enrich = enrich
        self._allow_private = allow_private
        self._trust_env_proxy = trust_env_proxy
        self._purpose_hint = purpose_hint or WEB_SEARCH_PURPOSE_HINT
        # The app's client, which the search and a provider fetch instance are called through.
        # None outside an app, where each provider opens its own per call.
        self._web_provider_client = web_provider_client
        self._tally = tally
        self._spans = ToolSpans(trace, "otari_web_search", WEB_SEARCH_TOOL_NAME)
        self._stack = AsyncExitStack()
        # The built-in fetcher's client for enrichment, opened only where it enriches.
        self._retrieval_client: httpx.AsyncClient | None = None
        self._entered = False
        # The hits shown so far, which the next call's numbering continues from.
        self._numbered = 0
        self._last_hits: list[any_search.SearchHit] = []
        self._last_failure: WebToolFailure | None = None

    async def __aenter__(self) -> Self:
        if self._enrich and self._builtin_enrichment:
            self._retrieval_client = await self._stack.enter_async_context(
                retrieval_client(trust_env_proxy=self._trust_env_proxy)
            )
        self._entered = True
        return self

    async def __aexit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _tb: TracebackType | None,
    ) -> None:
        self._entered = False
        self._retrieval_client = None
        await self._stack.aclose()

    @property
    def openai_tools(self) -> list[dict[str, Any]]:
        return [web_search_tool_definition()]

    def owns_tool(self, name: str) -> bool:
        return name == WEB_SEARCH_TOOL_NAME

    def purpose_hints(self) -> list[tuple[str, str]]:
        return [(WEB_SEARCH_TOOL_NAME, self._purpose_hint)]

    def take_last_hits(self) -> list[any_search.SearchHit]:
        """The hits the last call showed the model, clearing them; empty when it showed none."""
        hits, self._last_hits = self._last_hits, []
        return hits

    def take_last_failure(self) -> WebToolFailure | None:
        """The last call's failure, clearing it; ``None`` when it did not fail."""
        failure, self._last_failure = self._last_failure, None
        return failure

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Search once and record the call on the request's tally and trace."""
        if not self.owns_tool(name):
            raise KeyError(f"WebSearchBackend does not own tool {name!r}")
        if not self._entered:
            raise RuntimeError("WebSearchBackend not entered as an async context manager")
        self._last_hits = []
        self._last_failure = None
        started = datetime.now(UTC)
        with _tracer.start_as_current_span(
            WEB_SEARCH_TOOL_NAME, record_exception=False, set_status_on_exception=False
        ) as span:
            span.set_attribute("tool.name", WEB_SEARCH_TOOL_NAME)
            span.set_attribute("tool.type", "otari_web_search")
            span.set_attribute("web_search.provider", self._search.provider)
            try:
                if not self._cap.claim():
                    searched = _Searched(self._refused(CALL_LIMIT_TAG, SEARCH_CALL_LIMIT_TEXT))
                elif self._tally is not None and not await self._tally.admit(
                    WEB_SEARCH_TOOL_NAME,
                    label=self._search.label,
                    rate_keys=self._search.rate_keys,
                    deployment_paid=not self._search.on_credential,
                ):
                    searched = _Searched(self._refused(BUDGET_EXCEEDED_TAG, TOOL_BUDGET_EXCEEDED_TEXT))
                else:
                    searched = await self._call(arguments)
            except Exception:
                # Every failure the backend knows is returned; one it does not is counted and
                # traced as failed before the loop turns it into its generic text.
                span.set_status(trace.StatusCode.ERROR, INTERNAL_ERROR_TAG)
                if self._tally is not None:
                    self._tally.record_failure(WEB_SEARCH_TOOL_NAME)
                self._spans.record(started, ok=False, arguments=arguments)
                raise
            if self._last_failure is not None:
                span.set_status(trace.StatusCode.ERROR, self._last_failure.tag)
            else:
                span.set_attribute("web_search.result_count", len(self._last_hits))
        if self._tally is not None:
            # Under what served it (design L): the label and rate keys the resolver gave, and the
            # search's reported cost, which ``None`` leaves to the rate.
            self._tally.record_result(
                WEB_SEARCH_TOOL_NAME,
                searched.text,
                label=self._search.label,
                rate_keys=self._search.rate_keys,
                cost=searched.cost,
                deployment_paid=not self._search.on_credential,
            )
        self._spans.record(started, ok=not is_tool_error(searched.text), arguments=arguments, result=searched.text)
        return searched.text

    async def _call(self, arguments: dict[str, Any]) -> _Searched:
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            return _Searched(self._refused(EMPTY_QUERY_TAG, _EMPTY_QUERY_TEXT))
        query = query.strip()
        resolved = self._search
        try:
            async with any_search.AnySearch.create(
                resolved.provider,
                api_key=resolved.api_key,
                api_base=resolved.api_base,
                timeout=resolved.timeout,
                client=web_provider_client_for(self._web_provider_client, tenant_endpoint=resolved.tenant_endpoint),
            ) as engine:
                # The count is always passed, so a native count option cannot raise it.
                result = await engine.search(query, max_results=self._max_results, **resolved.options)
        except any_search.ProviderError as exc:
            return _Searched(self._failed(WebToolFailure.from_provider_error(exc), exc.status))
        except any_search.MissingCredentialError as exc:
            return _Searched(self._failed(self._raised(exc, MISSING_CREDENTIAL_TAG), None))
        except any_search.UnsupportedParameterError as exc:
            return _Searched(self._failed(self._raised(exc, UNSUPPORTED_OPTION_TAG), None))
        except Exception as exc:
            return _Searched(self._failed(self._raised(exc, INTERNAL_ERROR_TAG), None))
        if (error := result.error) is not None:
            return _Searched(self._failed(WebToolFailure.of(resolved.provider, error.status, error.tag), error.status))
        hits = [hit for hit in result.hits if self._permits(hit.url)][: self._max_results]
        bodies = await self._enriched(hits) if self._enrich else {}
        self._last_hits = hits
        first, self._numbered = self._numbered + 1, self._numbered + len(hits)
        text = truncate_utf8(
            _results_text(query, hits, bodies, first=first),
            WEB_RETRIEVAL_RESULT_MAX_BYTES,
            suffix=_RESULT_TRUNCATION_NOTICE,
        ).text
        return _Searched(text, result.cost)

    def _permits(self, url: str) -> bool:
        """Whether a hit's URL passes the search's domain rules; one that cannot be read does not."""
        try:
            return self._policy.permits(canonicalize_web_url(url).origin.host)
        except WebURLValidationError:
            return False

    async def _enriched(self, hits: list[any_search.SearchHit]) -> dict[int, str]:
        """The text each hit without any is shown from its page, by its index.

        The pages are held on the request's budgets as one hold before any is
        read, which stays until settlement as the model call's own does, and
        each that comes back is recorded as a fetch on the instance that read
        it, with the cost the provider reported.
        """
        wanted = [index for index, hit in enumerate(hits) if not _text(hit)][:_ENRICHMENT_MAX_HITS]
        if not wanted:
            return {}
        enrichment = self._enrichment
        if self._tally is not None and not await self._tally.admit(
            WEB_FETCH_TOOL_NAME,
            label=enrichment.label,
            rate_keys=enrichment.rate_keys,
            deployment_paid=not enrichment.on_credential,
            calls=len(wanted),
        ):
            logger.info(
                "Enrichment through '%s' (%s) left %d results their snippets: their pages do not fit the budget",
                enrichment.name or enrichment.label,
                enrichment.provider,
                len(wanted),
            )
            return {}
        semaphore = asyncio.Semaphore(_ENRICHMENT_CONCURRENCY)

        async def page(url: str) -> FetchedPage | None:
            async with semaphore:
                return await self._page(url)

        pages = await asyncio.gather(*(page(hits[index].url) for index in wanted))
        bodies: dict[int, str] = {}
        for index, fetched in zip(wanted, pages, strict=True):
            if fetched is None:
                continue
            if self._tally is not None:
                self._tally.record_success(
                    WEB_FETCH_TOOL_NAME,
                    label=enrichment.label,
                    rate_keys=enrichment.rate_keys,
                    cost=fetched.cost,
                    deployment_paid=not enrichment.on_credential,
                )
            if text := fetched.text.strip():
                snippet = hits[index].snippet.strip()
                bodies[index] = text if self._builtin_enrichment or not snippet else f"{snippet}\n\n{text}"
        return bodies

    async def _page(self, url: str) -> FetchedPage | None:
        """The page at a hit's URL, or ``None`` when it cannot be fetched or is refused."""
        enrichment = self._enrichment
        fetched = await fetch_page(
            enrichment,
            url,
            client=self._retrieval_client if self._builtin_enrichment else self._web_provider_client,
            policy=self._policy,
            allow_private=self._allow_private,
            max_chars=min(enrichment.max_chars or _ENRICHMENT_MAX_CHARS, _ENRICHMENT_MAX_CHARS),
        )
        if isinstance(fetched, FetchFailure):
            logger.debug(
                "Enrichment through '%s' (%s) left a result its snippet: %s",
                enrichment.name or enrichment.label,
                enrichment.provider,
                fetched.failure.tag,
            )
            return None
        return fetched

    def _raised(self, exc: Exception, tag: str) -> WebToolFailure:
        # The class alone: an exception's text can carry the query.
        resolved = self._search
        logger.warning(
            "Search on '%s' (%s) failed: %s", resolved.name or resolved.label, resolved.provider, type(exc).__name__
        )
        return WebToolFailure.of(resolved.provider, None, tag)

    def _refused(self, tag: str, text: str) -> str:
        """A call refused before any provider was called."""
        self._last_failure = WebToolFailure.of(None, None, tag)
        return text

    def _failed(self, failure: WebToolFailure, status: int | None) -> str:
        """The text for a failed search, which names ``status`` as the provider gave it.

        The failure itself keeps a status only from 400, as every web tool failure does.
        """
        self._last_failure = failure
        return f"[tool error] {failed_text(failure.provider, 'search', failure.tag, status)}"


def _results_text(query: str, hits: list[any_search.SearchHit], bodies: dict[int, str], *, first: int) -> str:
    """The hits as numbered blocks the model can cite, from ``[first]``."""
    if not hits:
        return f"No results for query: {query!r}"
    parts: list[str] = []
    for number, (index, hit) in enumerate(enumerate(hits), start=first):
        title = hit.title.strip() or "(untitled)"
        body = bodies.get(index) or _text(hit) or hit.snippet.strip()
        if len(body) > _BODY_MAX_CHARS:
            body = body[:_BODY_MAX_CHARS].rstrip() + "…"
        date = published_date(hit.published)
        header = f"[{number}] {title}" + (f" ({date})" if date else "")
        parts.append(f"{header}\n{hit.url.strip()}\n{body}".rstrip())
    return "\n\n".join(parts)


def _text(hit: any_search.SearchHit) -> str:
    """The page text a provider gave with a hit, ``""`` when it gave none or only blanks."""
    return (hit.text or "").strip()
