"""The fake provider: canned answers, with its behavior taken from native options.

It is a provider like any other, in the ``test`` tier, so the library's own
tests and a host's can drive every path a real provider takes without a
network or a monkeypatch.
"""

import asyncio
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ConfigDict

from any_search._api import AnySearch
from any_search._errors import ProviderError
from any_search._types import OptionSpec, ProviderMetadata, SearchError, SearchHit, SearchResult, TimeRange

CANNED_HITS: tuple[dict[str, Any], ...] = (
    {
        "url": "https://example.com/fake/1",
        "title": "Fake result one",
        "snippet": "The first canned result.",
        "published": "2026-01-15T09:30:00Z",
    },
    {
        "url": "https://example.org/fake/2",
        "title": "Fake result two",
        "snippet": "The second canned result.",
        "text": "The page text of the second canned result.",
    },
    {
        "url": "https://example.net/fake/3",
        "title": "Fake result three",
        "snippet": "The third canned result.",
    },
)


class FakeOptions(BaseModel):
    """The native options, parsed. Their names match the metadata's option list."""

    model_config = ConfigDict(extra="forbid")

    hits: list[dict[str, Any]] | None = None
    cost: Decimal | None = None
    delay: float = 0.0
    error: str | None = None
    error_status: int | None = None
    in_body_error: str | None = None
    leak_query: bool = False
    account: str | None = None


class FakeProvider(AnySearch):
    """Answers from canned data; see the option descriptions for what each one does."""

    METADATA = ProviderMetadata(
        name="fake",
        doc_url="https://github.com/mozilla-ai/otari/tree/main/any-search#the-fake-provider",
        env_key=None,
        env_api_base=None,
        requires_api_key=False,
        requires_api_base=False,
        default_api_base=None,
        tier="test",
        max_results=10,
        query_in_url=False,
        key_in_url=False,
        options=[
            OptionSpec(
                name="hits",
                type="array",
                description=(
                    "The hits to answer with, each an object with url and optionally title, snippet, text and "
                    "published. An item without url is dropped. Three canned hits when unset."
                ),
            ),
            OptionSpec(name="cost", type="number", description="A cost in USD to report."),
            OptionSpec(name="delay", type="number", default=0.0, description="Seconds to wait before answering."),
            OptionSpec(name="error", type="string", description="Fail the call with a ProviderError of this tag."),
            OptionSpec(name="error_status", type="integer", description="The HTTP status of either error."),
            OptionSpec(
                name="in_body_error",
                type="string",
                description="Answer with no hits and this tag as an error signaled inside a successful response.",
            ),
            OptionSpec(
                name="leak_query",
                type="boolean",
                default=False,
                description=(
                    "Raise a RuntimeError whose message carries the query, as a careless adapter might, so a "
                    "host can prove it never passes exception text on."
                ),
            ),
            OptionSpec(
                name="account",
                type="string",
                operator_only=True,
                description="An account name, echoed in raw. Operator-only, so a host can exercise that rule.",
            ),
        ],
    )

    async def _search(
        self,
        query: str,
        *,
        max_results: int | None,
        time_range: TimeRange | None,
        options: dict[str, Any],
    ) -> SearchResult:
        parsed = FakeOptions.model_validate(options)
        if parsed.delay > 0:
            await asyncio.sleep(parsed.delay)
        if parsed.leak_query:
            raise RuntimeError(f"fake search failed for {query!r}")
        if parsed.error is not None:
            raise ProviderError(self.METADATA.name, parsed.error_status, parsed.error)
        # The request is echoed, minus the query, so a host's test can see what reached the provider.
        request = {"max_results": max_results, "time_range": time_range, "options": options}
        if parsed.in_body_error is not None:
            return SearchResult(
                provider=self.METADATA.name,
                hits=[],
                cost_source="none",
                error=SearchError(tag=parsed.in_body_error, status=parsed.error_status),
                raw={"request": request, "results": []},
            )
        items = [dict(item) for item in (CANNED_HITS if parsed.hits is None else parsed.hits)]
        hits = [
            SearchHit(
                url=item["url"],
                title=item.get("title") or "",
                snippet=item.get("snippet") or "",
                text=item.get("text"),
                published=item.get("published"),
                raw=item,
            )
            for item in items
            if item.get("url")
        ]
        limit = min(max_results or self.METADATA.max_results, self.METADATA.max_results)
        return SearchResult(
            provider=self.METADATA.name,
            hits=hits[:limit],
            cost=parsed.cost,
            cost_source="none" if parsed.cost is None else "reported",
            raw={"request": request, "results": items},
        )
