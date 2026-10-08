"""Exa contents: ``POST /contents``, with the key in the ``x-api-key`` header.

The page text is always asked for, capped at ``max_chars``, or at 10,000
characters when the caller passes none, which is what Exa's own Python SDK
asks for. One more character than the cap is requested, so a page longer than
it is cut here and marked ``text_truncated``.

A request-level failure is an HTTP status with a ``tag`` in the body. A page
Exa could not fetch comes back inside a 200, in ``statuses[].error``, and
becomes ``FetchedPage.error``, except ``CRAWL_EMPTY_CONTENT``: the page was
fetched and has no text, which is an empty page, not a failure.

API: https://exa.ai/docs/reference/get-contents. Pricing, read 2026-10-08:
https://exa.ai/docs/admin/pricing.
"""

import re
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Literal

import httpx

from any_fetch._api import AnyFetch
from any_fetch._errors import ProviderError, UnsupportedParameterError
from any_fetch._types import FetchedPage, FetchError, OptionSpec, ProviderMetadata

DEFAULT_MAX_CHARS = 10_000
"""The text cap Exa's Python SDK asks for when its caller names none."""

MAX_CHARACTERS = 1_000_000
"""The most text Exa returns for one page."""

# The page error Exa reports for a page that was fetched and has no text.
EMPTY_CONTENT = "CRAWL_EMPTY_CONTENT"

# Native options copied into the request as they are; ``text`` has rules of its own, below.
_PASSED_THROUGH = ("highlights", "summary", "extras", "maxAgeHours", "livecrawlTimeout", "snapshotAsOf")

# An error tag is kept only when it looks like one, so no text from the body
# reaches an exception.
_TAG = re.compile(r"[A-Za-z0-9_.-]{1,64}")


class ExaProvider(AnyFetch):
    """Exa's contents endpoint."""

    METADATA = ProviderMetadata(
        name="exa",
        doc_url="https://exa.ai/docs/reference/get-contents",
        env_key="EXA_API_KEY",
        env_api_base=None,
        requires_api_key=True,
        requires_api_base=False,
        default_api_base="https://api.exa.ai",
        tier="production",
        max_urls_per_call=100,
        # Exa does not document whether its crawler runs a page's scripts.
        renders_javascript=False,
        formats=["markdown"],
        options=[
            OptionSpec(
                name="text",
                type="object",
                description=(
                    "How the text is extracted: verbosity, includeHtmlTags, includeSections, excludeSections. "
                    "Its length is max_chars, so maxCharacters is refused here."
                ),
            ),
            OptionSpec(name="highlights", type="object", description="Excerpts of the page; kept in raw."),
            OptionSpec(name="summary", type="object", description="A summary of the page; kept in raw."),
            OptionSpec(name="extras", type="object", description="Links and image links from the page; kept in raw."),
            OptionSpec(
                name="maxAgeHours",
                type="integer",
                description="Use a cached copy younger than this; 0 always fetches the page afresh.",
            ),
            OptionSpec(name="livecrawlTimeout", type="integer", description="Milliseconds to wait for a fresh fetch."),
            OptionSpec(
                name="snapshotAsOf",
                type="string",
                description="Return the newest stored copy as of this ISO 8601 date or date-time.",
            ),
        ],
    )

    async def _fetch(self, url: str, *, max_chars: int | None, options: dict[str, Any]) -> FetchedPage:
        text_options = options.get("text") or {}
        if not isinstance(text_options, dict):
            raise ProviderError(self.METADATA.name, None, "invalid_option")
        if "maxCharacters" in text_options:
            raise UnsupportedParameterError(self.METADATA.name, "text.maxCharacters")
        limit = max_chars or DEFAULT_MAX_CHARS
        text_request = {**text_options, "maxCharacters": min(limit + 1, MAX_CHARACTERS)}
        payload: dict[str, Any] = {"urls": [url], "text": text_request}
        payload.update({name: options[name] for name in _PASSED_THROUGH if name in options})

        endpoint = f"{(self._api_base or '').rstrip('/')}/contents"
        response = await self._http.request("POST", endpoint, json=payload, headers={"x-api-key": self._api_key or ""})
        body = _body(response)
        results = body.get("results")
        statuses = body.get("statuses")
        if not isinstance(results, list):
            raise ProviderError(self.METADATA.name, response.status_code, "invalid_response")
        cost = _cost(body)
        cost_source: Literal["reported", "none"] = "none" if cost is None else "reported"

        error = _page_error(statuses)
        if error is not None:
            return FetchedPage(
                url=url,
                final_url=url,
                text="",
                content_type="",
                cost=cost,
                cost_source=cost_source,
                error=error,
                raw=body,
            )
        item = next((result for result in results if isinstance(result, dict)), {})
        text = item.get("text")
        text = text if isinstance(text, str) else ""
        title = item.get("title")
        final_url = item.get("url")
        return FetchedPage(
            url=url,
            final_url=final_url if isinstance(final_url, str) and final_url else url,
            title=title if isinstance(title, str) else "",
            text=text[:limit],
            # Exa returns extracted text and does not say what the page was served as.
            content_type="",
            published=_published(item.get("publishedDate")),
            cost=cost,
            cost_source=cost_source,
            text_truncated=len(text) > limit,
            raw=body,
        )


def _body(response: httpx.Response) -> dict[str, Any]:
    """Return the JSON object Exa answered with, or raise the error it signaled."""
    try:
        body = response.json()
    except ValueError:
        body = None
    if response.status_code >= 400:
        tag = body.get("tag") if isinstance(body, dict) else None
        raise ProviderError(ExaProvider.METADATA.name, response.status_code, _tag(tag) or "http_error")
    if not isinstance(body, dict):
        raise ProviderError(ExaProvider.METADATA.name, response.status_code, "invalid_response")
    return body


def _tag(value: Any) -> str | None:
    return value if isinstance(value, str) and _TAG.fullmatch(value) else None


def _page_error(statuses: Any) -> FetchError | None:
    """The error Exa reported for the one URL asked for, inside a successful response."""
    if not isinstance(statuses, list):
        return None
    for status in statuses:
        if isinstance(status, dict) and status.get("status") == "error":
            error = status.get("error")
            error = error if isinstance(error, dict) else {}
            if error.get("tag") == EMPTY_CONTENT:
                return None
            code = error.get("httpStatusCode")
            return FetchError(
                tag=_tag(error.get("tag")) or "fetch_error",
                status=code if isinstance(code, int) and not isinstance(code, bool) else None,
            )
    return None


def _published(value: Any) -> datetime | None:
    """Exa's ``publishedDate``: a date, or a date-time. A value without a zone is taken as UTC."""
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _cost(body: dict[str, Any]) -> Decimal | None:
    costs = body.get("costDollars")
    total = costs.get("total") if isinstance(costs, dict) else None
    if isinstance(total, bool) or not isinstance(total, int | float):
        return None
    return Decimal(str(total))
