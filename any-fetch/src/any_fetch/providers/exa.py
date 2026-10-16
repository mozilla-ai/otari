"""Exa contents: ``POST /contents``, with the key in the ``x-api-key`` header.

The page text is always asked for, capped at ``max_chars``, or at 10,000
characters when the caller passes none, which is what Exa's own Python SDK
asks for. One more character than the cap is requested, so a page longer than
it is cut here and marked ``text_truncated``.

A request-level failure is an HTTP status with a ``tag`` in the body. A page
Exa could not fetch comes back inside a 200, in ``statuses[].error``, and
becomes ``FetchedPage.error``, except ``CRAWL_EMPTY_CONTENT``: the page was
fetched and has no text, which is an empty page, not a failure.

Every field read from a successful answer is checked, so a change on Exa's
side fails the call rather than degrading it: ``results`` must be there, and
``statuses``, which Exa documents as optional, needs an entry for the URL,
``success`` or ``error``, when it has any (the only entry when its id is not
the URL, which Exa may have normalized); a success needs its page's ``url`` and
``title``; and a ``text`` must be a string, a
``publishedDate`` parse and a ``costDollars.total`` be a number when there is
one. A page without text, which Exa documents as optional, is an empty page. Anything else raises
``unexpected_response``, an answer with no page and no reason included. Exa
documents the cost as optional, so a fetch without one reports none, and the
live tests say so. Fields the adapter does not read are ignored, so a field Exa
adds breaks nothing.

API: https://exa.ai/docs/reference/get-contents. Pricing, read 2026-10-08:
https://exa.ai/docs/admin/pricing.
"""

import math
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
        # The text is always asked for, so true means Exa's defaults, and a value that is not an object is refused.
        text_options = options.get("text")
        if text_options is None or text_options is True:
            text_options = {}
        if not isinstance(text_options, dict):
            raise ProviderError(self.METADATA.name, None, "invalid_option")
        if "maxCharacters" in text_options:
            raise UnsupportedParameterError(self.METADATA.name, "text.maxCharacters")
        limit = max_chars or DEFAULT_MAX_CHARS
        requested = min(limit + 1, MAX_CHARACTERS)
        payload: dict[str, Any] = {"urls": [url], "text": {**text_options, "maxCharacters": requested}}
        payload.update({name: options[name] for name in _PASSED_THROUGH if name in options})

        endpoint = f"{(self._api_base or '').rstrip('/')}/contents"
        response = await self._http.request("POST", endpoint, json=payload, headers={"x-api-key": self._api_key or ""})
        body = _body(response)
        results = body.get("results")
        # Exa documents statuses as optional; without them the page in results is the answer.
        statuses = body.get("statuses", [])
        if not isinstance(results, list) or not isinstance(statuses, list):
            raise _unexpected(response)
        cost = _cost(body, response)
        cost_source: Literal["reported", "none"] = "none" if cost is None else "reported"

        status = _status_of(url, statuses, response) if statuses else None
        if status is not None and status["status"] == "error":
            # None for a page fetched with no text, which is an empty page rather than a failure.
            return FetchedPage(
                url=url,
                final_url=url,
                text="",
                content_type="",
                cost=cost,
                cost_source=cost_source,
                error=_page_error(status, response),
                raw=body,
            )
        final_url, title, text, published = _page(results, response)
        return FetchedPage(
            url=url,
            final_url=final_url,
            title=title,
            text=text[:limit],
            # Exa returns extracted text and does not say what the page was served as.
            content_type="",
            published=published,
            cost=cost,
            cost_source=cost_source,
            # At Exa's own limit there is no character to spare, so text that fills it counts as cut.
            text_truncated=len(text) >= requested,
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
    """Exa's error tag when it looks like a code, else ``None``, so no body text reaches an error."""
    return value if isinstance(value, str) and _TAG.fullmatch(value) else None


def _unexpected(response: httpx.Response) -> ProviderError:
    """A successful answer that lacks a field Exa documents, or carries one of the wrong type."""
    return ProviderError(ExaProvider.METADATA.name, response.status_code, "unexpected_response")


def _status_of(url: str, statuses: list[Any], response: httpx.Response) -> dict[str, Any]:
    """Exa's status for the one URL asked for: the entry with its id, else the only one Exa sent.

    Exa may answer for the URL as it normalized it, so one entry under another
    id still counts; with several and none for the URL, nothing says which is ours.
    """
    checked: list[dict[str, Any]] = []
    for status in statuses:
        if not isinstance(status, dict) or not isinstance(status.get("id"), str):
            raise _unexpected(response)
        if status.get("status") not in ("success", "error"):
            raise _unexpected(response)
        checked.append(status)
    for status in checked:
        if status["id"] == url:
            return status
    if len(checked) != 1:
        raise _unexpected(response)
    return checked[0]


def _page_error(status: dict[str, Any], response: httpx.Response) -> FetchError | None:
    """The error Exa reported for the URL inside a successful response; None for a page with no text."""
    error = status.get("error")
    if error is None:
        error = {}
    if not isinstance(error, dict):
        raise _unexpected(response)
    tag, code = error.get("tag"), error.get("httpStatusCode")
    if tag is not None and not isinstance(tag, str):
        raise _unexpected(response)
    if code is not None and (not isinstance(code, int) or isinstance(code, bool)):
        raise _unexpected(response)
    if tag == EMPTY_CONTENT:
        return None
    return FetchError(tag=_tag(tag) or "fetch_error", status=code)


def _page(results: list[Any], response: httpx.Response) -> tuple[str, str, str, datetime | None]:
    """The page Exa fetched: its url, title, text and date, each checked.

    Exa documents ``url`` and ``title`` on every result and ``text`` as
    optional, so a page without text is an empty one.
    """
    item = results[0] if results else None
    # A success with no page is an answer Exa does not document.
    if not isinstance(item, dict):
        raise _unexpected(response)
    url, title, text = item.get("url"), item.get("title"), item.get("text", "")
    if not isinstance(url, str) or not url or not isinstance(title, str) or not isinstance(text, str):
        raise _unexpected(response)
    published = None
    if "publishedDate" in item:
        published = _published(item["publishedDate"])
        if published is None:
            raise _unexpected(response)
    return url, title, text, published


def _published(value: Any) -> datetime | None:
    """Exa's ``publishedDate``: a date, or a date-time. A value without a zone is taken as UTC."""
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _cost(body: dict[str, Any], response: httpx.Response) -> Decimal | None:
    """Exa's ``costDollars.total``, or ``None`` when it sent none."""
    costs = body.get("costDollars")
    if costs is None:
        return None
    if not isinstance(costs, dict):
        raise _unexpected(response)
    total = costs.get("total")
    if total is None:
        return None
    if isinstance(total, bool) or not isinstance(total, int | float) or not 0 <= total < math.inf:
        raise _unexpected(response)
    return Decimal(str(total))
