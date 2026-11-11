"""You.com Extract: ``POST /v1/extract``, with the key in the ``X-API-Key`` header.

One web page or PDF per call, as Markdown. ``parse_effort`` is ``lite`` unless
passed: You.com's own default, ``standard``, bills each PDF page on top of the
URL. ``full_page`` is sent as given, with ``markdown`` added when its formats
lack it. ``extraction_source`` and ``timeout`` are sent as given; ``None``
counts as not passed. You.com has no length cap, so ``max_chars`` cuts the text.

``final_url`` is the URL as asked, even after a redirect. You.com reports
neither a content type, a publish date, nor a cost.

401 and 403 are tagged by status (``unauthorized``, ``forbidden``): those
bodies are the gateway's. Any other status is tagged with ``error.code`` when it
carries one (``invalid_url``, ``site_blocked``, ``site_unreachable``,
``unsupported_content``, ``timeout``); the message and details are not read. A
402 without one is tagged from its flat ``error``, else ``payment_required``.
Anything else is ``http_error``. A site's own error page comes back as that
page's text.

A body without ``contents`` is ``FetchedPage.error``: ``not_cached`` when the
request asked for the cache only, else ``nothing_extracted``.

``url`` and ``title`` must be strings when present. ``contents``, when present,
must hold ``full_page.markdown`` as a string; an empty string is an empty page.
Anything else raises ``unexpected_response``. Unread fields are ignored.

API: https://you.com/docs/api-reference/extract. Pricing, read 2026-11-10,
https://you.com/docs/administration/billing:
$0.90 per 1,000 URLs; ``standard`` adds $0.50 per 1,000 PDF pages.
"""

from typing import Any

import httpx

from any_fetch._api import AnyFetch
from any_fetch._errors import ProviderError, safe_tag
from any_fetch._types import FetchedPage, FetchError, OptionSpec, ProviderMetadata

DEFAULT_PARSE_EFFORT = "lite"
"""The PDF parse effort sent when none is passed: the tier that bills no PDF page."""

# Native options copied into the request as they are; full_page and parse_effort have rules of their own.
_PASSED_THROUGH = ("extraction_source", "timeout")

# The statuses whose body is the gateway's, not an error code.
_STATUS_TAGS = {401: "unauthorized", 403: "forbidden"}


class YouDotComProvider(AnyFetch):
    """You.com's Extract API."""

    METADATA = ProviderMetadata(
        name="youdotcom",
        doc_url="https://you.com/docs/api-reference/extract",
        env_key="YDC_API_KEY",
        env_api_base=None,
        requires_api_key=True,
        requires_api_base=False,
        default_api_base="https://ydc-index.io",
        tier="production",
        max_urls_per_call=1,
        # You.com does not document running a page's scripts, and answered a page that needs them with its fallback.
        renders_javascript=False,
        # You.com answers with the URL as asked, even after a redirect (checked 2026-11-10).
        reports_final_url=False,
        formats=["markdown"],
        options=[
            OptionSpec(
                name="full_page",
                type="object",
                description=(
                    'Formats to return, as {"extraction_formats": ["markdown", "html"]}. Markdown is always requested. '
                    "HTML version, when additionally requested, is returned in raw output."
                ),
            ),
            OptionSpec(
                name="extraction_source",
                type="string",
                enum=["blend", "cache", "fetch"],
                default="blend",
                description=(
                    "blend uses a cached copy when You.com has one, else crawls live; cache never crawls and errors "
                    "not_cached when it has none; fetch always crawls live."
                ),
            ),
            OptionSpec(
                name="timeout",
                type="integer",
                default=30,
                description=(
                    "Seconds You.com may spend, 2 to 100, before a 504. Pass the fetcher a longer timeout: afetch's "
                    "timeout argument is the fetcher's, so this option goes through fetch."
                ),
            ),
            OptionSpec(
                name="parse_effort",
                type="string",
                enum=["lite", "standard"],
                default=DEFAULT_PARSE_EFFORT,
                cost_tier=True,
                description=(
                    "Parse effort for PDFs; no effect on web pages. lite, sent unless passed, reads the text "
                    "embedded in each page and adds no charge. standard, You.com's own default, handles complex "
                    "pages, such as charts, tables and images, through an advanced model, and adds a charge per "
                    "page in the PDF file."
                ),
            ),
        ],
    )

    async def _fetch(self, url: str, *, max_chars: int | None, options: dict[str, Any]) -> FetchedPage:
        payload: dict[str, Any] = {"url": url, "parse_effort": options.get("parse_effort") or DEFAULT_PARSE_EFFORT}
        payload.update({name: options[name] for name in _PASSED_THROUGH if options.get(name) is not None})
        full_page = options.get("full_page")
        if full_page is not None:
            payload["full_page"] = _with_markdown(full_page, self.METADATA.name)

        endpoint = f"{(self._api_base or '').rstrip('/')}/v1/extract"
        response = await self._http.request("POST", endpoint, json=payload, headers={"X-API-Key": self._api_key or ""})
        body = _body(response)
        final_url, title = body.get("url", url), body.get("title", "")
        if not isinstance(final_url, str) or not isinstance(title, str):
            raise _unexpected(response)
        if "contents" not in body:
            tag = "not_cached" if payload.get("extraction_source") == "cache" else "nothing_extracted"
            return FetchedPage(
                url=url,
                final_url=final_url or url,
                title=title,
                text="",
                content_type="",
                cost_source="none",
                error=FetchError(tag=tag),
                raw=body,
            )
        text = _markdown(body["contents"], response)
        return FetchedPage(
            url=url,
            final_url=final_url or url,
            title=title,
            text=text[:max_chars],
            # You.com returns extracted text and does not say what the page was served as.
            content_type="",
            cost_source="none",
            text_truncated=max_chars is not None and len(text) > max_chars,
            raw=body,
        )


def _with_markdown(full_page: Any, provider: str) -> dict[str, Any]:
    """``full_page`` as given, with ``markdown`` among its formats, since the text is the Markdown.

    A list without formats is You.com's ``["markdown"]`` already. Incorrect values handled accordingly by You.com
    """
    if not isinstance(full_page, dict):
        raise ProviderError(provider, None, "invalid_option")
    formats = full_page.get("extraction_formats")
    if isinstance(formats, list) and formats and "markdown" not in formats:
        return {**full_page, "extraction_formats": ["markdown", *formats]}
    return dict(full_page)


def _body(response: httpx.Response) -> dict[str, Any]:
    """Return the JSON object You.com answered with, or raise the error it signaled.

    Only a code reaches the error: You.com's ``message`` can change wording and its ``details`` can
    name the URL's host.
    """
    try:
        body = response.json()
    except ValueError:
        body = None
    status = response.status_code
    if status >= 400:
        error = body.get("error") if isinstance(body, dict) else None
        tag = _STATUS_TAGS.get(status)
        if tag is None and isinstance(error, dict):
            tag = safe_tag(error.get("code"))
        if status == 402:
            # The gateway in front of the API answers with a flat ``error`` of its own.
            tag = tag or safe_tag(error) or "payment_required"
        raise ProviderError(YouDotComProvider.METADATA.name, status, tag or "http_error")
    if not isinstance(body, dict):
        raise ProviderError(YouDotComProvider.METADATA.name, status, "invalid_response")
    return body


def _unexpected(response: httpx.Response) -> ProviderError:
    """A successful answer that lacks a field the adapter needs, or carries one of the wrong type."""
    return ProviderError(YouDotComProvider.METADATA.name, response.status_code, "unexpected_response")


def _markdown(contents: Any, response: httpx.Response) -> str:
    """The page's Markdown from ``contents.full_page``, which the adapter always asks for."""
    full_page = contents.get("full_page") if isinstance(contents, dict) else None
    markdown = full_page.get("markdown") if isinstance(full_page, dict) else None
    if not isinstance(markdown, str):
        raise _unexpected(response)
    return markdown
