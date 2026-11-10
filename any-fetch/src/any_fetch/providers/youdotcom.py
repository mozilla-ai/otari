"""You.com Extract: ``POST /v1/extract``, with the key in the ``X-API-Key`` header.

You.com's Extract API fetches one web page or PDF per call and returns it as
Markdown, the page's text, with its title when You.com found one. You.com's
Python SDK (``youdotcom`` 3.6.0) has no method for it yet, so the adapter sends
the API's own defaults, except ``parse_effort``: it sends ``lite`` unless told
otherwise, where You.com's default is ``standard``, which bills each PDF page
on top of the URL. ``parse_effort`` is the cost tier. ``full_page`` is sent as
given, with ``markdown`` added to its ``extraction_formats`` when they lack
it, since the text is the Markdown; an HTML copy stays in ``raw``. The other
native options, ``extraction_source`` and ``timeout``, are sent as given, and
an option passed as ``None`` counts as not passed. You.com has no length cap,
so ``max_chars`` cuts the text it returns, and without ``max_chars`` the whole
page is returned.

``final_url`` is the ``url`` You.com answered for, which is the URL as asked
even after a redirect (checked live 2026-11-10), so a caller cannot check
where a redirect You.com followed led. You.com says neither what the page was
served as nor when it was published, so ``content_type`` is empty and
``published`` is ``None``; ``metadata`` (the site's name and favicon) stays in
``raw``. You.com reports no cost.

You.com signals a failure with an HTTP status. A 422, 500, 502, 503 or 504
carries ``error.code``, a code You.com documents as stable (``invalid_url``,
``site_blocked``, ``site_unreachable``, ``unsupported_content``, ``timeout``),
which becomes the tag; its ``message`` and ``details`` are never read. A page
You.com cannot extract fails the call with a 502: a zero-byte file came back
``site_unreachable``. A refused key is a 403 with ``{"message": "Forbidden"}``
from the gateway in front of the API, so 401 and 403 are tagged by status,
``unauthorized`` and ``forbidden``. A 402's ``error`` is a code, either
itself or in its ``code``, since the gateway answers a missing key with a body
of its own, and ``payment_required`` when it is neither. Any other status is
``http_error``. A page whose site answered with an error status, such as a 404
or a 500, comes back as the text of that page, which can be empty, since
You.com does not say what status the site answered with.

You.com documents an answer without ``contents``, seen live as ``{}``, as
nothing having been extracted. It is what ``extraction_source`` ``cache``
answers for a page You.com has no copy of, so it becomes ``FetchedPage.error``,
tagged ``not_cached`` when the request asked for the cache only and
``nothing_extracted`` otherwise.

Every field read from a successful answer is checked, so a change on You.com's
side fails the call rather than degrading it. ``url`` and ``title``, which
You.com documents as optional, must be strings when present; ``contents``,
when present, must be an object whose ``full_page`` is an object whose
``markdown`` is a string, which the adapter always asks for. An empty
``markdown`` is an empty page. Anything else raises ``unexpected_response``.
Fields the adapter does not read are ignored, so a field You.com adds breaks
nothing.

API: https://you.com/docs/api-reference/extract, and the guide on You.com's
preview of its docs, https://ydc-preview-extract-official-launch-docs.docs.buildwithfern.com/docs/guides/extract.
Pricing, read 2026-11-10, from the same preview:
https://ydc-preview-extract-official-launch-docs.docs.buildwithfern.com/docs/administration/billing:
$0.90 per 1,000 URLs; ``parse_effort`` ``standard`` adds $0.50 per 1,000 PDF
pages, so a 350-page PDF costs $0.0009 under ``lite`` and $0.1759 under
``standard``. Volume discounts by arrangement.
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
                    'The formats to return, as {"extraction_formats": ["markdown", "html"]}. Markdown, which '
                    "becomes the text, is always asked for; an HTML copy stays in raw."
                ),
            ),
            OptionSpec(
                name="extraction_source",
                type="string",
                enum=["blend", "cache", "fetch"],
                default="blend",
                description=(
                    "Where the page comes from: blend serves You.com's cached copy when it has one and crawls the "
                    "page live otherwise; cache serves the cached copy only, the fastest, and answers the error "
                    "not_cached for a page it has none of; fetch always crawls the page live, the freshest and the "
                    "slowest."
                ),
            ),
            OptionSpec(
                name="timeout",
                type="integer",
                default=30,
                description=(
                    "Seconds, from 2 to 100, You.com may spend on the whole extraction, its cache lookup and crawl "
                    "included, before failing it with a 504. Keep the fetcher's own timeout above it, which is 15 s "
                    "unless set, or the fetcher gives up first. afetch takes its timeout argument as the fetcher's, "
                    "so pass this one through fetch."
                ),
            ),
            OptionSpec(
                name="parse_effort",
                type="string",
                enum=["lite", "standard"],
                default=DEFAULT_PARSE_EFFORT,
                cost_tier=True,
                description=(
                    "How much work goes into each PDF page; no effect on a web page. lite, sent unless passed, "
                    "extracts the text each page embeds, at the price of the URL alone; standard also reads charts "
                    "and images with a model, and You.com bills each PDF page on top of the URL."
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

    A list without formats is You.com's ``["markdown"]`` already, and a value of the wrong type is
    You.com's to refuse.
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
