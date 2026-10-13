"""Tabstack extract: ``POST /extract/markdown``, with the key in a bearer header.

Tabstack, Mozilla's web API for agents, fetches one page per call and returns
its text as markdown. The adapter always asks for the page's metadata as a
field of its own (``metadata: true``), which is where the title comes from;
without it Tabstack puts the metadata in the text as YAML front matter, which
its own documentation advises against parsing. ``metadata`` is therefore not
an option. Tabstack has no length cap, so ``max_chars`` cuts the text it
returns, and without ``max_chars`` the whole page is returned, as Tabstack's
own SDK returns it.

A failed call is an HTTP status. Its body is a sentence (``{"error": "..."}``,
or plain text for a refused key) that can quote the URL, so it is never read:
the tag is ``http_error``. Tabstack reports no error inside a successful
answer. A page it cannot fetch fails the call, and a page its site answered
with an error status, such as a 404, comes back as that page's text, since
Tabstack does not say what status the site answered with.

Every field read from a successful answer is checked, so a change on
Tabstack's side fails the call rather than degrading it: ``content`` must be a
string and ``url`` a non-empty one, both of which Tabstack documents as always
present, and ``metadata`` and its ``title``, which it documents as optional,
must be an object and a string when they are there. Anything else raises
``unexpected_response``. Tabstack reports no cost, and no date the adapter can
rely on: its ``created_at`` is documented as ISO 8601 but carries a PDF's own
date format (``D:20240410211143Z``), so it is left in ``raw``. Fields the
adapter does not read are ignored, so a field Tabstack adds breaks nothing.

API: https://docs.tabstack.ai/api/resources/extract/methods/markdown. Pricing,
read 2026-10-13: https://docs.tabstack.ai/pricing, 10 credits per call.
"""

from typing import Any

import httpx

from any_fetch._api import AnyFetch
from any_fetch._errors import ProviderError
from any_fetch._types import FetchedPage, OptionSpec, ProviderMetadata


class TabstackProvider(AnyFetch):
    """Tabstack's markdown extraction endpoint."""

    METADATA = ProviderMetadata(
        name="tabstack",
        doc_url="https://docs.tabstack.ai/api/resources/extract/methods/markdown",
        env_key="TABSTACK_API_KEY",
        # The variable Tabstack's own SDKs read.
        env_api_base="TABSTACK_BASE_URL",
        requires_api_key=True,
        requires_api_base=False,
        default_api_base="https://api.tabstack.ai/v1",
        tier="production",
        max_urls_per_call=1,
        # With effort "max", which renders the page in a headless browser.
        renders_javascript=True,
        formats=["markdown"],
        options=[
            OptionSpec(
                name="content",
                type="string",
                enum=["main", "full"],
                default="main",
                description="main is the page's main content; full is the whole page, navigation and footer included.",
            ),
            OptionSpec(
                name="effort",
                type="string",
                enum=["min", "standard", "max"],
                default="standard",
                description=(
                    "How hard Tabstack works to fetch the page: min fetches the HTML with no fallback (1 to 5 s), "
                    "standard adds light JavaScript handling (3 to 15 s), and max renders the page in a browser "
                    "(15 to 60 s), so raise the timeout to match."
                ),
            ),
            OptionSpec(
                name="geo_target",
                type="object",
                description='Fetch through a proxy in a country, as {"country": "GB"} (ISO 3166-1 alpha-2).',
            ),
            OptionSpec(name="nocache", type="boolean", description="Fetch the page afresh rather than from the cache."),
        ],
    )

    async def _fetch(self, url: str, *, max_chars: int | None, options: dict[str, Any]) -> FetchedPage:
        # Every native option is sent as it is, and Tabstack refuses a value it does not take.
        payload: dict[str, Any] = {"url": url, "metadata": True, **options}

        endpoint = f"{(self._api_base or '').rstrip('/')}/extract/markdown"
        response = await self._http.request(
            "POST", endpoint, json=payload, headers={"Authorization": f"Bearer {self._api_key or ''}"}
        )
        body = _body(response)
        text, final_url = body.get("content"), body.get("url")
        if not isinstance(text, str) or not isinstance(final_url, str) or not final_url:
            raise _unexpected(response)
        return FetchedPage(
            url=url,
            final_url=final_url,
            title=_title(body, response),
            text=text[:max_chars],
            # Tabstack returns extracted text and does not say what the page was served as.
            content_type="",
            cost_source="none",
            text_truncated=max_chars is not None and len(text) > max_chars,
            raw=body,
        )


def _body(response: httpx.Response) -> dict[str, Any]:
    """Return the JSON object Tabstack answered with, or raise the error it signaled.

    An error body is a sentence that can quote the URL, so only the status is kept.
    """
    if response.status_code >= 400:
        raise ProviderError(TabstackProvider.METADATA.name, response.status_code, "http_error")
    try:
        body = response.json()
    except ValueError:
        body = None
    if not isinstance(body, dict):
        raise ProviderError(TabstackProvider.METADATA.name, response.status_code, "invalid_response")
    return body


def _unexpected(response: httpx.Response) -> ProviderError:
    """A successful answer that lacks a field Tabstack documents, or carries one of the wrong type."""
    return ProviderError(TabstackProvider.METADATA.name, response.status_code, "unexpected_response")


def _title(body: dict[str, Any], response: httpx.Response) -> str:
    """The page's title from ``metadata``, ``""`` when Tabstack found none; both are optional."""
    if "metadata" not in body:
        return ""
    metadata = body["metadata"]
    if not isinstance(metadata, dict):
        raise _unexpected(response)
    title = metadata.get("title", "")
    if not isinstance(title, str):
        raise _unexpected(response)
    return title
