"""DeepSeek: ``POST /v1/tools/fetch``, with the key in a bearer header.

DeepSeek's Fetch API fetches one page per call and returns its title and its
text as markdown. It takes the URL and nothing else, so there are no native
options. It has no length cap, so ``max_chars`` cuts the text it returns, and
without ``max_chars`` the whole page is returned. It says neither where the
page was found nor what it was served as, so ``final_url`` is the URL asked
for and ``content_type`` is empty.

DeepSeek signals every failure with an HTTP status and an OpenAI-shaped body.
Its ``error.type`` becomes the tag, and ``error.code`` when there is no type,
since DeepSeek sends ``invalid_request_error`` as the code of every failure.
The message is never read: it carries the key's last four characters for a
refused key. A page DeepSeek cannot turn into markdown fails the call with a
404 tagged ``not_found_error``, and nothing in the answer says why. Seen live
for a page that does not exist, a site's 500, a private address, a PDF, a
zero-byte file and HTML with an empty body, so DeepSeek never answers a page
with no text, and a missing page is never the site's own error page. A
successful answer carries no error of its own. It follows a redirect, but does
not say where it ended.

Every field read from a successful answer is checked, so a change on
DeepSeek's side fails the call rather than degrading it: ``title`` and
``content``, which DeepSeek documents as always present, must be strings.
Anything else raises ``unexpected_response``. DeepSeek reports no cost and no
date. Fields the adapter does not read are ignored, so a field DeepSeek adds
breaks nothing.

API: https://api-docs.deepseek.com/api/fetch and
https://api-docs.deepseek.com/guides/search. Pricing, read 2026-10-19: $0.75
per 1,000 successful fetches; 25 fetches a second per account, apart from
searches, and a per-domain limit DeepSeek does not state. DeepSeek's free
period covers only calls made through its Harness, Claude Code or Codex, not
this adapter's.
"""

from typing import Any

import httpx

from any_fetch._api import AnyFetch
from any_fetch._errors import ProviderError, safe_tag
from any_fetch._types import FetchedPage, ProviderMetadata


class DeepSeekProvider(AnyFetch):
    """DeepSeek's Fetch API."""

    METADATA = ProviderMetadata(
        name="deepseek",
        doc_url="https://api-docs.deepseek.com/api/fetch",
        env_key="DEEPSEEK_API_KEY",
        env_api_base=None,
        requires_api_key=True,
        requires_api_base=False,
        default_api_base="https://api.deepseek.com",
        tier="production",
        max_urls_per_call=1,
        renders_javascript=False,
        # DeepSeek follows a redirect without saying where it led (seen live).
        reports_final_url=False,
        formats=["markdown"],
        options=[],
    )

    async def _fetch(self, url: str, *, max_chars: int | None, options: dict[str, Any]) -> FetchedPage:
        endpoint = f"{(self._api_base or '').rstrip('/')}/v1/tools/fetch"
        response = await self._http.request(
            "POST", endpoint, json={"url": url}, headers={"Authorization": f"Bearer {self._api_key or ''}"}
        )
        body = _body(response)
        title, text = body.get("title"), body.get("content")
        if not isinstance(title, str) or not isinstance(text, str):
            raise ProviderError(DeepSeekProvider.METADATA.name, response.status_code, "unexpected_response")
        return FetchedPage(
            url=url,
            final_url=url,
            title=title,
            text=text[:max_chars],
            content_type="",
            cost_source="none",
            text_truncated=max_chars is not None and len(text) > max_chars,
            raw=body,
        )


def _body(response: httpx.Response) -> dict[str, Any]:
    """Return the JSON object DeepSeek answered with, or raise the error it signaled."""
    try:
        body = response.json()
    except ValueError:
        body = None
    if response.status_code >= 400:
        error = body.get("error") if isinstance(body, dict) else None
        tag = (safe_tag(error.get("type")) or safe_tag(error.get("code"))) if isinstance(error, dict) else None
        raise ProviderError(DeepSeekProvider.METADATA.name, response.status_code, tag or "http_error")
    if not isinstance(body, dict):
        raise ProviderError(DeepSeekProvider.METADATA.name, response.status_code, "invalid_response")
    return body
