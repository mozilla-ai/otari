"""Live fetches against every production provider, skipped unless ``ANY_FETCH_LIVE=1``.

    ANY_FETCH_LIVE=1 uv run pytest any-fetch/tests/test_live.py

A provider whose key is not in the environment is skipped too. CI runs them
weekly, on demand, and on a pull request that changes either library
(``.github/workflows/otari-libs-live.yml``), so a change on a provider's side
shows up without anyone following its changelog. They check the data as well
as the call: ``EXPECTED`` says what each provider must return, and a
production provider joins it as it joins the conformance scenarios.
"""

import os
from dataclasses import dataclass
from typing import Literal

import pytest

from any_fetch import AnyFetch, ProviderError

live = pytest.mark.skipif(os.environ.get("ANY_FETCH_LIVE") != "1", reason="set ANY_FETCH_LIVE=1 to run")

PROVIDERS = [
    provider
    for provider in AnyFetch.get_supported_providers()
    if provider != "builtin" and AnyFetch.get_provider_metadata(provider).tier == "production"
]
URL = "https://www.python.org/downloads/"
MISSING_URL = "https://www.python.org/no-such-page-any-fetch-live-test"


@dataclass(frozen=True)
class Expected:
    """What a provider's live answers must hold."""

    # It reports what each fetch cost.
    reports_cost: bool
    # It returns the page's title.
    titles: bool
    # What it does with a page the site answered with an error: report it as an
    # error inside the answer, return the text of the site's own error page, or
    # fail the call.
    missing_page: Literal["in_body_error", "site_page", "fails"]


EXPECTED: dict[str, Expected] = {
    # DeepSeek fails the call with a 404 for any page it cannot turn into markdown.
    "deepseek": Expected(reports_cost=False, titles=True, missing_page="fails"),
    "exa": Expected(reports_cost=True, titles=True, missing_page="in_body_error"),
    "tabstack": Expected(reports_cost=False, titles=True, missing_page="site_page"),
}


def _require_key(provider: str) -> None:
    env_key = AnyFetch.get_provider_metadata(provider).env_key
    if env_key and not os.environ.get(env_key):
        pytest.skip(f"{env_key} is not set")


def test_every_production_provider_says_what_it_must_return() -> None:
    assert sorted(EXPECTED) == sorted(PROVIDERS)


@live
@pytest.mark.asyncio
@pytest.mark.parametrize("provider", PROVIDERS)
async def test_a_live_fetch_returns_the_page_text(provider: str) -> None:
    _require_key(provider)
    async with AnyFetch.create(provider) as fetcher:
        page = await fetcher.fetch(URL, max_chars=500)
    assert page.error is None
    assert page.text
    assert len(page.text) <= 500
    assert page.final_url.startswith("https://")
    assert (page.cost is None) == (page.cost_source == "none")
    if EXPECTED[provider].reports_cost:
        assert page.cost_source == "reported"
    if EXPECTED[provider].titles:
        assert page.title


@live
@pytest.mark.asyncio
@pytest.mark.parametrize("provider", PROVIDERS)
async def test_a_live_fetch_of_a_missing_page_is_an_error_or_the_sites_error_page(provider: str) -> None:
    _require_key(provider)
    missing_page = EXPECTED[provider].missing_page
    async with AnyFetch.create(provider) as fetcher:
        if missing_page == "fails":
            with pytest.raises(ProviderError) as raised:
                await fetcher.fetch(MISSING_URL, max_chars=500)
            # DeepSeek's status, the only provider that fails the call so far.
            assert raised.value.status == 404
            return
        page = await fetcher.fetch(MISSING_URL, max_chars=500)
    if missing_page == "in_body_error":
        assert page.error is not None
        assert page.text == ""
    else:
        assert page.error is None
        assert page.text
    if EXPECTED[provider].reports_cost:
        assert page.cost_source == "reported"
