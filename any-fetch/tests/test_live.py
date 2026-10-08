"""Live fetches against every production provider, skipped unless ``ANY_FETCH_LIVE=1``.

    ANY_FETCH_LIVE=1 uv run pytest any-fetch/tests/test_live.py

A provider whose key is not in the environment is skipped too. CI never sets
the variable; run these once by hand when a provider is added or changed.
"""

import os

import pytest

from any_fetch import AnyFetch

pytestmark = pytest.mark.skipif(os.environ.get("ANY_FETCH_LIVE") != "1", reason="set ANY_FETCH_LIVE=1 to run")

PROVIDERS = [
    provider
    for provider in AnyFetch.get_supported_providers()
    if provider != "builtin" and AnyFetch.get_provider_metadata(provider).tier == "production"
]
URL = "https://www.python.org/downloads/"


def _require_key(provider: str) -> None:
    env_key = AnyFetch.get_provider_metadata(provider).env_key
    if env_key and not os.environ.get(env_key):
        pytest.skip(f"{env_key} is not set")


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


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", PROVIDERS)
async def test_a_live_fetch_of_a_missing_page_is_an_error_inside_the_answer(provider: str) -> None:
    _require_key(provider)
    async with AnyFetch.create(provider) as fetcher:
        page = await fetcher.fetch("https://www.python.org/no-such-page-any-fetch-live-test", max_chars=500)
    assert page.error is not None
    assert page.text == ""
