"""Live searches against every production provider, skipped unless ``ANY_SEARCH_LIVE=1``.

    ANY_SEARCH_LIVE=1 uv run pytest any-search/tests/test_live.py

A provider whose key is not in the environment is skipped too. CI never sets
the variable; run these once by hand when a provider is added or changed.
"""

import os

import pytest

from any_search import AnySearch

pytestmark = pytest.mark.skipif(os.environ.get("ANY_SEARCH_LIVE") != "1", reason="set ANY_SEARCH_LIVE=1 to run")

PROVIDERS = [
    provider
    for provider in AnySearch.get_supported_providers()
    if AnySearch.get_provider_metadata(provider).tier == "production"
]


def _require_key(provider: str) -> None:
    env_key = AnySearch.get_provider_metadata(provider).env_key
    if env_key and not os.environ.get(env_key):
        pytest.skip(f"{env_key} is not set")


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", PROVIDERS)
async def test_a_live_search_returns_hits(provider: str) -> None:
    _require_key(provider)
    async with AnySearch.create(provider) as engine:
        result = await engine.search("latest stable python release", max_results=3)
    assert result.error is None
    assert 1 <= len(result.hits) <= 3
    assert all(hit.url.startswith(("http://", "https://")) for hit in result.hits)
    assert (result.cost is None) == (result.cost_source == "none")


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", PROVIDERS)
async def test_a_live_search_takes_a_time_range(provider: str) -> None:
    _require_key(provider)
    async with AnySearch.create(provider) as engine:
        result = await engine.search("python release", max_results=2, time_range="year")
    assert result.error is None
