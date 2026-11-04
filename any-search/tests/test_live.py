"""Live searches against every production provider, skipped unless ``ANY_SEARCH_LIVE=1``.

    ANY_SEARCH_LIVE=1 uv run pytest any-search/tests/test_live.py

A provider whose key is not in the environment is skipped too. CI runs them
weekly, on demand, and on a pull request that changes either library
(``.github/workflows/otari-libs-live.yml``), so a change on a provider's side
shows up without anyone following its changelog. They check the data as well
as the call: ``EXPECTED`` says what each provider must return, and a
production provider joins it as it joins the conformance scenarios.
"""

import os
from dataclasses import dataclass
from typing import Any

import pytest

from any_search import AnySearch

live = pytest.mark.skipif(os.environ.get("ANY_SEARCH_LIVE") != "1", reason="set ANY_SEARCH_LIVE=1 to run")

PROVIDERS = [
    provider
    for provider in AnySearch.get_supported_providers()
    if AnySearch.get_provider_metadata(provider).tier == "production"
]


@dataclass(frozen=True)
class Expected:
    """What a provider's live answers must hold."""

    # It reports what each search cost.
    reports_cost: bool
    # The native options under which a search returns dated hits; None for a provider that dates none.
    dated: dict[str, Any] | None = None
    # Its hits carry a snippet by default.
    snippets: bool = False


EXPECTED: dict[str, Expected] = {
    # DeepSeek returns page text only on request, so the snippet is what a hit says by default.
    "deepseek": Expected(reports_cost=False, dated={}, snippets=True),
    "exa": Expected(reports_cost=True, dated={"category": "news"}),
    # Mojeek returns no page text, so the snippet is all a hit says; it dates hits by default (date=1).
    "mojeek": Expected(reports_cost=False, dated={}, snippets=True),
    # Serply returns Google's snippet and no page text, and dates no hit.
    "serply": Expected(reports_cost=False, snippets=True),
}


def _require_key(provider: str) -> None:
    env_key = AnySearch.get_provider_metadata(provider).env_key
    if env_key and not os.environ.get(env_key):
        pytest.skip(f"{env_key} is not set")


def test_every_production_provider_says_what_it_must_return() -> None:
    assert sorted(EXPECTED) == sorted(PROVIDERS)


@live
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
    if EXPECTED[provider].reports_cost:
        assert result.cost_source == "reported"
    if EXPECTED[provider].snippets:
        assert any(hit.snippet for hit in result.hits)


@live
@pytest.mark.asyncio
@pytest.mark.parametrize("provider", PROVIDERS)
async def test_a_live_search_takes_a_time_range(provider: str) -> None:
    _require_key(provider)
    async with AnySearch.create(provider) as engine:
        result = await engine.search("python release", max_results=2, time_range="year")
    assert result.error is None


@live
@pytest.mark.asyncio
@pytest.mark.parametrize("provider", PROVIDERS)
async def test_a_live_search_dates_its_hits(provider: str) -> None:
    dated = EXPECTED[provider].dated
    if dated is None:
        pytest.skip(f"{provider} dates no hits")
    _require_key(provider)
    async with AnySearch.create(provider) as engine:
        # A query for news itself, so dated hits come back whatever the week's news is.
        result = await engine.search("latest technology news", max_results=10, **dated)
    assert result.error is None
    assert any(hit.published is not None for hit in result.hits)
