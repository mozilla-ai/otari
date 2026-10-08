"""Looking providers up by name, and building them."""

from typing import Any

import pytest

from any_search import AnySearch, UnsupportedProviderError, asearch
from any_search.providers.fake import FakeProvider


def test_supported_providers() -> None:
    assert AnySearch.get_supported_providers() == ["fake"]


def test_provider_class_and_metadata() -> None:
    assert AnySearch.get_provider_class("fake") is FakeProvider
    assert AnySearch.get_provider_metadata("fake") is FakeProvider.METADATA


@pytest.mark.parametrize("lookup", ["create", "get_provider_class", "get_provider_metadata"])
def test_an_unknown_provider_is_refused_with_the_supported_ones(lookup: str) -> None:
    with pytest.raises(UnsupportedProviderError, match="supported: fake"):
        getattr(AnySearch, lookup)("nope")


@pytest.mark.asyncio
async def test_create_builds_the_provider() -> None:
    async with AnySearch.create("fake", timeout=3.0) as fake:
        assert isinstance(fake, FakeProvider)
        assert fake.metadata.name == "fake"


@pytest.mark.asyncio
async def test_asearch_runs_one_search() -> None:
    result = await asearch("fake", "q", max_results=1, account="acme")
    assert result.provider == "fake"
    assert len(result.hits) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(("argument", "value"), [("max_results", 0), ("time_range", "decade")])
async def test_a_shared_parameter_out_of_range_is_refused(argument: str, value: Any) -> None:
    async with AnySearch.create("fake") as fake:
        with pytest.raises(ValueError, match=argument):
            await fake.search("q", **{argument: value})
