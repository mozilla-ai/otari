"""Looking providers up by name, building them, and the host's builtin."""

from collections.abc import Iterator
from typing import Any

import pytest

from any_fetch import (
    AnyFetch,
    BuiltinNotRegisteredError,
    FetchedPage,
    UnsupportedParameterError,
    UnsupportedProviderError,
    afetch,
)
from any_fetch._http import ProviderHttp
from any_fetch.providers import builtin
from any_fetch.providers.builtin import BuiltinProvider
from any_fetch.providers.fake import FakeProvider


class _HostFetcher:
    """Stands in for the fetcher a host's factory builds."""

    def __init__(self, **settings: Any) -> None:
        self.settings = settings
        self.calls: list[tuple[str, int | None]] = []
        self.closed = False

    async def fetch(self, url: str, *, max_chars: int | None) -> FetchedPage:
        self.calls.append((url, max_chars))
        return FetchedPage(
            url=url, final_url=url, text="host text", content_type="text/html", cost_source="none", raw={}
        )

    async def aclose(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def _no_builtin(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    # Each test starts unregistered, and whatever it registers is undone after it.
    monkeypatch.setattr(builtin, "_factory", None)
    yield


def test_supported_providers() -> None:
    assert AnyFetch.get_supported_providers() == ["builtin", "fake"]


def test_provider_class_and_metadata() -> None:
    assert AnyFetch.get_provider_class("fake") is FakeProvider
    assert AnyFetch.get_provider_class("builtin") is BuiltinProvider
    assert AnyFetch.get_provider_metadata("builtin").tier == "production"


@pytest.mark.parametrize("lookup", ["create", "get_provider_class", "get_provider_metadata"])
def test_an_unknown_provider_is_refused_with_the_supported_ones(lookup: str) -> None:
    with pytest.raises(UnsupportedProviderError, match="supported: builtin, fake"):
        getattr(AnyFetch, lookup)("nope")


@pytest.mark.asyncio
async def test_afetch_fetches_one_page() -> None:
    page = await afetch("fake", "https://example.com/", max_chars=3, account="acme")
    assert (page.text, page.text_truncated) == ("The", True)


def test_builtin_before_registration_is_a_clear_error() -> None:
    with pytest.raises(BuiltinNotRegisteredError, match="register_builtin"):
        AnyFetch.create("builtin")


@pytest.mark.asyncio
async def test_builtin_passes_its_arguments_to_the_factory_and_wraps_what_it_builds() -> None:
    built: list[_HostFetcher] = []

    def factory(**settings: Any) -> _HostFetcher:
        built.append(_HostFetcher(**settings))
        return built[-1]

    AnyFetch.register_builtin(factory)
    client = object()
    async with AnyFetch.create("builtin", client=client, allow_private_hosts=False) as fetcher:
        assert isinstance(fetcher, BuiltinProvider)
        page = await fetcher.fetch("https://example.com/", max_chars=100)
    [host] = built
    assert host.settings == {"client": client, "allow_private_hosts": False}
    assert host.calls == [("https://example.com/", 100)]
    assert page.text == "host text"
    assert host.closed


@pytest.mark.asyncio
async def test_builtin_takes_no_native_option() -> None:
    AnyFetch.register_builtin(_HostFetcher)
    async with AnyFetch.create("builtin") as fetcher:
        with pytest.raises(UnsupportedParameterError):
            await fetcher.fetch("https://example.com/", render=True)


def test_registering_again_replaces_the_factory() -> None:
    first, second = _HostFetcher(), _HostFetcher()
    AnyFetch.register_builtin(lambda **settings: first)
    AnyFetch.register_builtin(lambda **settings: second)
    fetcher = AnyFetch.create("builtin")
    assert isinstance(fetcher, BuiltinProvider)
    assert fetcher._fetcher is second


class _FailingCloseFetcher(_HostFetcher):
    async def aclose(self) -> None:
        raise RuntimeError("host fetcher failed to close")


@pytest.mark.asyncio
async def test_builtin_closes_its_own_client_when_the_host_fetcher_fails_to_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed: list[bool] = []

    async def _record_close(self: ProviderHttp) -> None:
        closed.append(True)

    monkeypatch.setattr(ProviderHttp, "aclose", _record_close)
    AnyFetch.register_builtin(_FailingCloseFetcher)
    provider = AnyFetch.create("builtin")
    with pytest.raises(RuntimeError, match="failed to close"):
        await provider.aclose()
    assert closed == [True]
