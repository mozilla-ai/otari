"""What every provider must do, checked over ``AnyFetch.get_supported_providers()``.

A provider joins by adding to ``SCENARIOS`` how to drive it into each case. A
provider that calls HTTP answers from a handler on ``httpx.MockTransport``,
replaying its recorded fixtures. ``builtin`` is the host's: it is skipped
while no factory is registered, and the host checks its own (in otari, the
fetch backend's tests).
"""

import logging
from collections.abc import AsyncIterator, Callable
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Literal, get_args

import httpx
import pytest

from any_fetch import (
    AnyFetch,
    BuiltinNotRegisteredError,
    FetchedPage,
    FetchError,
    ProviderError,
    UnsupportedParameterError,
)
from any_fetch.providers import builtin

Case = Literal["normal", "empty", "error", "in_body_error"]


@dataclass(frozen=True)
class Scenario:
    options: dict[str, Any] = field(default_factory=dict)
    # Answers the provider's requests; None for a provider that sends none.
    handler: Callable[[httpx.Request], httpx.Response] | None = None


# A provider that signals no error inside a successful response leaves "in_body_error" out.
SCENARIOS: dict[str, dict[Case, Scenario]] = {
    "fake": {
        "normal": Scenario(),
        "empty": Scenario({"text": ""}),
        "error": Scenario({"error": "http_status", "error_status": 404}),
        "in_body_error": Scenario({"in_body_error": "crawl_not_found"}),
    },
}

PROVIDERS = AnyFetch.get_supported_providers()
# The test tier answers from memory, and builtin is the host's.
HTTP_PROVIDERS = [
    provider
    for provider in PROVIDERS
    if provider != "builtin" and AnyFetch.get_provider_metadata(provider).tier != "test"
]
URL = "https://example.com/conformance"
SENTINEL_URL = "https://sentinel-host.example/sentinel-path-3b8f"
SENTINEL_KEY = "sentinel-key-9d4e7a"
TIMEOUT = 7.5


def _scenario(provider: str, case: Case) -> Scenario:
    if provider == "builtin" and not builtin.is_registered():
        pytest.skip("no builtin factory is registered")
    scenario = SCENARIOS[provider].get(case)
    if scenario is None:
        pytest.skip(f"{provider} signals no {case}")
    return scenario


@asynccontextmanager
async def _provider(provider: str, scenario: Scenario, **kwargs: Any) -> AsyncIterator[AnyFetch]:
    async with AsyncExitStack() as stack:
        if scenario.handler is not None and "client" not in kwargs:
            transport = httpx.MockTransport(scenario.handler)
            kwargs["client"] = await stack.enter_async_context(httpx.AsyncClient(transport=transport))
        yield await stack.enter_async_context(AnyFetch.create(provider, timeout=TIMEOUT, **kwargs))


async def _fetch(provider: str, case: Case, **kwargs: Any) -> FetchedPage:
    scenario = _scenario(provider, case)
    async with _provider(provider, scenario) as fetcher:
        return await fetcher.fetch(URL, **scenario.options, **kwargs)


def _exception_texts(exc: BaseException | None) -> list[str]:
    texts: list[str] = []
    while exc is not None:
        texts += [str(exc), repr(exc)]
        exc = exc.__cause__ or exc.__context__
    return texts


def test_every_provider_has_scenarios() -> None:
    assert sorted([*SCENARIOS, "builtin"]) == PROVIDERS
    for cases in SCENARIOS.values():
        assert {"normal", "empty", "error"} <= cases.keys()


def test_builtin_before_registration_is_a_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(builtin, "_factory", None)
    with pytest.raises(BuiltinNotRegisteredError):
        AnyFetch.create("builtin")


@pytest.mark.parametrize("provider", PROVIDERS)
def test_metadata_describes_the_provider(provider: str) -> None:
    metadata = AnyFetch.get_provider_metadata(provider)
    assert metadata.name == provider
    assert metadata.max_urls_per_call >= 1
    assert metadata.formats
    assert metadata.env_key or not metadata.requires_api_key
    names = [option.name for option in metadata.options]
    assert len(names) == len(set(names))
    assert "max_chars" not in names


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", PROVIDERS)
async def test_a_normal_answer_fills_the_envelope(provider: str) -> None:
    page = await _fetch(provider, "normal")
    assert page.url == URL
    assert page.final_url
    assert page.text
    assert isinstance(page.title, str)
    assert isinstance(page.content_type, str)
    assert page.error is None
    assert (page.cost is None) == (page.cost_source == "none")
    assert "raw=" not in repr(page)


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", PROVIDERS)
async def test_max_chars_is_honored(provider: str) -> None:
    page = await _fetch(provider, "normal", max_chars=10)
    assert len(page.text) <= 10
    assert page.text_truncated


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", PROVIDERS)
async def test_an_empty_answer_has_no_text(provider: str) -> None:
    page = await _fetch(provider, "empty")
    assert (page.text, page.error) == ("", None)


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", PROVIDERS)
async def test_a_failure_raises_a_provider_error(provider: str) -> None:
    with pytest.raises(ProviderError) as raised:
        await _fetch(provider, "error")
    assert raised.value.provider == provider
    assert raised.value.tag


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", PROVIDERS)
async def test_an_error_inside_a_successful_answer_is_returned(provider: str) -> None:
    page = await _fetch(provider, "in_body_error")
    assert isinstance(page.error, FetchError)
    assert page.error.tag


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", PROVIDERS)
async def test_an_unknown_option_is_refused(provider: str) -> None:
    with pytest.raises(UnsupportedParameterError):
        await _fetch(provider, "normal", not_an_option=1)


@pytest.mark.asyncio
@pytest.mark.usefixtures("log_filter")
@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize("case", get_args(Case))
async def test_no_exception_repr_or_log_record_carries_the_url_or_the_key(
    provider: str, case: Case, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    caplog.set_level(logging.DEBUG, logger="httpx")
    scenario = _scenario(provider, case)
    texts: list[str] = []
    async with _provider(provider, scenario, api_key=SENTINEL_KEY) as fetcher:
        texts.append(repr(fetcher))
        try:
            await fetcher.fetch(SENTINEL_URL, **scenario.options)
        except Exception as exc:
            texts += _exception_texts(exc)
    texts += [record.getMessage() for record in caplog.records]
    sentinels = ("sentinel-host", "sentinel-path", SENTINEL_KEY)
    assert [text for text in texts if any(sentinel in text for sentinel in sentinels)] == []


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", HTTP_PROVIDERS)
async def test_a_given_client_is_used_with_the_timeout_and_left_open(provider: str) -> None:
    scenario = _scenario(provider, "normal")
    handler = scenario.handler
    assert handler is not None
    sent: list[httpx.Request] = []

    def answer(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return handler(request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(answer)) as client:
        async with _provider(provider, scenario, client=client) as fetcher:
            await fetcher.fetch(URL, **scenario.options)
        assert sent
        expected = {"connect": TIMEOUT, "read": TIMEOUT, "write": TIMEOUT, "pool": TIMEOUT}
        assert all(request.extensions["timeout"] == expected for request in sent)
        assert not client.is_closed
