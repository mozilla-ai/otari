"""What every provider must do, checked over ``AnySearch.get_supported_providers()``.

A provider joins by adding to ``SCENARIOS`` how to drive it into each case. A
provider that calls HTTP answers from a handler on ``httpx.MockTransport``,
replaying its recorded fixtures.
"""

import logging
from collections.abc import AsyncIterator, Callable
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Literal, get_args

import httpx
import pytest

from any_search import (
    AnySearch,
    ProviderError,
    SearchError,
    SearchResult,
    TimeRange,
    UnsupportedParameterError,
)

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
        "empty": Scenario({"hits": []}),
        "error": Scenario({"error": "rate_limit", "error_status": 429}),
        "in_body_error": Scenario({"in_body_error": "engine_unavailable"}),
    },
}

PROVIDERS = AnySearch.get_supported_providers()
# The test tier answers from memory and sends no requests.
HTTP_PROVIDERS = [provider for provider in PROVIDERS if AnySearch.get_provider_metadata(provider).tier != "test"]
SENTINEL_QUERY = "sentinel-query-5f1c2b"
SENTINEL_KEY = "sentinel-key-9d4e7a"
TIMEOUT = 7.5


def _scenario(provider: str, case: Case) -> Scenario:
    scenario = SCENARIOS[provider].get(case)
    if scenario is None:
        pytest.skip(f"{provider} signals no {case}")
    return scenario


@asynccontextmanager
async def _provider(provider: str, scenario: Scenario, **kwargs: Any) -> AsyncIterator[AnySearch]:
    async with AsyncExitStack() as stack:
        if scenario.handler is not None and "client" not in kwargs:
            transport = httpx.MockTransport(scenario.handler)
            kwargs["client"] = await stack.enter_async_context(httpx.AsyncClient(transport=transport))
        yield await stack.enter_async_context(AnySearch.create(provider, timeout=TIMEOUT, **kwargs))


async def _search(provider: str, case: Case, **kwargs: Any) -> SearchResult:
    scenario = _scenario(provider, case)
    async with _provider(provider, scenario) as engine:
        return await engine.search("conformance", **scenario.options, **kwargs)


def _exception_texts(exc: BaseException | None) -> list[str]:
    texts: list[str] = []
    while exc is not None:
        texts += [str(exc), repr(exc)]
        exc = exc.__cause__ or exc.__context__
    return texts


def test_every_provider_has_scenarios() -> None:
    assert sorted(SCENARIOS) == PROVIDERS
    for cases in SCENARIOS.values():
        assert {"normal", "empty", "error"} <= cases.keys()


@pytest.mark.parametrize("provider", PROVIDERS)
def test_metadata_describes_the_provider(provider: str) -> None:
    metadata = AnySearch.get_provider_metadata(provider)
    assert metadata.name == provider
    assert metadata.max_results >= 1
    assert metadata.env_key or not metadata.requires_api_key
    names = [option.name for option in metadata.options]
    assert len(names) == len(set(names))
    assert not {"max_results", "time_range"} & set(names)


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", PROVIDERS)
async def test_a_normal_answer_fills_the_envelope(provider: str) -> None:
    result = await _search(provider, "normal")
    assert result.provider == provider
    assert result.hits
    assert result.error is None
    for hit in result.hits:
        assert hit.url
        assert isinstance(hit.title, str)
        assert isinstance(hit.snippet, str)
    assert (result.cost is None) == (result.cost_source == "none")
    assert "raw=" not in repr(result)
    assert "raw=" not in repr(result.hits[0])


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", PROVIDERS)
async def test_max_results_is_honored(provider: str) -> None:
    assert len((await _search(provider, "normal", max_results=1)).hits) <= 1


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize("time_range", get_args(TimeRange))
async def test_every_time_range_is_taken(provider: str, time_range: TimeRange) -> None:
    assert (await _search(provider, "normal", time_range=time_range)).error is None


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", PROVIDERS)
async def test_an_empty_answer_has_no_hits(provider: str) -> None:
    result = await _search(provider, "empty")
    assert (result.hits, result.error) == ([], None)


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", PROVIDERS)
async def test_a_failure_raises_a_provider_error(provider: str) -> None:
    with pytest.raises(ProviderError) as raised:
        await _search(provider, "error")
    assert raised.value.provider == provider
    assert raised.value.tag


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", PROVIDERS)
async def test_an_error_inside_a_successful_answer_is_returned(provider: str) -> None:
    result = await _search(provider, "in_body_error")
    assert isinstance(result.error, SearchError)
    assert result.error.tag


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", PROVIDERS)
async def test_an_unknown_option_is_refused(provider: str) -> None:
    with pytest.raises(UnsupportedParameterError):
        await _search(provider, "normal", not_an_option=1)


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize("case", get_args(Case))
async def test_no_exception_repr_or_log_record_carries_the_query_or_the_key(
    provider: str, case: Case, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    caplog.set_level(logging.DEBUG, logger="httpx")
    scenario = _scenario(provider, case)
    texts: list[str] = []
    async with _provider(provider, scenario, api_key=SENTINEL_KEY) as engine:
        texts.append(repr(engine))
        try:
            await engine.search(SENTINEL_QUERY, **scenario.options)
        except Exception as exc:
            texts += _exception_texts(exc)
    texts += [record.getMessage() for record in caplog.records]
    assert [text for text in texts if SENTINEL_QUERY in text or SENTINEL_KEY in text] == []


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
        async with _provider(provider, scenario, client=client) as engine:
            await engine.search("conformance", **scenario.options)
        assert sent
        expected = {"connect": TIMEOUT, "read": TIMEOUT, "write": TIMEOUT, "pool": TIMEOUT}
        assert all(request.extensions["timeout"] == expected for request in sent)
        assert not client.is_closed
