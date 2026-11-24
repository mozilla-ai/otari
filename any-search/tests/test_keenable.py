"""The Keenable adapter: the request it sends, keyed and keyless, and how it reads Keenable's recorded answers."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest

from any_search import AnySearch, ProviderError, SearchResult, UnsupportedParameterError
from any_search._http import USER_AGENT
from any_search.providers.keenable import DEFAULT_COUNT, DEFAULT_TITLE, MAX_RESULTS, TITLE_HEADER

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "keenable"
KEY = "keen_test-key"
QUERY = "python 3.14 release announcement"
SENTINEL = "sentinel-body-7c1e"


def _recorded(case: str) -> httpx.Response:
    recorded = json.loads((FIXTURES / f"{case}.json").read_text())
    return httpx.Response(recorded["status"], json=recorded["body"])


async def _search(
    answer: httpx.Response | None = None,
    *,
    api_key: str | None = KEY,
    api_base: str | None = None,
    **kwargs: Any,
) -> tuple[SearchResult, httpx.Request]:
    """Search through a transport that answers with ``answer`` and keeps the request it was sent."""
    sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return answer if answer is not None else _recorded("normal")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        async with AnySearch.create("keenable", api_key=api_key, api_base=api_base, client=client) as keenable:
            result = await keenable.search(QUERY, **kwargs)
    return result, sent[0]


@pytest.fixture(autouse=True)
def _no_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neither the developer's key nor their base URL reaches a test."""
    monkeypatch.delenv("KEENABLE_API_KEY", raising=False)
    monkeypatch.delenv("KEENABLE_API_URL", raising=False)


def test_metadata() -> None:
    metadata = AnySearch.get_provider_metadata("keenable")
    assert (metadata.tier, metadata.env_key, metadata.requires_api_key) == ("production", "KEENABLE_API_KEY", False)
    assert (metadata.env_api_base, metadata.default_api_base) == ("KEENABLE_API_URL", "https://api.keenable.ai")
    assert (metadata.query_in_url, metadata.key_in_url, metadata.reports_cost) == (False, False, False)
    assert metadata.max_results == MAX_RESULTS == 50
    options = {option.name: option for option in metadata.options}
    assert [name for name, option in options.items() if option.operator_only] == [TITLE_HEADER]
    assert options[TITLE_HEADER].default == DEFAULT_TITLE == "any-search"
    assert not any(option.cost_tier for option in metadata.options)
    # Deprecated by Keenable, so not offered.
    assert "safety_mode" not in options


@pytest.mark.asyncio
async def test_a_key_goes_to_the_keyed_endpoint_in_a_header() -> None:
    _, request = await _search()
    assert (request.method, str(request.url)) == ("POST", "https://api.keenable.ai/v1/search")
    assert json.loads(request.content) == {"query": QUERY, "max_results": DEFAULT_COUNT}
    assert request.headers["X-API-Key"] == KEY
    assert (request.headers[TITLE_HEADER], request.headers["User-Agent"]) == (DEFAULT_TITLE, USER_AGENT)
    assert "Authorization" not in request.headers


@pytest.mark.asyncio
@pytest.mark.parametrize("api_key", [None, ""])
async def test_without_a_key_the_search_goes_to_the_keyless_endpoint_with_the_title(api_key: str | None) -> None:
    _, request = await _search(api_key=api_key)
    assert str(request.url) == "https://api.keenable.ai/v1/search/public"
    assert request.headers[TITLE_HEADER] == DEFAULT_TITLE
    assert "X-API-Key" not in request.headers


@pytest.mark.asyncio
async def test_the_key_comes_from_the_environment_when_none_is_passed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KEENABLE_API_KEY", "keen_from-the-environment")
    _, request = await _search(api_key=None)
    assert str(request.url) == "https://api.keenable.ai/v1/search"
    assert request.headers["X-API-Key"] == "keen_from-the-environment"


@pytest.mark.asyncio
async def test_an_explicit_empty_key_never_falls_back_to_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KEENABLE_API_KEY", "keen_the-hosts-own")
    _, request = await _search(api_key="")
    assert str(request.url) == "https://api.keenable.ai/v1/search/public"
    assert "X-API-Key" not in request.headers


@pytest.mark.asyncio
@pytest.mark.parametrize("api_key", [KEY, None])
async def test_the_title_option_names_the_app(api_key: str | None) -> None:
    title: dict[str, Any] = {TITLE_HEADER: "Acme gateway"}
    _, request = await _search(api_key=api_key, **title)
    assert request.headers[TITLE_HEADER] == "Acme gateway"
    assert TITLE_HEADER not in json.loads(request.content)


@pytest.mark.asyncio
async def test_api_base_replaces_the_endpoint_keyed_and_keyless() -> None:
    _, keyed = await _search(api_base="https://proxy.example/keenable/")
    _, keyless = await _search(api_key="", api_base="https://proxy.example/keenable/")
    assert str(keyed.url) == "https://proxy.example/keenable/v1/search"
    assert str(keyless.url) == "https://proxy.example/keenable/v1/search/public"


@pytest.mark.asyncio
async def test_native_options_are_sent_as_given_and_none_is_not_sent() -> None:
    options: dict[str, Any] = {
        "mode": "realtime",
        "site": "python.org",
        "exclude_domains": ["example.com", ".gov"],
        "published_after": "2026-01-01",
        "published_before": "2026-11-01",
        "acquired_after": "30d",
        "acquired_before": "1d",
        "snippet_max_length": 180,
        "disable_cards": True,
        "moderation": "strict",
        "query_time": "7d",
        "objective": "Release notes for the newest Python.",
    }
    _, request = await _search(**options)
    assert json.loads(request.content) == {"query": QUERY, "max_results": DEFAULT_COUNT, **options}
    _, request = await _search(mode=None, site=None, **{TITLE_HEADER: None})
    assert json.loads(request.content) == {"query": QUERY, "max_results": DEFAULT_COUNT}
    assert request.headers[TITLE_HEADER] == DEFAULT_TITLE


@pytest.mark.asyncio
@pytest.mark.parametrize(("max_results", "sent"), [(None, DEFAULT_COUNT), (1, 1), (50, 50), (51, 50), (500, 50)])
async def test_max_results_is_sent_capped_at_keenables_limit(max_results: int | None, sent: int) -> None:
    _, request = await _search(max_results=max_results)
    assert json.loads(request.content)["max_results"] == sent


@pytest.mark.asyncio
@pytest.mark.parametrize(("time_range", "span"), [("day", "1d"), ("week", "7d"), ("month", "30d"), ("year", "365d")])
async def test_time_range_becomes_published_after_and_wins_over_it(time_range: Any, span: str) -> None:
    _, request = await _search(time_range=time_range, published_after="2020-01-01")
    assert json.loads(request.content)["published_after"] == span


@pytest.mark.asyncio
async def test_an_option_keenable_does_not_take_is_refused() -> None:
    with pytest.raises(UnsupportedParameterError):
        await _search(safety_mode="strict")


@pytest.mark.asyncio
async def test_reads_the_recorded_answer() -> None:
    result, _ = await _search()
    assert result.provider == "keenable"
    assert [hit.url for hit in result.hits] == [
        "https://python.org/downloads/release/python-3149",
        "https://blog.python.org/2026/10/python-3152-31410-31318-31216-31118",
        "https://docs.python.org/3.14/whatsnew/3.14.html",
    ]
    assert all(hit.title and hit.snippet and hit.text is None for hit in result.hits)
    assert [hit.published for hit in result.hits] == [None, datetime(2026, 10, 26, tzinfo=UTC), None]
    assert (result.cost, result.cost_source, result.error) == (None, "none", None)
    assert result.raw["mode"] == "pro"
    assert result.hits[0].raw["acquired_at"]


@pytest.mark.asyncio
async def test_an_empty_answer_has_no_hits_and_no_error() -> None:
    result, _ = await _search(_recorded("empty"))
    assert (result.hits, result.error) == ([], None)


def _item(name: str, **fields: Any) -> dict[str, Any]:
    return {"title": f"Page {name}", "url": f"https://example.com/{name}", "snippet": f"About {name}.", **fields}


def _answer(*items: dict[str, Any]) -> dict[str, Any]:
    return {"query": QUERY, "mode": "pro", "results": list(items) or [_item("a")]}


@pytest.mark.asyncio
async def test_a_blank_snippet_falls_back_to_the_description() -> None:
    body = _answer(
        _item("a", snippet="  ", description="The description."),
        _item("b", description="Unused."),
        {"title": "", "url": "https://example.com/c"},
    )
    result, _ = await _search(httpx.Response(200, json=body))
    assert [(hit.title, hit.snippet) for hit in result.hits] == [
        ("Page a", "The description."),
        ("Page b", "About b."),
        ("", ""),
    ]


@pytest.mark.asyncio
async def test_a_publish_date_without_a_zone_is_utc() -> None:
    body = _answer(_item("a", published_at="2026-11-20T18:34:00"), _item("b", published_at="2026-11-20T18:34:00+02:00"))
    result, _ = await _search(httpx.Response(200, json=body))
    assert result.hits[0].published == datetime(2026, 11, 20, 18, 34, tzinfo=UTC)
    assert result.hits[1].published == datetime(2026, 11, 20, 16, 34, tzinfo=UTC)


@pytest.mark.asyncio
async def test_only_the_results_within_the_count_are_read() -> None:
    body = _answer(_item("a"), {"url": 7})
    result, _ = await _search(httpx.Response(200, json=body), max_results=1)
    assert [hit.url for hit in result.hits] == ["https://example.com/a"]


@pytest.mark.asyncio
async def test_unread_fields_are_ignored() -> None:
    body = _answer(_item("a", acquired_at=7, score={"any": "thing"}))
    body.update(usage={"credits": 1}, warnings=None, newTopLevel=[])
    result, _ = await _search(httpx.Response(200, json=body))
    assert [hit.url for hit in result.hits] == ["https://example.com/a"]


@pytest.mark.asyncio
async def test_a_refused_key_raises_with_keenables_code_and_no_body_text() -> None:
    with pytest.raises(ProviderError) as raised:
        await _search(_recorded("error"))
    assert (raised.value.status, raised.value.tag) == (401, "invalid_api_key")
    assert "Malformed" not in str(raised.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("answer", "tag"),
    [
        (httpx.Response(400, json={"code": "missing_app_title", "message": SENTINEL}), "missing_app_title"),
        (httpx.Response(400, json={"code": "unknown_parameter", "param": SENTINEL}), "unknown_parameter"),
        (httpx.Response(402, json={"code": "insufficient_credits", "fix": SENTINEL}), "insufficient_credits"),
        (httpx.Response(429, json={"code": "public_rate_limit_hourly"}), "public_rate_limit_hourly"),
        (httpx.Response(429, json={"code": "rate_limit_per_second"}), "rate_limit_per_second"),
        (httpx.Response(429, json={"code": f"has spaces {SENTINEL}"}), "rate_limited"),
        (httpx.Response(429, text=SENTINEL), "rate_limited"),
        (httpx.Response(431, text=SENTINEL), "http_error"),
        (httpx.Response(500, json={"error": SENTINEL}), "http_error"),
        (httpx.Response(503, json={"code": 7}), "http_error"),
    ],
)
async def test_any_failure_is_tagged_by_keenables_code_without_its_text(answer: httpx.Response, tag: str) -> None:
    with pytest.raises(ProviderError) as raised:
        await _search(answer)
    assert (raised.value.status, raised.value.tag) == (answer.status_code, tag)
    assert SENTINEL not in str(raised.value) and SENTINEL not in repr(raised.value)


def _setting(value: Any, *path: str | int) -> Any:
    """A change to an answer that sets the field at ``path`` to ``value``."""

    def change(body: dict[str, Any]) -> None:
        parent: Any = body
        for step in path[:-1]:
            parent = parent[step]
        parent[path[-1]] = value

    return change


def _without(*path: str | int) -> Any:
    """A change to an answer that deletes the field at ``path``."""

    def change(body: dict[str, Any]) -> None:
        parent: Any = body
        for step in path[:-1]:
            parent = parent[step]
        del parent[path[-1]]

    return change


ITEM = ("results", 0)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change",
    [
        pytest.param(_without("results"), id="results-missing"),
        pytest.param(_setting(None, "results"), id="results-null"),
        pytest.param(_setting({}, "results"), id="results-not-a-list"),
        pytest.param(_setting(["an item"], "results"), id="item-not-an-object"),
        pytest.param(_without(*ITEM, "url"), id="url-missing"),
        pytest.param(_setting("", *ITEM, "url"), id="url-empty"),
        pytest.param(_setting(7, *ITEM, "url"), id="url-not-a-string"),
        pytest.param(_without(*ITEM, "title"), id="title-missing"),
        pytest.param(_setting(None, *ITEM, "title"), id="title-null"),
        pytest.param(_setting(None, *ITEM, "snippet"), id="snippet-null"),
        pytest.param(_setting(["text"], *ITEM, "description"), id="description-not-a-string"),
        pytest.param(_setting(20261101, *ITEM, "published_at"), id="published-not-a-string"),
        pytest.param(_setting("yesterday", *ITEM, "published_at"), id="published-not-iso"),
        pytest.param(_setting("2026-13-45T00:00:00Z", *ITEM, "published_at"), id="published-unparsable"),
    ],
)
async def test_a_field_missing_or_of_the_wrong_type_raises_without_the_body(change: Any) -> None:
    body = _answer(_item("a", description=SENTINEL))
    change(body)
    with pytest.raises(ProviderError) as raised:
        await _search(httpx.Response(200, json=body))
    assert (raised.value.status, raised.value.tag) == (200, "unexpected_response")
    assert SENTINEL not in str(raised.value) and SENTINEL not in repr(raised.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "answer",
    [httpx.Response(200, text="<html>results</html>"), httpx.Response(200, json=[])],
)
async def test_a_body_that_is_not_a_search_answer_is_refused(answer: httpx.Response) -> None:
    with pytest.raises(ProviderError) as raised:
        await _search(answer)
    assert (raised.value.status, raised.value.tag) == (200, "invalid_response")


def test_no_key_is_needed() -> None:
    assert repr(AnySearch.create("keenable")) == "<KeenableProvider provider='keenable'>"
