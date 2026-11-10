"""Reading the MCP Registry: which entries the catalog keeps, and how one page is fetched."""

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from gateway.services.mcp_catalog import _registry
from gateway.services.mcp_catalog._registry import (
    REGISTRY_URL,
    RegistryClient,
    RegistryError,
    entry_from_registry_item,
    fallback_title,
)
from gateway.services.url_safety import UnsafeURLError

_PAGES = Path(__file__).parent.parent / "fixtures" / "mcp_registry"


def _recorded(name: str) -> dict[str, Any]:
    page: dict[str, Any] = json.loads((_PAGES / f"{name}.json").read_text())
    return page


def _item(*, name: str = "com.example/mcp", status: str = "active", **server: Any) -> dict[str, Any]:
    server.setdefault("remotes", [{"type": "streamable-http", "url": "https://mcp.example.com/mcp"}])
    return {
        "server": {"name": name, "description": "An example.", "version": "1.0.0", **server},
        "_meta": {"io.modelcontextprotocol.registry/official": {"status": status}},
    }


@pytest.fixture(autouse=True)
def _registry_address_is_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    """The address check resolves the registry's host, and a unit test has no resolver to rely on."""

    async def safe(url: str, **_kwargs: Any) -> None:
        return None

    monkeypatch.setattr(_registry, "validate_mcp_url", safe)


def test_the_recorded_pages_keep_only_what_otari_can_connect_to() -> None:
    items = [*_recorded("page_1")["servers"], *_recorded("page_2")["servers"]]

    kept = {entry.name: entry for item in items if (entry := entry_from_registry_item(item)) is not None}

    assert sorted(kept) == [
        "ac.inference.sh/mcp",
        "ac.snag/snag",
        "academy.apexai/apex-ai-academy",
        "ae.easyvoice/tts",
        "africa.lomi/mcp",
        "agency.thirdhemisphere/ai-authority-index",
        "ai.adadvisor/mcp-server",
    ]
    # A package with no remote and a deprecated server are the two left out.
    assert len(items) == 9
    assert kept["ac.inference.sh/mcp"].title == "inference.sh"
    assert kept["ac.inference.sh/mcp"].url == "https://api.inference.sh/mcp"
    assert kept["ac.snag/snag"].title == "Snag"
    assert kept["ae.easyvoice/tts"].requires_token is True
    # An optional Authorization header, and an optional header of another name, ask nothing of the caller.
    assert kept["academy.apexai/apex-ai-academy"].requires_token is False
    assert kept["agency.thirdhemisphere/ai-authority-index"].requires_token is False
    assert kept["africa.lomi/mcp"].url == "https://mcp.lomi.africa/mcp"
    assert kept["ae.easyvoice/tts"].website_url == "https://easyvoice.ae/tts-mcp-server"
    assert all(entry.source == "registry" for entry in kept.values())


@pytest.mark.parametrize(
    "server",
    [
        {"remotes": []},
        {"remotes": None},
        {"remotes": [{"type": "sse", "url": "https://mcp.example.com/sse"}]},
        {"remotes": [{"type": "streamable-http", "url": "http://mcp.example.com/mcp"}]},
        {"remotes": [{"type": "streamable-http", "url": "https://{tenant}.example.com/mcp"}]},
        {"remotes": [{"type": "streamable-http", "url": "https://10.0.0.5/mcp"}]},
        {"remotes": [{"type": "streamable-http", "url": "https://169.254.169.254/latest"}]},
        {"remotes": [{"type": "streamable-http", "url": "https://localhost/mcp"}]},
        {"remotes": [{"type": "streamable-http", "url": "https://intranet/mcp"}]},
        {"remotes": [{"type": "streamable-http", "url": "https://user:secret@mcp.example.com/mcp"}]},
        {"remotes": [{"type": "streamable-http", "url": "https://127.1/mcp"}]},
        {"remotes": [{"type": "streamable-http", "url": "https://mcp.example.com/m cp"}]},
        {"remotes": [{"type": "streamable-http", "url": "https://mcp.example.com/\x00mcp"}]},
        {"remotes": [{"type": "streamable-http", "url": "https://mcp.example.com/\ud83d"}]},
        {"remotes": [{"type": "streamable-http", "url": "https://mcp.example.com/" + "a" * 2048}]},
        {
            "remotes": [
                {
                    "type": "streamable-http",
                    "url": "https://mcp.example.com/mcp",
                    "headers": [{"name": "X-API-Key", "isRequired": True, "isSecret": True}],
                }
            ]
        },
    ],
    ids=[
        "no remote",
        "remotes missing",
        "sse only",
        "plain http",
        "templated url",
        "private address",
        "metadata address",
        "localhost",
        "single label host",
        "credentials in url",
        "address in short form",
        "space in url",
        "nul in url",
        "lone surrogate in url",
        "url too long",
        "a required header Otari cannot send",
    ],
)
def test_a_server_otari_cannot_connect_to_is_left_out(server: dict[str, Any]) -> None:
    assert entry_from_registry_item(_item(**server)) is None


@pytest.mark.parametrize("status", ["deprecated", "deleted"])
def test_a_server_the_registry_retired_is_left_out(status: str) -> None:
    assert entry_from_registry_item(_item(status=status)) is None


@pytest.mark.parametrize(
    "item",
    [
        None,
        "text",
        {},
        {"server": []},
        {"server": {"name": ""}},
        {"server": {"name": 7}},
        {"server": {"name": "com.example/\ud83d", "remotes": [{"type": "streamable-http", "url": "https://x.io/"}]}},
        {"server": {"name": "com.example/a b", "remotes": [{"type": "streamable-http", "url": "https://x.io/"}]}},
    ],
)
def test_a_malformed_item_is_left_out(item: object) -> None:
    assert entry_from_registry_item(item) is None


def test_a_second_remote_is_used_when_the_first_cannot_be() -> None:
    entry = entry_from_registry_item(
        _item(
            remotes=[
                {"type": "sse", "url": "https://mcp.example.com/sse"},
                {"type": "streamable-http", "url": " https://mcp.example.com/mcp "},
            ]
        )
    )

    assert entry is not None
    assert entry.url == "https://mcp.example.com/mcp"


def test_text_from_the_registry_is_stored_as_one_bounded_line() -> None:
    entry = entry_from_registry_item(
        _item(
            title="Example\n\tServer" + " x" * 200, description="line one\r\nline two", websiteUrl="http://example.com"
        )
    )

    assert entry is not None
    assert entry.title.startswith("Example Server x")
    assert len(entry.title) <= 128
    assert entry.description == "line one line two"
    # A website that is not https is dropped, and the server is kept.
    assert entry.website_url is None


def test_text_no_database_can_store_is_dropped_and_the_server_kept() -> None:
    # json.loads accepts a lone surrogate, SQLite refuses to encode one, and PostgreSQL refuses a NUL.
    entry = entry_from_registry_item(_item(title="Bad \ud83d", description="nul \x00 here", version="1\ud83d"))

    assert entry is not None
    assert (entry.title, entry.description, entry.version) == ("Example", None, None)
    entry.model_dump_json().encode("utf-8")


@pytest.mark.parametrize(
    ("name", "title"),
    [
        ("com.stripe/mcp", "Stripe"),
        ("io.github.getsentry/sentry-mcp", "Sentry"),
        ("ac.snag/snag", "Snag"),
        ("com.example/mcp-server", "Example"),
        ("com.example/billing_tools", "Billing tools"),
        ("io.github.acme/mcp", "Acme"),
        ("ai.acme.mcp/server", "Acme"),
        ("nonsense", "Nonsense"),
    ],
)
def test_a_server_without_a_title_gets_one_from_its_name(name: str, title: str) -> None:
    assert fallback_title(name) == title


class _Registry:
    """Answers the registry's URL from a script, keeping the query of every request."""

    def __init__(self, *answers: httpx.Response | Exception) -> None:
        self._answers = list(answers)
        self.queries: list[dict[str, str]] = []
        self.slept: list[float] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        assert str(request.url).startswith(REGISTRY_URL)
        self.queries.append(dict(request.url.params))
        answer = self._answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    async def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)

    def client(self) -> RegistryClient:
        return RegistryClient(transport=httpx.MockTransport(self), sleep=self.sleep)


@pytest.mark.asyncio
async def test_a_page_is_read_with_its_cursor() -> None:
    registry = _Registry(httpx.Response(200, json=_recorded("page_1")), httpx.Response(200, json=_recorded("page_2")))

    async with registry.client() as client:
        first = await client.fetch_page(None)
        second = await client.fetch_page(first.next_cursor)

    assert len(first.items) == 6
    assert first.next_cursor == "agency.thirdhemisphere/ai-authority-index:1.0.0"
    assert len(second.items) == 3
    assert second.next_cursor is None
    assert registry.queries == [
        {"version": "latest", "limit": "100"},
        {"version": "latest", "limit": "100", "cursor": "agency.thirdhemisphere/ai-authority-index:1.0.0"},
    ]


@pytest.mark.asyncio
async def test_a_server_error_and_a_network_error_are_retried() -> None:
    registry = _Registry(
        httpx.Response(503), httpx.ConnectError("refused"), httpx.Response(200, json=_recorded("page_2"))
    )

    async with registry.client() as client:
        page = await client.fetch_page(None)

    assert len(page.items) == 3
    assert registry.slept == [2.0, 4.0]


@pytest.mark.asyncio
async def test_a_registry_that_stays_down_is_given_up_on() -> None:
    registry = _Registry(httpx.Response(500), httpx.Response(502), httpx.Response(503))

    async with registry.client() as client:
        with pytest.raises(RegistryError, match="did not answer after 3 attempts: HTTP 503"):
            await client.fetch_page(None)


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [301, 400, 404, 429])
async def test_a_refusal_and_a_redirect_are_not_retried(status: int) -> None:
    registry = _Registry(httpx.Response(status, headers={"Location": "http://169.254.169.254/"}))

    async with registry.client() as client:
        with pytest.raises(RegistryError, match=f"HTTP {status}"):
            await client.fetch_page(None)

    assert len(registry.queries) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body", [b"<html>", b"[]", b'{"servers": {}}', b'{"metadata": {}}'], ids=["not json", "a list", "no list", "absent"]
)
async def test_an_answer_of_the_wrong_shape_is_refused(body: bytes) -> None:
    registry = _Registry(httpx.Response(200, content=body))

    async with registry.client() as client:
        with pytest.raises(RegistryError):
            await client.fetch_page(None)


@pytest.mark.asyncio
async def test_an_oversized_page_is_refused_without_a_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_registry, "MAX_PAGE_BYTES", 64)
    registry = _Registry(httpx.Response(200, json=_recorded("page_1")))

    async with registry.client() as client:
        with pytest.raises(RegistryError, match="exceeded 64 bytes"):
            await client.fetch_page(None)

    assert len(registry.queries) == 1


@pytest.mark.asyncio
async def test_nothing_is_sent_when_the_registry_address_fails_the_safety_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def unsafe(url: str, **_kwargs: Any) -> None:
        raise UnsafeURLError("resolves to 10.0.0.1")

    monkeypatch.setattr(_registry, "validate_mcp_url", unsafe)
    registry = _Registry()

    async with registry.client() as client:
        with pytest.raises(RegistryError, match="URL safety check") as raised:
            await client.fetch_page(None)

    assert registry.queries == []
    assert "10.0.0.1" not in str(raised.value)


@pytest.mark.asyncio
async def test_a_page_that_trickles_in_is_given_up_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_registry, "PAGE_DEADLINE_SECONDS", 0.05)

    async def stalls(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(5)
        return httpx.Response(200, json=_recorded("page_1"))

    async with RegistryClient(transport=httpx.MockTransport(stalls)) as client:
        with pytest.raises(RegistryError, match="took more than 0 seconds"):
            await client.fetch_page(None)


@pytest.mark.asyncio
async def test_a_client_that_fetches_nothing_opens_nothing() -> None:
    async with RegistryClient() as client:
        assert client._client is None
