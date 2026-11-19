"""Every client that dials an MCP server or a guardrail URL a tenant wrote takes the owned-endpoint rule.

A workspace, an organization or a caller chooses these URLs, so each request is
resolved again at connect time, refused unless every address is public, and dialed
at the address that was checked. The checks in ``url_safety`` resolve the name once,
when a URL is saved or sent; these tests re-point the name between that check and
the connection (DNS rebinding) and assert nothing internal is reached.

The connect-time resolver and the socket are both stood in for: the resolver by
patching :class:`SystemAddressResolver`, the socket by the pinned transport's
per-address transport, which records what it would have dialed.
"""

from __future__ import annotations

import ipaddress
from collections.abc import Callable, Iterator, Sequence
from typing import Any

import httpx
import httpx2
import pytest
from mcp.shared._httpx_utils import request_within_origin

from gateway.models.guardrails import GuardrailConfig
from gateway.services import guardrails as guardrails_module
from gateway.services import mcp_stateless, url_safety, web_retrieval_network
from gateway.services.guardrails import GuardrailsNotReachableError, run_input_guardrails
from gateway.services.mcp_client import _origin_bound_http_client
from gateway.services.providers import OwnedEndpointHttpx2Transport, OwnedEndpointTransport
from gateway.services.tools import _mcp_oauth
from gateway.services.url_safety import UnsafeURLError, validate_mcp_url
from gateway.services.web_retrieval_network import PinnedAsyncHTTPTransport

pytestmark = pytest.mark.asyncio

PUBLIC = "93.184.216.34"
HOST = "mcp.example.com"
URL = f"https://{HOST}/mcp"

INTERNAL = ["127.0.0.1", "::1", "10.0.0.5", "192.168.1.20", "100.64.0.1", "169.254.169.254"]


class _Socket:
    """What the pinned transport would have dialed, and the answer each dial gets."""

    def __init__(self) -> None:
        self.dialed: list[httpx.Request] = []
        self.answers: list[httpx.Response] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.dialed.append(request)
        if self.answers:
            return self.answers.pop(0)
        return httpx.Response(200, json={"ok": True})


class _Resolver:
    """The connect-time resolver: answers in order, repeating the last answer."""

    def __init__(self) -> None:
        self.answers: list[list[str]] = [[PUBLIC]]
        self.asked: list[str] = []

    def answer(self, *addresses: str) -> None:
        self.answers = [list(addresses)]

    def then(self, *rounds: Sequence[str]) -> None:
        self.answers = [list(r) for r in rounds]

    async def resolve(self, host: str, port: int) -> Sequence[Any]:
        self.asked.append(host)
        current = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        return tuple(ipaddress.ip_address(a) for a in current)


@pytest.fixture
def socket(monkeypatch: pytest.MonkeyPatch) -> _Socket:
    fake = _Socket()
    monkeypatch.setattr(PinnedAsyncHTTPTransport, "_build_transport", lambda self: httpx.MockTransport(fake.handle))
    return fake


@pytest.fixture
def resolver(monkeypatch: pytest.MonkeyPatch) -> _Resolver:
    fake = _Resolver()

    async def resolve(self: object, host: str, port: int) -> Sequence[Any]:
        return await fake.resolve(host, port)

    monkeypatch.setattr(web_retrieval_network.SystemAddressResolver, "resolve", resolve)
    return fake


@pytest.fixture
def saved_as_public(monkeypatch: pytest.MonkeyPatch) -> None:
    """The save-time check sees a public address, which is what a rebinding name answers first."""

    async def public(_host: str) -> list[object]:
        return [ipaddress.ip_address(PUBLIC)]

    monkeypatch.setattr(url_safety, "_resolve_all_async", public)


@pytest.fixture(autouse=True)
def _retired_flags_set(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Every test runs with the old MCP flags wide open, to show they reach none of this."""
    monkeypatch.setenv("OTARI_MCP_ALLOW_LOOPBACK", "true")
    monkeypatch.setenv("OTARI_MCP_ALLOW_PRIVATE_HOSTS", "true")
    yield


# --------------------------------------------------------------------------- #
# The three MCP clients and the guardrail client, as one parametrized subject
# --------------------------------------------------------------------------- #


async def _stateless(url: str) -> int:
    async with mcp_stateless.build_http_client_factory()(None) as client:
        return (await client.get(url)).status_code


async def _managed(url: str) -> int:
    async with _origin_bound_http_client(url) as client:
        return (await client.get(url)).status_code


async def _oauth(url: str) -> int:
    async with _mcp_oauth._http_client(httpx.Timeout(5.0)) as client:
        return (await client.get(url)).status_code


async def _guardrail(url: str) -> int:
    async with guardrails_module._tenant_endpoint_client() as client:
        return (await client.get(url)).status_code


CLIENTS: dict[str, Callable[[str], Any]] = {
    "stateless MCP": _stateless,
    "managed MCP": _managed,
    "MCP OAuth": _oauth,
    "guardrail URL": _guardrail,
}
REFUSED = (httpx.TransportError, httpx2.TransportError)


@pytest.mark.parametrize("dial", CLIENTS.values(), ids=CLIENTS.keys())
async def test_a_public_name_is_dialed_at_the_address_that_was_checked(
    dial: Callable[[str], Any], socket: _Socket, resolver: _Resolver
) -> None:
    assert await dial(URL) == 200

    assert resolver.asked == [HOST]
    [sent] = socket.dialed
    assert sent.url.host == PUBLIC
    assert sent.headers["host"] == HOST
    assert sent.extensions["sni_hostname"] == HOST


@pytest.mark.parametrize("address", INTERNAL)
@pytest.mark.parametrize("dial", CLIENTS.values(), ids=CLIENTS.keys())
async def test_a_name_resolving_inside_is_refused_before_anything_is_sent(
    dial: Callable[[str], Any], address: str, socket: _Socket, resolver: _Resolver
) -> None:
    resolver.answer(address)

    with pytest.raises(REFUSED):
        await dial(URL)

    assert socket.dialed == []


@pytest.mark.parametrize("dial", CLIENTS.values(), ids=CLIENTS.keys())
async def test_one_internal_answer_among_public_ones_refuses_the_name(
    dial: Callable[[str], Any], socket: _Socket, resolver: _Resolver
) -> None:
    resolver.answer(PUBLIC, "127.0.0.1")

    with pytest.raises(REFUSED):
        await dial(URL)

    assert socket.dialed == []


@pytest.mark.parametrize("literal", ["http://127.0.0.1:8000/mcp", "http://[::1]:8000/mcp", "https://10.0.0.5/mcp"])
@pytest.mark.parametrize("dial", CLIENTS.values(), ids=CLIENTS.keys())
async def test_an_internal_address_literal_is_refused(
    dial: Callable[[str], Any], literal: str, socket: _Socket, resolver: _Resolver
) -> None:
    with pytest.raises(REFUSED):
        await dial(literal)

    assert socket.dialed == []
    assert resolver.asked == []


LOOPBACK_SPELLINGS = ["127.1", "0177.0.0.1", "[::1]", "0x7f.0.0.1", "2130706433", "localhost", "[::ffff:127.0.0.1]"]


@pytest.mark.parametrize("host", LOOPBACK_SPELLINGS)
@pytest.mark.parametrize("dial", CLIENTS.values(), ids=CLIENTS.keys())
async def test_every_loopback_spelling_is_refused_at_connect(
    dial: Callable[[str], Any], host: str, socket: _Socket
) -> None:
    """Through the system resolver, as in production. Some spellings the HTTP library refuses as a URL first."""
    with pytest.raises((*REFUSED, httpx.InvalidURL, httpx2.InvalidURL)):
        await dial(f"http://{host}:8000/mcp")

    assert socket.dialed == []


@pytest.mark.parametrize("address", ["127.0.0.1", "10.0.0.5"])
@pytest.mark.parametrize("dial", CLIENTS.values(), ids=CLIENTS.keys())
async def test_a_name_rebound_after_the_save_time_check_reaches_nothing_internal(
    dial: Callable[[str], Any], address: str, socket: _Socket, resolver: _Resolver, saved_as_public: None
) -> None:
    """Public when the URL was checked, internal when the client connects."""
    await validate_mcp_url(URL, has_authorization_token=True)
    resolver.answer(address)

    with pytest.raises(REFUSED):
        await dial(URL)

    assert socket.dialed == []


@pytest.mark.parametrize("dial", [_stateless, _oauth, _guardrail], ids=["stateless MCP", "MCP OAuth", "guardrail URL"])
@pytest.mark.parametrize("location", [f"https://{HOST}/elsewhere", "http://127.0.0.1:8000/", "https://other.example/"])
async def test_a_redirect_is_not_followed(
    dial: Callable[[str], Any], location: str, socket: _Socket, resolver: _Resolver
) -> None:
    socket.answers = [httpx.Response(307, headers={"location": location})]

    assert await dial(URL) == 307

    assert len(socket.dialed) == 1


async def test_the_managed_client_follows_no_redirect_off_its_origin(socket: _Socket, resolver: _Resolver) -> None:
    socket.answers = [httpx.Response(307, headers={"location": "http://127.0.0.1:8000/mcp"})]

    async with _origin_bound_http_client(URL) as client:
        response = await request_within_origin(client, "POST", URL, json={"method": "tools/list"})

    assert response.status_code == 307
    assert len(socket.dialed) == 1


async def test_a_redirect_the_managed_client_follows_is_checked_and_pinned_again(
    socket: _Socket, resolver: _Resolver
) -> None:
    """The SDK follows a redirect on the same origin; the name is resolved and refused again for that hop."""
    socket.answers = [httpx.Response(307, headers={"location": f"https://{HOST}/mcp/v2"})]
    resolver.then([PUBLIC], ["127.0.0.1"])

    async with _origin_bound_http_client(URL) as client:
        with pytest.raises(REFUSED):
            await request_within_origin(client, "POST", URL, json={"method": "tools/list"})

    assert len(socket.dialed) == 1
    assert resolver.asked == [HOST, HOST]


async def test_each_client_is_built_on_the_owned_endpoint_transport() -> None:
    async with (
        mcp_stateless.build_http_client_factory()(None) as stateless,
        _origin_bound_http_client(URL) as managed,
        _mcp_oauth._http_client(httpx.Timeout(1.0)) as oauth,
        guardrails_module._tenant_endpoint_client() as guardrail,
    ):
        pass

    assert isinstance(stateless._transport, OwnedEndpointHttpx2Transport)  # noqa: SLF001
    assert isinstance(managed._transport, OwnedEndpointHttpx2Transport)  # noqa: SLF001
    assert isinstance(oauth._transport, OwnedEndpointTransport)  # noqa: SLF001
    assert isinstance(guardrail._transport, OwnedEndpointTransport)  # noqa: SLF001
    assert (stateless.follow_redirects, oauth.follow_redirects, guardrail.follow_redirects) == (False, False, False)
    assert stateless.max_redirects == 0


# --------------------------------------------------------------------------- #
# Through the paths that use the clients
# --------------------------------------------------------------------------- #


async def test_an_oauth_token_request_to_a_rebound_name_is_refused(
    socket: _Socket, resolver: _Resolver, saved_as_public: None
) -> None:
    resolver.answer("127.0.0.1")

    with pytest.raises(_mcp_oauth.McpOAuthError):
        await _mcp_oauth.refresh(
            f"https://{HOST}/token", _mcp_oauth.McpOAuthClient(client_id="c1"), refresh_token="r", resource=None
        )

    assert socket.dialed == []


@pytest.mark.parametrize("url", ["https://127.0.0.1/token", "https://10.0.0.5/token"])
async def test_an_oauth_address_inside_is_refused_by_its_check(url: str, socket: _Socket, resolver: _Resolver) -> None:
    with pytest.raises(_mcp_oauth.McpOAuthError, match="safety check"):
        await _mcp_oauth.refresh(url, _mcp_oauth.McpOAuthClient(client_id="c1"), refresh_token="r", resource=None)

    assert socket.dialed == []


async def test_a_caller_guardrail_url_rebound_inside_is_not_evaluated_there(
    socket: _Socket, resolver: _Resolver, saved_as_public: None
) -> None:
    resolver.answer("127.0.0.1")
    entry = GuardrailConfig(profile="pii", url=f"https://{HOST}", mode="block")

    with pytest.raises(GuardrailsNotReachableError):
        await run_input_guardrails([entry], "hello", default_url=None)

    assert socket.dialed == []


@pytest.mark.parametrize("url", ["http://127.0.0.1:8000", "http://localhost:8000", "https://10.0.0.5"])
async def test_a_caller_guardrail_url_inside_is_refused(
    url: str, monkeypatch: pytest.MonkeyPatch, socket: _Socket
) -> None:
    async def loopback(_host: str) -> list[object]:
        return [ipaddress.ip_address("127.0.0.1")]

    monkeypatch.setattr(url_safety, "_resolve_all_async", loopback)
    entry = GuardrailConfig(profile="pii", url=url, mode="block")

    with pytest.raises(UnsafeURLError):
        await run_input_guardrails([entry], "hello", default_url=None)

    assert socket.dialed == []


async def test_the_operator_guardrail_service_is_not_held_to_the_tenant_rule(
    monkeypatch: pytest.MonkeyPatch, socket: _Socket
) -> None:
    """``OTARI_GUARDRAILS_URL`` is the operator's, often a sidecar on this host, so it keeps the plain client."""
    operator: list[httpx.Request] = []

    def answer(request: httpx.Request) -> httpx.Response:
        operator.append(request)
        return httpx.Response(200, json={"result": {"valid": True}})

    real = httpx.AsyncClient

    def client(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs.setdefault("transport", httpx.MockTransport(answer))
        return real(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", client)
    entry = GuardrailConfig(profile="pii", mode="block")

    verdict = await run_input_guardrails([entry], "hello", default_url="http://127.0.0.1:8000")

    assert [str(r.url) for r in operator] == ["http://127.0.0.1:8000/validate"]
    assert socket.dialed == []
    assert verdict.results[0].valid is True
