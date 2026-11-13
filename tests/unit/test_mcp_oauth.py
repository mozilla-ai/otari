"""The wire half of an MCP sign-in: discovery, client registration, PKCE and the token endpoint.

Every URL is an IP literal in a public range, so the safety check each one passes never reaches a
DNS resolver, and the HTTP client is one on a mock transport.
"""

from __future__ import annotations

import asyncio
import base64
import gzip
import json
import zlib
from collections.abc import Callable
from typing import Any
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest

from gateway.services.tools import _mcp_oauth
from gateway.services.tools._mcp_oauth import (
    AUTH_METHOD_BASIC,
    AUTH_METHOD_NONE,
    AUTH_METHOD_POST,
    McpOAuthClient,
    McpOAuthDiscoveryError,
    McpOAuthMetadata,
    McpOAuthRegistrationError,
    McpOAuthTokenError,
)

pytestmark = pytest.mark.asyncio

SERVER = "https://93.184.216.34"
SERVER_URL = f"{SERVER}/mcp"
ISSUER = "https://93.184.216.35"
PRIVATE = "https://10.0.0.5"

Handler = Callable[[httpx.Request], httpx.Response]


class _Wire:
    """A mock transport that answers by URL and keeps every request it saw."""

    def __init__(self) -> None:
        self.routes: dict[tuple[str, str], Handler | httpx.Response] = {}
        self.requests: list[httpx.Request] = []

    def on(self, method: str, url: str, answer: Handler | httpx.Response) -> None:
        self.routes[(method, url)] = answer

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        answer = self.routes.get((request.method, str(request.url)))
        if answer is None:
            return httpx.Response(404)
        return answer(request) if callable(answer) else answer

    def sent_to(self, url: str) -> list[httpx.Request]:
        return [request for request in self.requests if str(request.url) == url]


@pytest.fixture
def wire(monkeypatch: pytest.MonkeyPatch) -> _Wire:
    fake = _Wire()
    monkeypatch.setattr(
        _mcp_oauth,
        "_http_client",
        lambda timeout: httpx.AsyncClient(transport=httpx.MockTransport(fake.handle), follow_redirects=False),
    )
    return fake


def _as_document(**overrides: Any) -> dict[str, Any]:
    document: dict[str, Any] = {
        "issuer": ISSUER,
        "authorization_endpoint": f"{ISSUER}/authorize",
        "token_endpoint": f"{ISSUER}/token",
        "registration_endpoint": f"{ISSUER}/register",
        "revocation_endpoint": f"{ISSUER}/revoke",
        "code_challenge_methods_supported": ["S256"],
        "token_endpoint_auth_methods_supported": ["client_secret_post", "none"],
    }
    document.update(overrides)
    return document


def _publish(wire: _Wire, *, resource: dict[str, Any] | None = None, server: dict[str, Any] | None = None) -> None:
    """A server that challenges with its metadata URL, as the current specification has it."""
    metadata_url = f"{SERVER}/.well-known/oauth-protected-resource/mcp"
    wire.on(
        "POST",
        SERVER_URL,
        httpx.Response(
            401, headers={"WWW-Authenticate": f'Bearer resource_metadata="{metadata_url}", scope="files:read"'}
        ),
    )
    wire.on(
        "GET",
        metadata_url,
        httpx.Response(
            200,
            json=resource
            or {"resource": SERVER_URL, "authorization_servers": [ISSUER], "scopes_supported": ["a", "b"]},
        ),
    )
    wire.on(
        "GET",
        f"{ISSUER}/.well-known/oauth-authorization-server",
        httpx.Response(200, json=server or _as_document()),
    )


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #


async def test_a_bearer_challenge_yields_its_metadata_url_and_scope() -> None:
    header = 'Basic realm="x", Bearer error="invalid_token", resource_metadata="https://a/b", scope="read write"'

    assert _mcp_oauth.parse_bearer_challenge(header) == ("https://a/b", "read write")
    assert _mcp_oauth.parse_bearer_challenge('Basic realm="x"') == (None, None)
    assert _mcp_oauth.parse_bearer_challenge("") == (None, None)


async def test_protected_resource_metadata_is_looked_for_at_the_hint_then_the_path_then_the_root() -> None:
    assert _mcp_oauth.protected_resource_urls("https://a/hint", "https://a/mcp/v1") == [
        "https://a/hint",
        "https://a/.well-known/oauth-protected-resource/mcp/v1",
        "https://a/.well-known/oauth-protected-resource",
    ]
    assert _mcp_oauth.protected_resource_urls(None, "https://a") == ["https://a/.well-known/oauth-protected-resource"]


async def test_authorization_server_metadata_is_looked_for_under_a_path_aware_issuer() -> None:
    assert _mcp_oauth.authorization_server_urls("https://a/tenant/") == [
        "https://a/.well-known/oauth-authorization-server/tenant",
        "https://a/.well-known/openid-configuration/tenant",
        "https://a/tenant/.well-known/openid-configuration",
    ]


async def test_the_code_challenge_is_the_s256_of_the_verifier() -> None:
    # The worked example of RFC 7636, appendix B.
    verifier = "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"

    assert _mcp_oauth.code_challenge(verifier) == "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"
    assert len(_mcp_oauth.new_code_verifier()) == 43
    assert _mcp_oauth.new_code_verifier() != _mcp_oauth.new_code_verifier()


async def test_the_authorization_url_carries_pkce_the_state_and_the_resource() -> None:
    metadata = McpOAuthMetadata(
        authorization_endpoint=f"{ISSUER}/authorize?tenant=1",
        token_endpoint=f"{ISSUER}/token",
        scope="files:read",
        resource=SERVER_URL,
    )

    url = _mcp_oauth.authorization_url(
        metadata, client_id="client", redirect_uri="https://otari.example/cb", state="s", verifier="v"
    )

    query = {key: value[0] for key, value in parse_qs(urlsplit(url).query).items()}
    assert query == {
        "tenant": "1",
        "response_type": "code",
        "client_id": "client",
        "redirect_uri": "https://otari.example/cb",
        "state": "s",
        "code_challenge": _mcp_oauth.code_challenge("v"),
        "code_challenge_method": "S256",
        "scope": "files:read",
        "resource": SERVER_URL,
    }


async def test_a_supplied_client_authenticates_by_what_it_holds_and_the_server_takes() -> None:
    def metadata(*methods: str) -> McpOAuthMetadata:
        return McpOAuthMetadata(authorization_endpoint="a", token_endpoint="t", token_auth_methods=methods)

    assert _mcp_oauth.auth_method_for(None, metadata("client_secret_basic")) == AUTH_METHOD_NONE
    # RFC 8414 section 2: metadata that lists no methods takes client_secret_basic.
    assert _mcp_oauth.auth_method_for("s", metadata()) == AUTH_METHOD_BASIC
    assert _mcp_oauth.auth_method_for("s", metadata("client_secret_basic")) == AUTH_METHOD_BASIC
    assert _mcp_oauth.auth_method_for("s", metadata("client_secret_basic", "client_secret_post")) == AUTH_METHOD_POST


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #


async def test_discovery_follows_the_challenge_to_the_authorization_server(wire: _Wire) -> None:
    _publish(wire)

    metadata = await _mcp_oauth.discover(SERVER_URL)

    assert metadata == McpOAuthMetadata(
        authorization_endpoint=f"{ISSUER}/authorize",
        token_endpoint=f"{ISSUER}/token",
        registration_endpoint=f"{ISSUER}/register",
        revocation_endpoint=f"{ISSUER}/revoke",
        scope="files:read",
        resource=SERVER_URL,
        token_auth_methods=("client_secret_post", "none"),
        issuer=ISSUER,
    )
    assert "Authorization" not in wire.sent_to(SERVER_URL)[0].headers


async def test_discovery_falls_back_to_the_well_known_urls_and_the_published_scopes(wire: _Wire) -> None:
    wire.on("POST", SERVER_URL, httpx.Response(401, headers={"WWW-Authenticate": "Bearer"}))
    wire.on(
        "GET",
        f"{SERVER}/.well-known/oauth-protected-resource",
        httpx.Response(200, json={"authorization_servers": [f"{ISSUER}/tenant"], "scopes_supported": ["a", "b"]}),
    )
    wire.on(
        "GET",
        f"{ISSUER}/.well-known/openid-configuration/tenant",
        httpx.Response(
            200, json=_as_document(issuer=f"{ISSUER}/tenant", registration_endpoint=None, revocation_endpoint=None)
        ),
    )

    metadata = await _mcp_oauth.discover(SERVER_URL)

    assert metadata.scope == "a b"
    assert metadata.resource == SERVER_URL, "with no published resource the server's own URL names it"
    assert metadata.registration_endpoint is None


async def test_a_server_with_no_resource_metadata_is_its_own_authorization_server(wire: _Wire) -> None:
    wire.on(
        "GET",
        f"{SERVER}/.well-known/oauth-authorization-server",
        httpx.Response(
            200,
            json=_as_document(
                issuer=SERVER, authorization_endpoint=f"{SERVER}/authorize", token_endpoint=f"{SERVER}/token"
            ),
        ),
    )

    metadata = await _mcp_oauth.discover(SERVER_URL)

    assert metadata.token_endpoint == f"{SERVER}/token"
    assert metadata.resource is None


async def test_that_fallback_refuses_metadata_naming_another_origin_as_issuer(wire: _Wire) -> None:
    wire.on("GET", f"{SERVER}/.well-known/oauth-authorization-server", httpx.Response(200, json=_as_document()))

    with pytest.raises(McpOAuthDiscoveryError, match="another origin"):
        await _mcp_oauth.discover(SERVER_URL)


async def test_discovery_does_not_guess_past_a_failing_metadata_endpoint(wire: _Wire) -> None:
    wire.on("GET", f"{SERVER}/.well-known/oauth-protected-resource/mcp", httpx.Response(503))
    wire.on("GET", f"{SERVER}/.well-known/oauth-authorization-server", httpx.Response(200, json=_as_document()))

    with pytest.raises(McpOAuthDiscoveryError, match="could not be read"):
        await _mcp_oauth.discover(SERVER_URL)


async def test_resource_metadata_describing_another_resource_is_refused(wire: _Wire) -> None:
    _publish(wire, resource={"resource": f"{ISSUER}/mcp", "authorization_servers": [ISSUER]})

    with pytest.raises(McpOAuthDiscoveryError, match="another resource"):
        await _mcp_oauth.discover(SERVER_URL)


@pytest.mark.parametrize("field", ["authorization_endpoint", "token_endpoint", "registration_endpoint"])
async def test_an_endpoint_on_a_private_address_is_refused_before_anything_is_sent_to_it(
    wire: _Wire, field: str
) -> None:
    _publish(wire, server=_as_document(**{field: f"{PRIVATE}/x"}))

    with pytest.raises(McpOAuthDiscoveryError, match="safety check") as raised:
        await _mcp_oauth.discover(SERVER_URL)

    assert "10.0.0.5" not in str(raised.value), "the address stays in the log"
    assert not [request for request in wire.requests if request.url.host == "10.0.0.5"]


async def test_an_authorization_server_on_a_private_address_is_never_asked(wire: _Wire) -> None:
    _publish(wire, resource={"resource": SERVER_URL, "authorization_servers": [PRIVATE]})

    with pytest.raises(McpOAuthDiscoveryError, match="safety check"):
        await _mcp_oauth.discover(SERVER_URL)

    assert not [request for request in wire.requests if request.url.host == "10.0.0.5"]


async def test_a_plain_http_endpoint_is_refused(wire: _Wire) -> None:
    _publish(wire, server=_as_document(token_endpoint="http://93.184.216.35/token"))

    with pytest.raises(McpOAuthDiscoveryError, match="safety check"):
        await _mcp_oauth.discover(SERVER_URL)


async def test_metadata_naming_another_issuer_is_refused(wire: _Wire) -> None:
    _publish(wire, server=_as_document(issuer="https://93.184.216.99"))

    with pytest.raises(McpOAuthDiscoveryError, match="own issuer"):
        await _mcp_oauth.discover(SERVER_URL)


@pytest.mark.parametrize(
    "resource",
    [
        {"resource": SERVER_URL, "authorization_servers": ["https://[bad"]},
        {"resource": "https://[bad", "authorization_servers": [ISSUER]},
    ],
)
async def test_a_url_no_parser_accepts_is_a_discovery_failure_and_not_a_crash(
    wire: _Wire, resource: dict[str, Any]
) -> None:
    _publish(wire, resource=resource)

    with pytest.raises(McpOAuthDiscoveryError):
        await _mcp_oauth.discover(SERVER_URL)


async def test_metadata_with_no_issuer_is_refused_as_the_fallback_path_refuses_it(wire: _Wire) -> None:
    document = _as_document()
    del document["issuer"]
    _publish(wire, server=document)

    with pytest.raises(McpOAuthDiscoveryError, match="own issuer"):
        await _mcp_oauth.discover(SERVER_URL)


@pytest.mark.parametrize("methods", [None, [], ["plain"]], ids=["absent", "empty", "plain-only"])
async def test_an_authorization_server_that_does_not_say_s256_is_refused_as_a_setup_problem(
    wire: _Wire, methods: list[str] | None
) -> None:
    """The MCP authorization specification: a client refuses a server that does not advertise S256."""
    document = _as_document(code_challenge_methods_supported=methods)
    if methods is None:
        del document["code_challenge_methods_supported"]
    _publish(wire, server=document)

    with pytest.raises(_mcp_oauth.McpOAuthPkceUnsupportedError, match="S256"):
        await _mcp_oauth.discover(SERVER_URL)


async def test_an_authorization_server_that_sends_iss_is_recorded_as_doing_so(wire: _Wire) -> None:
    _publish(wire, server=_as_document(authorization_response_iss_parameter_supported=True))

    metadata = await _mcp_oauth.discover(SERVER_URL)

    assert (metadata.issuer, metadata.sends_iss) == (ISSUER, True)


async def test_a_token_endpoint_that_does_not_answer_in_time_is_a_failure_for_now(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def slow(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(5)
        return httpx.Response(200, json={"access_token": "late"})

    monkeypatch.setattr(
        _mcp_oauth,
        "_http_client",
        lambda timeout: httpx.AsyncClient(transport=httpx.MockTransport(slow), follow_redirects=False),
    )

    with pytest.raises(McpOAuthTokenError, match="in time") as raised:
        await _mcp_oauth.refresh(
            f"{ISSUER}/token", McpOAuthClient(client_id="c1"), refresh_token="rt", resource=None, deadline=0.05
        )

    assert raised.value.is_permanent is False


async def test_an_answer_read_inside_the_deadline_is_returned_even_if_closing_runs_past_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The deadline stops at the read: a rotated refresh token that arrived is never thrown away."""

    class SlowToClose(httpx.AsyncByteStream):
        async def __aiter__(self) -> Any:
            yield b'{"access_token": "at", "refresh_token": "rotated"}'

        async def aclose(self) -> None:
            await asyncio.sleep(0.2)

    monkeypatch.setattr(
        _mcp_oauth,
        "_http_client",
        lambda timeout: httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, stream=SlowToClose())),
            follow_redirects=False,
        ),
    )

    tokens = await _mcp_oauth.refresh(
        f"{ISSUER}/token", McpOAuthClient(client_id="c1"), refresh_token="rt", resource=None, deadline=0.1
    )

    assert (tokens.access_token, tokens.refresh_token) == ("at", "rotated")


@pytest.mark.parametrize("encoding", ["gzip", "deflate"])
async def test_a_compressed_answer_is_read_though_identity_was_asked_for(wire: _Wire, encoding: str) -> None:
    """A token endpoint that compresses anyway must not lose a rotated refresh token to unreadable JSON."""
    raw = json.dumps({"access_token": "at", "refresh_token": "rotated"}).encode()
    body = gzip.compress(raw) if encoding == "gzip" else zlib.compress(raw)
    wire.on("POST", f"{ISSUER}/token", httpx.Response(200, content=body, headers={"Content-Encoding": encoding}))

    tokens = await _mcp_oauth.refresh(
        f"{ISSUER}/token", McpOAuthClient(client_id="c1"), refresh_token="rt", resource=None
    )

    assert (tokens.access_token, tokens.refresh_token) == ("at", "rotated")
    assert wire.requests[0].headers["Accept-Encoding"] == "identity"


async def test_an_answer_in_an_unknown_encoding_fails_for_now_and_keeps_the_old_token(wire: _Wire) -> None:
    wire.on("POST", f"{ISSUER}/token", httpx.Response(200, content=b"\x00\x01", headers={"Content-Encoding": "br"}))

    with pytest.raises(McpOAuthTokenError, match="encoding") as raised:
        await _mcp_oauth.refresh(f"{ISSUER}/token", McpOAuthClient(client_id="c1"), refresh_token="rt", resource=None)

    assert raised.value.is_permanent is False


async def test_a_compressed_answer_is_held_to_the_size_bound(wire: _Wire) -> None:
    bomb = gzip.compress(b"{" + b" " * (_mcp_oauth._MAX_RESPONSE_BYTES * 2) + b"}")
    wire.on("POST", f"{ISSUER}/token", httpx.Response(200, content=bomb, headers={"Content-Encoding": "gzip"}))

    with pytest.raises(McpOAuthTokenError, match="more than"):
        await _mcp_oauth.refresh(f"{ISSUER}/token", McpOAuthClient(client_id="c1"), refresh_token="rt", resource=None)


async def test_the_address_check_counts_against_the_deadline(monkeypatch: pytest.MonkeyPatch, wire: _Wire) -> None:
    """A slow resolver must not hold a renewal's row lock past its deadline."""

    async def slow_check(url: str, *, has_authorization_token: bool, label: str = "MCP server") -> None:
        await asyncio.sleep(5)

    monkeypatch.setattr(_mcp_oauth, "validate_mcp_url", slow_check)

    with pytest.raises(McpOAuthTokenError, match="in time"):
        await _mcp_oauth.refresh(
            f"{ISSUER}/token", McpOAuthClient(client_id="c1"), refresh_token="rt", resource=None, deadline=0.05
        )

    assert wire.requests == []


async def test_an_authorization_server_without_s256_is_refused(wire: _Wire) -> None:
    _publish(wire, server=_as_document(code_challenge_methods_supported=["plain"]))

    with pytest.raises(McpOAuthDiscoveryError, match="S256"):
        await _mcp_oauth.discover(SERVER_URL)


async def test_a_redirect_is_not_followed(wire: _Wire) -> None:
    wire.on(
        "GET",
        f"{SERVER}/.well-known/oauth-protected-resource/mcp",
        httpx.Response(302, headers={"Location": f"{PRIVATE}/meta"}),
    )
    wire.on("GET", f"{SERVER}/.well-known/oauth-authorization-server", httpx.Response(404))

    with pytest.raises(McpOAuthDiscoveryError):
        await _mcp_oauth.discover(SERVER_URL)

    assert not [request for request in wire.requests if request.url.host == "10.0.0.5"]


async def test_an_oversized_answer_is_refused(wire: _Wire) -> None:
    wire.on(
        "GET",
        f"{SERVER}/.well-known/oauth-protected-resource/mcp",
        httpx.Response(200, content=b"x" * (_mcp_oauth._MAX_RESPONSE_BYTES + 1)),
    )

    with pytest.raises(McpOAuthDiscoveryError, match="more than"):
        await _mcp_oauth.discover(SERVER_URL)


# --------------------------------------------------------------------------- #
# Client registration
# --------------------------------------------------------------------------- #


async def test_registration_asks_for_a_confidential_client_with_the_redirect_uri(wire: _Wire) -> None:
    wire.on("POST", f"{ISSUER}/register", httpx.Response(201, json={"client_id": "c1", "client_secret": "s1"}))

    client = await _mcp_oauth.register_client(f"{ISSUER}/register", "https://otari.example/cb")

    assert client == McpOAuthClient(client_id="c1", client_secret="s1", auth_method=AUTH_METHOD_POST)
    sent = json.loads(wire.requests[0].content)
    assert sent["redirect_uris"] == ["https://otari.example/cb"]
    assert sent["grant_types"] == ["authorization_code", "refresh_token"]
    assert sent["client_name"] == "Otari"


async def test_registration_retries_as_a_public_client_when_a_secret_is_refused(wire: _Wire) -> None:
    def answer(request: httpx.Request) -> httpx.Response:
        if json.loads(request.content)["token_endpoint_auth_method"] == "none":
            return httpx.Response(201, json={"client_id": "public"})
        return httpx.Response(400, json={"error": "invalid_client_metadata"})

    wire.on("POST", f"{ISSUER}/register", answer)

    client = await _mcp_oauth.register_client(f"{ISSUER}/register", "https://otari.example/cb")

    assert client == McpOAuthClient(client_id="public", client_secret=None, auth_method=AUTH_METHOD_NONE)
    assert len(wire.requests) == 2


@pytest.mark.parametrize(
    "refusal",
    [httpx.Response(403), httpx.Response(400, json={"error": "invalid_redirect_uri"}), httpx.Response(201, json={})],
)
async def test_a_refused_registration_is_not_retried(wire: _Wire, refusal: httpx.Response) -> None:
    wire.on("POST", f"{ISSUER}/register", refusal)

    with pytest.raises(McpOAuthRegistrationError):
        await _mcp_oauth.register_client(f"{ISSUER}/register", "https://otari.example/cb")

    assert len(wire.requests) == 1


async def test_a_server_stored_with_a_trailing_slash_is_discovered_as_railway_publishes_itself(wire: _Wire) -> None:
    """Railway's shape: a root URL stored with its slash, a resource named without one, another origin to sign in at."""
    url = f"{SERVER}/"
    metadata_url = f"{SERVER}/.well-known/oauth-protected-resource"
    challenge = f'Bearer realm="mcp", resource_metadata="{metadata_url}", scope="workspace:member"'
    wire.on("POST", url, httpx.Response(401, headers={"WWW-Authenticate": challenge}))
    wire.on("GET", metadata_url, httpx.Response(200, json={"resource": SERVER, "authorization_servers": [ISSUER]}))
    wire.on(
        "GET",
        f"{ISSUER}/.well-known/oauth-authorization-server",
        httpx.Response(200, json=_as_document(client_id_metadata_document_supported=True)),
    )

    metadata = await _mcp_oauth.discover(url)

    assert metadata.registration_endpoint == f"{ISSUER}/register", "discovery was never what failed"
    assert metadata.takes_client_metadata is True
    assert (metadata.resource, metadata.scope) == (SERVER, "workspace:member")


async def test_a_refused_registration_keeps_the_status_and_code_it_was_refused_with(wire: _Wire) -> None:
    refusal = {"error": "access_denied", "error_description": "Dynamic client registration is disabled"}
    wire.on("POST", f"{ISSUER}/register", httpx.Response(403, json=refusal))

    with pytest.raises(McpOAuthRegistrationError) as refused:
        await _mcp_oauth.register_client(f"{ISSUER}/register", "https://otari.example/cb")

    assert (refused.value.status_code, refused.value.error_code) == (403, "access_denied")
    assert "disabled" not in str(refused.value), "what the remote party wrote is never repeated"


# --------------------------------------------------------------------------- #
# The token endpoint
# --------------------------------------------------------------------------- #


def _form(request: httpx.Request) -> dict[str, str]:
    return {key: value[0] for key, value in parse_qs(request.content.decode()).items()}


async def test_a_code_is_exchanged_with_its_verifier_and_resource(wire: _Wire) -> None:
    wire.on(
        "POST",
        f"{ISSUER}/token",
        httpx.Response(
            200,
            json={"access_token": "at", "token_type": "bearer", "refresh_token": "rt", "expires_in": "3600"},
        ),
    )

    tokens = await _mcp_oauth.exchange_code(
        f"{ISSUER}/token",
        McpOAuthClient(client_id="c1", client_secret="s1", auth_method=AUTH_METHOD_POST),
        code="code",
        redirect_uri="https://otari.example/cb",
        verifier="verifier",
        resource=SERVER_URL,
    )

    assert (tokens.access_token, tokens.refresh_token, tokens.expires_in) == ("at", "rt", 3600)
    assert _form(wire.requests[0]) == {
        "grant_type": "authorization_code",
        "code": "code",
        "redirect_uri": "https://otari.example/cb",
        "code_verifier": "verifier",
        "resource": SERVER_URL,
        "client_id": "c1",
        "client_secret": "s1",
    }


@pytest.mark.parametrize(
    ("stated", "kept"),
    [
        ("Infinity", None),
        ("1e400", None),
        ("99999999999999", 2**32),
        ("-5", None),
        ("0", None),
        ('"120"', 120),
        ("true", None),
        ('"\u00b2"', None),
        ('"' + "9" * 5000 + '"', None),
        ('"' + "0" * 70 + '120"', None),
        ('"' + "0" * 60 + '120"', 120),
        ("1" + "0" * 400, 2**32),
    ],
)
async def test_a_lifetime_no_clock_can_hold_does_not_crash(wire: _Wire, stated: str, kept: int | None) -> None:
    body = f'{{"access_token": "at", "expires_in": {stated}}}'.encode()
    wire.on("POST", f"{ISSUER}/token", httpx.Response(200, content=body))

    tokens = await _mcp_oauth.refresh(
        f"{ISSUER}/token", McpOAuthClient(client_id="c1"), refresh_token="rt", resource=None
    )

    assert tokens.expires_in == kept


async def test_registration_keeps_when_the_client_secret_expires(wire: _Wire) -> None:
    wire.on(
        "POST",
        f"{ISSUER}/register",
        httpx.Response(201, json={"client_id": "c1", "client_secret": "s1", "client_secret_expires_at": 1900000000}),
    )
    assert (await _mcp_oauth.register_client(f"{ISSUER}/register", "https://o/cb")).secret_expires_at == 1900000000

    wire.on(
        "POST",
        f"{ISSUER}/register",
        httpx.Response(201, json={"client_id": "c1", "client_secret": "s1", "client_secret_expires_at": 0}),
    )
    assert (await _mcp_oauth.register_client(f"{ISSUER}/register", "https://o/cb")).secret_expires_at is None


async def test_a_basic_client_sends_its_secret_in_the_header_and_a_public_one_sends_none(wire: _Wire) -> None:
    wire.on("POST", f"{ISSUER}/token", httpx.Response(200, json={"access_token": "at"}))

    await _mcp_oauth.refresh(
        f"{ISSUER}/token",
        McpOAuthClient(client_id="c1", client_secret="s1", auth_method=AUTH_METHOD_BASIC),
        refresh_token="rt",
        resource=None,
    )
    await _mcp_oauth.refresh(
        f"{ISSUER}/token",
        McpOAuthClient(client_id="c1", client_secret=None, auth_method=AUTH_METHOD_NONE),
        refresh_token="rt",
        resource=None,
    )

    basic, public = wire.requests
    assert basic.headers["Authorization"].startswith("Basic ")
    assert "client_secret" not in _form(basic) and "client_id" not in _form(basic)
    assert _form(public) == {"grant_type": "refresh_token", "refresh_token": "rt", "client_id": "c1"}


async def test_a_basic_client_form_encodes_its_id_and_secret_before_the_header(wire: _Wire) -> None:
    """RFC 6749 section 2.3.1: a ``+``, ``%`` or ``:`` in either half must not change what the server reads."""
    wire.on("POST", f"{ISSUER}/token", httpx.Response(200, json={"access_token": "at"}))
    wire.on("POST", f"{ISSUER}/revoke", httpx.Response(200))
    client = McpOAuthClient(client_id="app:1 x", client_secret="s+e%c:r/t", auth_method=AUTH_METHOD_BASIC)

    await _mcp_oauth.refresh(f"{ISSUER}/token", client, refresh_token="rt", resource=None)
    await _mcp_oauth.revoke(f"{ISSUER}/revoke", client, token="rt")

    expected = "Basic " + base64.b64encode(b"app%3A1+x:s%2Be%25c%3Ar%2Ft").decode()
    assert [request.headers["Authorization"] for request in wire.requests] == [expected, expected]


async def test_discovery_registration_and_the_code_exchange_have_a_total_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A server that sends its answer a few bytes at a time cannot hold a sign-in open past the deadline."""

    class Drip(httpx.AsyncByteStream):
        async def __aiter__(self) -> Any:
            while True:
                await asyncio.sleep(0.01)
                yield b" "

    monkeypatch.setattr(_mcp_oauth, "_REQUEST_DEADLINE_SECONDS", 0.1)
    monkeypatch.setattr(
        _mcp_oauth,
        "_http_client",
        lambda timeout: httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, stream=Drip())),
            follow_redirects=False,
        ),
    )

    with pytest.raises(McpOAuthDiscoveryError, match="in time"):
        await _mcp_oauth.discover(SERVER_URL)
    with pytest.raises(McpOAuthDiscoveryError, match="in time"):
        await _mcp_oauth.register_client(f"{ISSUER}/register", "https://o/cb")
    with pytest.raises(McpOAuthTokenError, match="in time"):
        await _mcp_oauth.exchange_code(
            f"{ISSUER}/token",
            McpOAuthClient(client_id="c1"),
            code="c",
            redirect_uri="https://o/cb",
            verifier="v",
            resource=None,
        )


async def test_a_refresh_keeps_the_token_it_sent_when_none_is_issued(wire: _Wire) -> None:
    wire.on("POST", f"{ISSUER}/token", httpx.Response(200, json={"access_token": "new"}))
    client = McpOAuthClient(client_id="c1")

    kept = await _mcp_oauth.refresh(f"{ISSUER}/token", client, refresh_token="old", resource=None)
    wire.on("POST", f"{ISSUER}/token", httpx.Response(200, json={"access_token": "new", "refresh_token": "rotated"}))
    rotated = await _mcp_oauth.refresh(f"{ISSUER}/token", client, refresh_token="old", resource=None)

    assert (kept.access_token, kept.refresh_token, kept.expires_in) == ("new", "old", None)
    assert rotated.refresh_token == "rotated"


@pytest.mark.parametrize(
    ("answer", "permanent"),
    [
        (httpx.Response(400, json={"error": "invalid_grant", "error_description": "secret detail"}), True),
        # A refused client, or a bare 401 from whatever sits in front, says nothing about the grant.
        (httpx.Response(401), False),
        (httpx.Response(401, json={"error": "invalid_client"}), False),
        (httpx.Response(400, json={"error": "temporarily_unavailable"}), False),
        (httpx.Response(503), False),
        (httpx.Response(429), False),
    ],
)
async def test_a_refused_refresh_says_whether_the_grant_is_gone(
    wire: _Wire, answer: httpx.Response, permanent: bool
) -> None:
    wire.on("POST", f"{ISSUER}/token", answer)

    with pytest.raises(McpOAuthTokenError) as raised:
        await _mcp_oauth.refresh(f"{ISSUER}/token", McpOAuthClient(client_id="c1"), refresh_token="rt", resource=None)

    assert raised.value.is_permanent is permanent
    assert raised.value.refuses_client is (raised.value.error_code == "invalid_client")
    assert "secret detail" not in str(raised.value)


async def test_an_unreachable_token_endpoint_is_not_a_dead_grant(wire: _Wire) -> None:
    def fail(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route")

    wire.on("POST", f"{ISSUER}/token", fail)

    with pytest.raises(McpOAuthTokenError) as raised:
        await _mcp_oauth.refresh(f"{ISSUER}/token", McpOAuthClient(client_id="c1"), refresh_token="rt", resource=None)

    assert raised.value.is_permanent is False


@pytest.mark.parametrize("body", [{"token_type": "bearer"}, {"access_token": "at", "token_type": "mac"}, None])
async def test_an_answer_without_a_bearer_token_is_refused(wire: _Wire, body: dict[str, str] | None) -> None:
    wire.on("POST", f"{ISSUER}/token", httpx.Response(200, json=body) if body else httpx.Response(200, text="ok"))

    with pytest.raises(McpOAuthTokenError, match="no bearer token"):
        await _mcp_oauth.exchange_code(
            f"{ISSUER}/token", McpOAuthClient(client_id="c1"), code="c", redirect_uri="r", verifier="v", resource=None
        )


async def test_a_token_endpoint_on_a_private_address_is_never_sent_a_credential(wire: _Wire) -> None:
    with pytest.raises(McpOAuthTokenError):
        await _mcp_oauth.refresh(f"{PRIVATE}/token", McpOAuthClient(client_id="c1"), refresh_token="rt", resource=None)

    assert wire.requests == []


async def test_revocation_is_best_effort(wire: _Wire) -> None:
    client = McpOAuthClient(client_id="c1", client_secret="s1", auth_method=AUTH_METHOD_POST)
    wire.on("POST", f"{ISSUER}/revoke", httpx.Response(200))

    assert await _mcp_oauth.revoke(f"{ISSUER}/revoke", client, token="rt") is True
    assert _form(wire.requests[0]) == {"token": "rt", "client_id": "c1", "client_secret": "s1"}
    wire.on("POST", f"{ISSUER}/revoke", httpx.Response(500))
    assert await _mcp_oauth.revoke(f"{ISSUER}/revoke", client, token="rt") is False
    assert await _mcp_oauth.revoke(f"{PRIVATE}/revoke", client, token="rt") is False
