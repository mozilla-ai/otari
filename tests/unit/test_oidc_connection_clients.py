"""Build the apron-auth client an organization's own OIDC connection signs in through.

Discovery is never exercised here: ``metadata`` is handed in already resolved, the shape
``_connection_service.metadata_for_sign_in`` serves. Nothing here reaches the network: a
callback exchange is exercised only up to the check it is testing, by monkeypatching
``apron_auth.OAuthClient.exchange_code`` the way ``tests/unit/test_oauth_service.py`` and
``tests/integration/test_oauth_api.py`` already patch apron-auth's client to test a code
exchange's outcome without one.
"""

import json
import uuid
from base64 import urlsafe_b64encode
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from typing import Any, cast

import httpx
import pytest
from apron_auth import OAuthClient, ServerMetadata, StoredStateClient
from apron_auth.errors import IssuerValidationError, StateError
from apron_auth.models import AuthorizationCallback
from apron_auth.models import OAuthPendingState as PendingState
from sqlalchemy import Select
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.exceptions.identity_exceptions import SsoNotAddressableError
from gateway.models.identity import OIDCConnection
from gateway.repositories.identity import OAuthPendingStateRepository
from gateway.services.identity import _connection_clients
from gateway.services.identity._connection_clients import (
    _provider_config_for,
    client_for,
    identity_handler_for,
    pending_flow_provider,
)
from gateway.services.identity._discovery import DiscoveredProvider
from gateway.services.secret_box import encrypt_secret, generate_secret_key
from gateway.services.url_safety import UnsafeURLError
from gateway.services.web_retrieval_network import (
    MAX_FEDERATION_FETCH_BYTES,
    BoundedFederationTransport,
    FederationResponseTooLargeError,
    RetrievalTargetError,
)

_ISSUER = "https://sso.example.test"
_REDIRECT_URI = "https://otari.example.com/auth/sso/callback"


@pytest.fixture(autouse=True)
def _secret_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Decrypting the connection's stored client secret needs a key configured."""
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())


@pytest.fixture(autouse=True)
def _clear_handler_cache() -> None:
    """Each test starts with no memoized handler, so one test's cache cannot answer for another's."""
    _connection_clients._handler_cache.clear()


def _server_metadata(*, issuer: str = _ISSUER, iss_parameter_supported: bool = True) -> ServerMetadata:
    return ServerMetadata(
        authorize_url=f"{issuer}/authorize",
        token_url=f"{issuer}/token",
        jwks_url=f"{issuer}/jwks",
        userinfo_url=f"{issuer}/userinfo",
        issuer=issuer,
        iss_parameter_supported=iss_parameter_supported,
        code_challenge_methods=["S256"],
        token_endpoint_auth_methods=["client_secret_basic"],
    )


def _metadata(
    *, issuer: str = _ISSUER, discovered_at: datetime | None = None, iss_parameter_supported: bool = True
) -> DiscoveredProvider:
    return DiscoveredProvider(
        resolved_issuer=issuer,
        document=_server_metadata(issuer=issuer, iss_parameter_supported=iss_parameter_supported).model_dump(
            mode="json"
        ),
        discovered_at=discovered_at or datetime.now(UTC),
    )


class _StubTransport(httpx.AsyncBaseTransport):
    """A classic transport answering every request through a caller-supplied handler."""

    def __init__(self, handler: Callable[[httpx.Request], Awaitable[httpx.Response]]) -> None:
        self._handler = handler

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return await self._handler(request)

    async def aclose(self) -> None:
        return None


def _unsigned_id_token(*, nonce: str, issuer: str, client_id: str) -> str:
    """An ID token carrying the claims apron-auth's exchange checks; its signature is never verified here."""
    claims = {
        "iss": issuer,
        "aud": client_id,
        "sub": "a-subject",
        "nonce": nonce,
        "exp": int((datetime.now(UTC) + timedelta(hours=1)).timestamp()),
    }

    def segment(value: dict[str, Any]) -> str:
        return urlsafe_b64encode(json.dumps(value).encode()).rstrip(b"=").decode()

    return f"{segment({'alg': 'RS256'})}.{segment(claims)}.signature"


def _connection(*, client_id: str = "a-client-id", client_secret: str = "a-client-secret") -> OIDCConnection:
    return OIDCConnection(
        id=uuid.uuid4(),
        organization_id=uuid.uuid4(),
        issuer=_ISSUER,
        resolved_issuer=_ISSUER,
        client_id=client_id,
        encrypted_client_secret=encrypt_secret(client_secret),
        client_secret_last4=client_secret[-4:],
        display_name="Acme SSO",
        discovery_document={},
        discovered_at=datetime.now(UTC),
    )


class _UntouchedSession:
    """An ``AsyncSession`` stand-in that fails the test if the pending-state store is read."""

    async def execute(self, *_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("the pending-state store must not be read before the issuer is validated")

    def add(self, *_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("the pending-state store must not be written before the issuer is validated")

    async def flush(self) -> None:
        raise AssertionError("the pending-state store must not be flushed before the issuer is validated")


def _untouched_pending() -> OAuthPendingStateRepository:
    """A pending-state repository whose session refuses every call, for the paths that take none."""
    return OAuthPendingStateRepository(cast("AsyncSession", _UntouchedSession()))


class _SelectOnlySession:
    """Records every statement it is asked to execute, and answers with a fixed scalar."""

    def __init__(self, scalar: str | None) -> None:
        self._scalar = scalar
        self.statements: list[Any] = []

    async def execute(self, statement: Any, *_args: Any, **_kwargs: Any) -> Any:
        self.statements.append(statement)

        class _Result:
            def scalar_one_or_none(_self) -> str | None:  # noqa: N805
                return self._scalar

        return _Result()


def test_the_handler_is_reused_for_one_issuer_and_discovery_time() -> None:
    metadata = _metadata()

    first = identity_handler_for(metadata, allow_private_hosts=False)
    second = identity_handler_for(metadata, allow_private_hosts=False)

    assert first is second


def test_a_refreshed_discovery_produces_a_new_handler() -> None:
    first_metadata = _metadata(discovered_at=datetime.now(UTC))
    refreshed_metadata = _metadata(discovered_at=first_metadata.discovered_at + timedelta(seconds=1))

    first = identity_handler_for(first_metadata, allow_private_hosts=False)
    second = identity_handler_for(refreshed_metadata, allow_private_hosts=False)

    assert first is not second


def test_the_cache_drops_the_least_recently_used_past_its_bound() -> None:
    base = datetime.now(UTC)
    evicted_metadata = _metadata(discovered_at=base)
    evicted_handler = identity_handler_for(evicted_metadata, allow_private_hosts=False)

    for offset in range(1, _connection_clients._HANDLER_CACHE_MAX + 1):
        identity_handler_for(_metadata(discovered_at=base + timedelta(seconds=offset)), allow_private_hosts=False)

    rebuilt = identity_handler_for(evicted_metadata, allow_private_hosts=False)

    assert rebuilt is not evicted_handler
    assert len(_connection_clients._handler_cache) == _connection_clients._HANDLER_CACHE_MAX


@pytest.mark.asyncio
async def test_pending_flow_provider_does_not_consume_the_row() -> None:
    session = _SelectOnlySession("oidc:some-connection-id")

    result = await pending_flow_provider(OAuthPendingStateRepository(cast("AsyncSession", session)), "a-state")

    assert result == "oidc:some-connection-id"
    assert len(session.statements) == 1
    assert isinstance(session.statements[0], Select)


@pytest.mark.asyncio
async def test_pending_flow_provider_answers_none_for_an_unknown_state() -> None:
    session = _SelectOnlySession(None)

    result = await pending_flow_provider(OAuthPendingStateRepository(cast("AsyncSession", session)), "a-state")

    assert result is None


def test_the_built_config_requires_iss_when_the_provider_advertises_the_parameter() -> None:
    config = _provider_config_for(_connection(), _metadata(iss_parameter_supported=True), redirect_uri=_REDIRECT_URI)

    assert config.issuer == _ISSUER
    assert config.require_iss is True


def test_the_built_config_does_not_require_iss_when_the_provider_does_not_advertise_the_parameter() -> None:
    config = _provider_config_for(_connection(), _metadata(iss_parameter_supported=False), redirect_uri=_REDIRECT_URI)

    assert config.issuer == _ISSUER
    assert config.require_iss is False


@pytest.mark.asyncio
async def test_a_callback_naming_another_issuer_is_refused_before_any_state_is_read() -> None:
    client = client_for(
        _connection(), _metadata(), pending=_untouched_pending(), redirect_uri=_REDIRECT_URI, allow_private_hosts=False
    )
    callback = AuthorizationCallback(code="a-code", state="a-state", iss="https://attacker.example")

    with pytest.raises(IssuerValidationError):
        await client.exchange_code(callback, browser_token="a-browser-token")


@pytest.mark.asyncio
async def test_a_callback_naming_the_right_issuer_passes_the_issuer_check() -> None:
    client = client_for(
        _connection(), _metadata(), pending=_untouched_pending(), redirect_uri=_REDIRECT_URI, allow_private_hosts=False
    )
    callback = AuthorizationCallback(code="a-code", state="a-state", iss=_ISSUER)

    # No browser token is presented, so the flow is refused for that, never for its issuer: proof
    # that the issuer check already passed rather than raising IssuerValidationError.
    with pytest.raises(StateError):
        await client.exchange_code(callback, browser_token=None)


@pytest.mark.asyncio
async def test_a_callback_naming_the_right_issuer_with_no_iss_at_all_is_refused_when_the_provider_advertises_it() -> (
    None
):
    client = client_for(
        _connection(),
        _metadata(iss_parameter_supported=True),
        pending=_untouched_pending(),
        redirect_uri=_REDIRECT_URI,
        allow_private_hosts=False,
    )
    callback = AuthorizationCallback(code="a-code", state="a-state")

    with pytest.raises(IssuerValidationError):
        await client.exchange_code(callback, browser_token="a-browser-token")


def test_client_for_returns_a_stored_state_client() -> None:
    client = client_for(
        _connection(), _metadata(), pending=_untouched_pending(), redirect_uri=_REDIRECT_URI, allow_private_hosts=False
    )

    assert isinstance(client, StoredStateClient)


def test_client_for_uses_the_memoized_handler() -> None:
    metadata = _metadata()
    handler = identity_handler_for(metadata, allow_private_hosts=False)

    client = client_for(
        _connection(), metadata, pending=_untouched_pending(), redirect_uri=_REDIRECT_URI, allow_private_hosts=False
    )

    assert client._identity_handler is handler


def test_a_deployment_with_no_https_address_is_told_it_is_the_one_to_fix() -> None:
    with pytest.raises(SsoNotAddressableError):
        _connection_clients.client_for(
            _connection(),
            _metadata(),
            db=cast("Any", None),
            redirect_uri="http://otari.example/auth/sso/callback",
            allow_private_hosts=False,
        )


def test_the_transport_factory_checks_and_bounds_every_url() -> None:
    """One transport for every URL the library fetches, the token endpoint included."""
    factory = _connection_clients._transport_factory_for(allow_private_hosts=False)

    for url in (f"{_ISSUER}/token", f"{_ISSUER}/userinfo", f"{_ISSUER}/jwks"):
        assert isinstance(factory(url), BoundedFederationTransport)


@pytest.mark.asyncio
async def test_a_fetch_past_the_byte_ceiling_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """The signing key set and the userinfo document are capped, not the discovery document alone.

    Exercises the factory's own product against an oversized body rather than asserting the
    wrapper's type, so removing the cap fails this test instead of passing it.
    """
    oversized = b" " * (MAX_FEDERATION_FETCH_BYTES + 1)

    async def respond(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=oversized)

    monkeypatch.setattr(_connection_clients, "CheckedFederationTransport", lambda **_kwargs: _StubTransport(respond))
    factory = _connection_clients._transport_factory_for(allow_private_hosts=False)

    async with httpx.AsyncClient(transport=factory(f"{_ISSUER}/userinfo")) as client:
        with pytest.raises(FederationResponseTooLargeError):
            await client.get(f"{_ISSUER}/userinfo")


@pytest.mark.asyncio
async def test_a_real_exchange_through_the_token_url_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = _connection()

    async def handler(_request: httpx.Request) -> httpx.Response:
        id_token = _unsigned_id_token(nonce="n", issuer=_ISSUER, client_id=connection.client_id)
        return httpx.Response(
            200, json={"access_token": "a-real-access-token", "token_type": "Bearer", "id_token": id_token}
        )

    monkeypatch.setattr(
        "gateway.services.identity._connection_clients.CheckedFederationTransport",
        lambda *, allow_private_hosts: _StubTransport(handler),
    )
    metadata = _metadata()
    config = _provider_config_for(connection, metadata, redirect_uri=_REDIRECT_URI)
    client = OAuthClient(
        config, transport_factory=_connection_clients._transport_factory_for(allow_private_hosts=False)
    )
    pending = PendingState(
        state="s",
        redirect_uri=_REDIRECT_URI,
        nonce="n",
        openid_requested=True,
        created_at=datetime.now(UTC).timestamp(),
    )
    callback = AuthorizationCallback(code="a-code", state="s", iss=_ISSUER)

    tokens = await client.exchange_code(callback, pending_state=pending)

    assert tokens.access_token == "a-real-access-token"


@pytest.mark.asyncio
async def test_the_ssrf_check_still_refuses_a_private_token_url() -> None:
    """Asserted by the refusal's cause, not its type: an unreachable address fails on any transport,
    so a refusal identified by type alone would pass with the checks removed."""
    token_url = "https://127.0.0.1:9/token"
    factory = _connection_clients._transport_factory_for(allow_private_hosts=False)
    transport = factory(token_url)

    with pytest.raises(httpx.ConnectError) as refusal:
        await transport.handle_async_request(httpx.Request("POST", token_url))

    assert isinstance(refusal.value.__cause__, UnsafeURLError | RetrievalTargetError)
