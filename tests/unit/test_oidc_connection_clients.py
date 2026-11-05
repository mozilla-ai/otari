"""Build the apron-auth client an organization's own OIDC connection signs in through.

Discovery is never exercised here: ``metadata`` is handed in already resolved, the shape
``_connection_service.metadata_for_sign_in`` serves. Most of a callback exchange is exercised only
up to the check it is testing, by monkeypatching ``apron_auth.OAuthClient.exchange_code`` the way
``tests/unit/test_oauth_service.py`` and ``tests/integration/test_oauth_api.py`` already patch
apron-auth's client to test a code exchange's outcome without one. A few tests go one layer
deeper, through a stubbed classic transport the way ``tests/unit/test_oidc_discovery.py`` stubs
one, to prove the token-endpoint request itself, and the SSRF check it carries, still work.
"""

import json
import uuid
from base64 import urlsafe_b64encode
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from typing import Any, cast

import httpx
import httpx2
import pytest
from apron_auth import INVALID_CLIENT, INVALID_GRANT, OAuthClient, ServerMetadata, StoredStateClient, TokenExchangeError
from apron_auth.errors import AuthorizationResponseError, IssuerValidationError, StateError
from apron_auth.models import AuthorizationCallback
from apron_auth.models import OAuthPendingState as PendingState
from sqlalchemy import Select
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.models.identity import OIDCConnection
from gateway.services.identity import _connection_clients
from gateway.services.identity._connection_clients import (
    ConnectionCredentialCheck,
    _provider_config_for,
    check_connection_credentials,
    client_for,
    identity_handler_for,
    pending_flow_provider,
)
from gateway.services.identity._discovery import DiscoveredProvider
from gateway.services.secret_box import encrypt_secret, generate_secret_key
from gateway.services.url_safety import UnsafeURLError
from gateway.services.web_retrieval_network import (
    CheckedFederationTransport,
    Httpx2CompatTransport,
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


def _server_metadata(
    *, issuer: str = _ISSUER, iss_parameter_supported: bool = True, token_url: str | None = None
) -> ServerMetadata:
    return ServerMetadata(
        authorize_url=f"{issuer}/authorize",
        token_url=token_url or f"{issuer}/token",
        jwks_url=f"{issuer}/jwks",
        userinfo_url=f"{issuer}/userinfo",
        issuer=issuer,
        iss_parameter_supported=iss_parameter_supported,
        code_challenge_methods=["S256"],
        token_endpoint_auth_methods=["client_secret_basic"],
    )


def _metadata(
    *,
    issuer: str = _ISSUER,
    discovered_at: datetime | None = None,
    iss_parameter_supported: bool = True,
    token_url: str | None = None,
) -> DiscoveredProvider:
    return DiscoveredProvider(
        resolved_issuer=issuer,
        document=_server_metadata(
            issuer=issuer, iss_parameter_supported=iss_parameter_supported, token_url=token_url
        ).model_dump(mode="json"),
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


def _untouched_db() -> AsyncSession:
    return cast("AsyncSession", _UntouchedSession())


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

    result = await pending_flow_provider(cast("AsyncSession", session), "a-state")

    assert result == "oidc:some-connection-id"
    assert len(session.statements) == 1
    assert isinstance(session.statements[0], Select)


@pytest.mark.asyncio
async def test_pending_flow_provider_answers_none_for_an_unknown_state() -> None:
    session = _SelectOnlySession(None)

    result = await pending_flow_provider(cast("AsyncSession", session), "a-state")

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
        _connection(), _metadata(), db=_untouched_db(), redirect_uri=_REDIRECT_URI, allow_private_hosts=False
    )
    callback = AuthorizationCallback(code="a-code", state="a-state", iss="https://attacker.example")

    with pytest.raises(IssuerValidationError):
        await client.exchange_code(callback, browser_token="a-browser-token")


@pytest.mark.asyncio
async def test_a_callback_naming_the_right_issuer_passes_the_issuer_check() -> None:
    client = client_for(
        _connection(), _metadata(), db=_untouched_db(), redirect_uri=_REDIRECT_URI, allow_private_hosts=False
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
        db=_untouched_db(),
        redirect_uri=_REDIRECT_URI,
        allow_private_hosts=False,
    )
    callback = AuthorizationCallback(code="a-code", state="a-state")

    with pytest.raises(IssuerValidationError):
        await client.exchange_code(callback, browser_token="a-browser-token")


def test_client_for_returns_a_stored_state_client() -> None:
    client = client_for(
        _connection(), _metadata(), db=_untouched_db(), redirect_uri=_REDIRECT_URI, allow_private_hosts=False
    )

    assert isinstance(client, StoredStateClient)


def test_client_for_uses_the_memoized_handler() -> None:
    metadata = _metadata()
    handler = identity_handler_for(metadata, allow_private_hosts=False)

    client = client_for(
        _connection(), metadata, db=_untouched_db(), redirect_uri=_REDIRECT_URI, allow_private_hosts=False
    )

    assert client._identity_handler is handler


def test_the_transport_factory_wraps_only_the_token_endpoint() -> None:
    metadata = _metadata()
    factory = _connection_clients._transport_factory_for(metadata, allow_private_hosts=False)

    token_transport = factory(f"{_ISSUER}/token")
    userinfo_transport = factory(f"{_ISSUER}/userinfo")

    assert isinstance(token_transport, Httpx2CompatTransport)
    assert isinstance(userinfo_transport, CheckedFederationTransport)
    assert not isinstance(userinfo_transport, Httpx2CompatTransport)


@pytest.mark.asyncio
async def test_a_real_exchange_through_the_token_url_succeeds_with_the_adapter(
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
        config, transport_factory=_connection_clients._transport_factory_for(metadata, allow_private_hosts=False)
    )
    pending = PendingState(state="s", redirect_uri=_REDIRECT_URI, nonce="n", created_at=datetime.now(UTC).timestamp())
    callback = AuthorizationCallback(code="a-code", state="s", iss=_ISSUER)

    tokens = await client.exchange_code(callback, pending_state=pending)

    assert tokens.access_token == "a-real-access-token"


@pytest.mark.asyncio
async def test_the_ssrf_check_still_refuses_a_private_token_url() -> None:
    """The adapter wraps the checked transport; it must not quietly become a plain pass-through.

    Asserts the refusal came from the outbound URL checks, by its cause, rather than that the
    request merely failed: an unreachable address fails on any transport, so a refusal identified
    by type alone would pass with the checks removed.
    """
    token_url = "https://127.0.0.1:9/token"
    metadata = _metadata(token_url=token_url)
    factory = _connection_clients._transport_factory_for(metadata, allow_private_hosts=False)
    transport = factory(token_url)
    assert isinstance(transport, Httpx2CompatTransport)

    with pytest.raises(httpx.ConnectError) as refusal:
        await transport.handle_async_request(httpx2.Request("POST", token_url))

    assert isinstance(refusal.value.__cause__, UnsafeURLError | RetrievalTargetError)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error_code", "expected"),
    [
        ("invalid_grant", ConnectionCredentialCheck.ACCEPTED),
        ("invalid_client", ConnectionCredentialCheck.REFUSED),
        ("unauthorized_client", ConnectionCredentialCheck.REFUSED),
    ],
)
async def test_the_probe_reads_a_providers_own_refusal_through_a_real_exchange(
    monkeypatch: pytest.MonkeyPatch, error_code: str, expected: ConnectionCredentialCheck
) -> None:
    """Drive the probe through the whole chain, stubbing only the socket.

    The tests below patch ``exchange_code`` and so assert this module's mapping alone. This one
    asserts the part the mapping rests on: that an error body on the wire reaches it as the
    matching ``error_code``.
    """

    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": error_code})

    monkeypatch.setattr(
        "gateway.services.identity._connection_clients.CheckedFederationTransport",
        lambda *, allow_private_hosts: _StubTransport(handler),
    )

    result = await check_connection_credentials(
        _connection(), _metadata(), redirect_uri=_REDIRECT_URI, allow_private_hosts=False
    )

    assert result is expected


@pytest.mark.asyncio
async def test_check_connection_credentials_reports_accepted_for_invalid_grant(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _raise(_self: OAuthClient, _callback: AuthorizationCallback, *, pending_state: Any) -> Any:
        raise TokenExchangeError("bad code", error_code=INVALID_GRANT)

    monkeypatch.setattr(OAuthClient, "exchange_code", _raise)

    result = await check_connection_credentials(
        _connection(), _metadata(), redirect_uri=_REDIRECT_URI, allow_private_hosts=False
    )

    assert result is ConnectionCredentialCheck.ACCEPTED


@pytest.mark.asyncio
@pytest.mark.parametrize("error_code", [INVALID_CLIENT, "unauthorized_client"])
async def test_check_connection_credentials_reports_refused_for_a_client_credential_error(
    monkeypatch: pytest.MonkeyPatch, error_code: str
) -> None:
    async def _raise(_self: OAuthClient, _callback: AuthorizationCallback, *, pending_state: Any) -> Any:
        raise TokenExchangeError("bad client", error_code=error_code)

    monkeypatch.setattr(OAuthClient, "exchange_code", _raise)

    result = await check_connection_credentials(
        _connection(), _metadata(), redirect_uri=_REDIRECT_URI, allow_private_hosts=False
    )

    assert result is ConnectionCredentialCheck.REFUSED


@pytest.mark.asyncio
async def test_check_connection_credentials_is_inconclusive_for_an_unrecognized_error_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _raise(_self: OAuthClient, _callback: AuthorizationCallback, *, pending_state: Any) -> Any:
        raise TokenExchangeError("server error", error_code="server_error")

    monkeypatch.setattr(OAuthClient, "exchange_code", _raise)

    result = await check_connection_credentials(
        _connection(), _metadata(), redirect_uri=_REDIRECT_URI, allow_private_hosts=False
    )

    assert result is ConnectionCredentialCheck.INCONCLUSIVE


@pytest.mark.asyncio
async def test_check_connection_credentials_is_inconclusive_on_an_unrelated_oauth_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _raise(_self: OAuthClient, _callback: AuthorizationCallback, *, pending_state: Any) -> Any:
        raise AuthorizationResponseError("no code to redeem")

    monkeypatch.setattr(OAuthClient, "exchange_code", _raise)

    result = await check_connection_credentials(
        _connection(), _metadata(), redirect_uri=_REDIRECT_URI, allow_private_hosts=False
    )

    assert result is ConnectionCredentialCheck.INCONCLUSIVE


@pytest.mark.asyncio
async def test_check_connection_credentials_is_inconclusive_when_nothing_was_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A provider that redeems the deliberately invalid code proves nothing either way."""

    async def _succeed(_self: OAuthClient, _callback: AuthorizationCallback, *, pending_state: Any) -> Any:
        return object()

    monkeypatch.setattr(OAuthClient, "exchange_code", _succeed)

    result = await check_connection_credentials(
        _connection(), _metadata(), redirect_uri=_REDIRECT_URI, allow_private_hosts=False
    )

    assert result is ConnectionCredentialCheck.INCONCLUSIVE
