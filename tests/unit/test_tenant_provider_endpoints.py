"""An organization's provider key reaches its ``api_base`` only through the pinned, public-only client."""

import ipaddress
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import httpx2
import pytest
from any_llm import AnyLLM, LLMProvider

from gateway.adapters.provider_file_adapter import AnyLlmProviderFiles, _AnyLlmFileSession
from gateway.core.config import GatewayConfig
from gateway.core.provider_params import TENANT_ENDPOINT_HTTPX2_PROVIDERS, TENANT_ENDPOINT_PROVIDERS
from gateway.exceptions.files_exceptions import ProviderFileUnavailableError
from gateway.models.provider_keys import OrgProviderKey
from gateway.services import provider_kwargs
from gateway.services.providers import owned_endpoint_sdk_client
from gateway.services.providers._owned_endpoint_network import OwnedEndpointTransport
from gateway.services.secret_box import encrypt_mapping, encrypt_secret, generate_secret_key
from gateway.services.tenancy import org_provider_key_service as store
from gateway.services.web_retrieval_network import PinnedTransportError, SystemAddressResolver
from gateway.types.endpoint_auth import ENDPOINT_AUTH_HEADERS_KEY
from gateway.types.provider_account import ResolvedCredential
from gateway.types.provider_file import ProviderFile

WORKSPACE_ID = uuid.uuid4()


def _sdk_clients(llm: AnyLLM) -> list[httpx.AsyncClient | httpx2.AsyncClient]:
    """The HTTP clients behind every SDK client a provider instance holds."""
    return [
        inner
        for value in vars(llm).values()
        if isinstance(inner := getattr(value, "_client", None), (httpx.AsyncClient, httpx2.AsyncClient))
    ]


def _sdk_default_headers(llm: AnyLLM) -> list[dict[str, str]]:
    """What every SDK client a provider instance holds adds to each request it makes."""
    return [
        dict(headers)
        for value in vars(llm).values()
        if (headers := getattr(value, "default_headers", None)) is not None
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", sorted(TENANT_ENDPOINT_PROVIDERS))
async def test_every_sdk_client_of_a_pinnable_provider_uses_the_pinned_client(provider: str) -> None:
    pinned = owned_endpoint_sdk_client(LLMProvider(provider))
    expected = httpx2.AsyncClient if provider in TENANT_ENDPOINT_HTTPX2_PROVIDERS else httpx.AsyncClient
    assert isinstance(pinned, expected)
    # Refusing the redirect is the point of pinning: a download that answers with
    # one would otherwise reach an address no check was run against.
    assert pinned.follow_redirects is False

    llm = AnyLLM.create(provider, api_key="k", api_base="https://203.0.114.1/v1", http_client=pinned)

    clients = _sdk_clients(llm)
    assert clients, f"{provider} holds no SDK client this test can see"
    assert all(client is pinned for client in clients)


@pytest.mark.asyncio
async def test_a_download_an_endpoint_redirects_is_refused_rather_than_followed() -> None:
    """The redirect is what reaches an address no check was run against, so the read fails instead."""
    followed: list[str] = []

    def answer(request: httpx.Request) -> httpx.Response:
        followed.append(str(request.url))
        return httpx.Response(302, headers={"location": "http://169.254.170.2/v2/metadata"})

    # follow_redirects=False is what the pinned client sets; the mock transport
    # stands in for the endpoint so no address is dialed.
    pinned = httpx.AsyncClient(transport=httpx.MockTransport(answer), follow_redirects=False)
    session = AnyLlmProviderFiles().open_session(
        provider=LLMProvider.OPENAI,
        instance="openai",
        credential=ResolvedCredential(
            api_key="sk-1", api_base="https://proxy.example/v1", client_args={"http_client": pinned}
        ),
    )
    file = ProviderFile(file_id="cfile_1", filename="out.txt", container_id="cntr_1")

    with pytest.raises(ProviderFileUnavailableError):
        async for _ in session.read(file, budget_bytes=1024):
            pass  # pragma: no cover - the first read raises
    await session.aclose()

    assert followed == ["https://proxy.example/v1/containers/cntr_1/files/cfile_1/content"]
    assert not pinned.is_closed


def _entry(**entry: Any) -> None:
    store._org_cache[(WORKSPACE_ID, "openai")] = entry


@pytest.fixture(autouse=True)
def _clean_overlay() -> Any:
    store.reset_org_provider_cache()
    yield
    store.reset_org_provider_cache()


@pytest.mark.asyncio
async def test_an_organization_key_with_an_api_base_dials_through_the_pinned_client() -> None:
    _entry(api_base="https://proxy.example/v1", api_key="sk-1", client_args={"timeout": 5})

    kwargs = provider_kwargs.get_provider_kwargs(GatewayConfig(), LLMProvider.OPENAI, workspace_id=WORKSPACE_ID)

    assert kwargs["client_args"]["http_client"] is owned_endpoint_sdk_client(LLMProvider.OPENAI)
    assert kwargs["client_args"]["timeout"] == 5


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", [LLMProvider.OPENAI, LLMProvider.ANTHROPIC])
async def test_a_file_session_on_an_organization_key_dials_through_the_pinned_client(provider: LLMProvider) -> None:
    store._org_cache[(WORKSPACE_ID, provider.value)] = {"api_base": "https://proxy.example/v1", "api_key": "sk-1"}
    kwargs = provider_kwargs.get_provider_kwargs(GatewayConfig(), provider, workspace_id=WORKSPACE_ID)
    pinned = owned_endpoint_sdk_client(provider)

    session = AnyLlmProviderFiles().open_session(
        provider=provider, instance=provider.value, credential=provider_kwargs.effective_credential(provider, kwargs)
    )
    assert isinstance(session, _AnyLlmFileSession)
    clients = _sdk_clients(session._llm)
    await session.aclose()

    assert clients
    assert all(client is pinned for client in clients)
    assert not pinned.is_closed


@pytest.mark.asyncio
async def test_an_organization_key_on_the_providers_own_endpoint_is_not_pinned() -> None:
    _entry(api_key="sk-1")

    kwargs = provider_kwargs.get_provider_kwargs(GatewayConfig(), LLMProvider.OPENAI, workspace_id=WORKSPACE_ID)

    assert "http_client" not in kwargs.get("client_args", {})


@pytest.mark.asyncio
async def test_an_operator_configured_api_base_is_not_pinned() -> None:
    config = GatewayConfig(providers={"openai": {"api_key": "sk-op", "api_base": "http://10.0.0.5/v1"}})

    kwargs = provider_kwargs.get_provider_kwargs(config, LLMProvider.OPENAI, workspace_id=WORKSPACE_ID)

    assert "http_client" not in kwargs.get("client_args", {})


@pytest.mark.asyncio
async def test_the_pinned_client_refuses_an_address_a_name_moved_to(monkeypatch: pytest.MonkeyPatch) -> None:
    async def resolve(self: SystemAddressResolver, host: str, port: int) -> tuple[ipaddress.IPv4Address, ...]:
        return (ipaddress.IPv4Address("169.254.169.254"),)

    monkeypatch.setattr(SystemAddressResolver, "resolve", resolve)

    with pytest.raises(PinnedTransportError):
        await OwnedEndpointTransport().handle_async_request(httpx.Request("POST", "https://proxy.example/v1/chat"))


def _row(provider: str, *, client_args: dict[str, Any] | None = None, **fields: Any) -> OrgProviderKey:
    """A stored key, with its client extras sealed as the database holds them."""
    return OrgProviderKey(
        organization_id=uuid.uuid4(),
        provider=provider,
        name="k",
        created_at=datetime.now(UTC),
        encrypted_client_args=encrypt_mapping(client_args),
        **fields,
    )


def test_a_stored_key_loses_the_client_args_its_provider_may_not_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    row = _row(
        "openai",
        client_args={"base_url": "http://10.0.0.1", "default_headers": {"X": "1"}, "organization": "org-1"},
    )

    assert store._row_to_entry(row) == {"client_args": {"organization": "org-1"}}


def test_a_stored_key_on_a_provider_that_cannot_be_pinned_is_not_served_with_its_api_base() -> None:
    row = _row("mistral", api_base="https://mistral.example/v1")

    with pytest.raises(store.UnservableEndpointError):
        store._row_to_entry(row)
    assert store.key_is_usable(row) is False


def test_a_stored_bedrock_key_keeps_its_credential_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    args = {"region_name": "us-east-1", "aws_access_key_id": "AKIA", "aws_secret_access_key": "s"}

    assert store._row_to_entry(_row("bedrock", client_args={**args, "endpoint_url": "http://10.0.0.1"})) == {
        "client_args": args
    }


def _key_with_endpoint_auth(**overrides: Any) -> OrgProviderKey:
    fields: dict[str, Any] = {
        "api_base": "https://proxy.example/v1",
        "endpoint_auth_method": "static_header",
        "endpoint_auth_header_name": "Fastly-Key",
        "encrypted_endpoint_auth_secret": encrypt_secret("edge-s3cret"),
        "endpoint_auth_last4": "cret",
        "endpoint_auth_expires_at": datetime.now(UTC) + timedelta(days=30),
    }
    return _row("openai", **{**fields, **overrides})


def test_a_stored_endpoint_credential_becomes_the_header_it_names(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())

    entry = store._row_to_entry(_key_with_endpoint_auth())

    assert entry[ENDPOINT_AUTH_HEADERS_KEY] == {"Fastly-Key": "edge-s3cret"}


# The proxy answers for the credential on every request, so a rotation date
# Otari cannot verify must never take a working key out of service.
def test_a_stored_endpoint_credential_past_its_expiry_is_still_served(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    key = _key_with_endpoint_auth(endpoint_auth_expires_at=datetime(2020, 1, 1, tzinfo=UTC))

    assert store._row_to_entry(key)[ENDPOINT_AUTH_HEADERS_KEY] == {"Fastly-Key": "edge-s3cret"}
    assert store.key_is_usable(key) is True


@pytest.mark.parametrize("header_name", ["authorization", "x-api-key", "Otari-Key", "transfer-encoding"])
def test_a_stored_credential_naming_a_header_a_tenant_may_not_set_refuses_the_key(
    header_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    key = _key_with_endpoint_auth(endpoint_auth_header_name=header_name)

    with pytest.raises(store.UnservableEndpointError):
        store._row_to_entry(key)
    assert store.key_is_usable(key) is False


@pytest.mark.asyncio
async def test_an_endpoint_credential_is_sent_as_a_default_header_and_never_to_the_provider() -> None:
    _entry(
        api_base="https://proxy.example/v1",
        api_key="sk-1",
        **{ENDPOINT_AUTH_HEADERS_KEY: {"Fastly-Key": "edge-s3cret"}},
    )

    kwargs = provider_kwargs.get_provider_kwargs(GatewayConfig(), LLMProvider.OPENAI, workspace_id=WORKSPACE_ID)

    assert kwargs["client_args"]["default_headers"] == {"Fastly-Key": "edge-s3cret"}
    assert kwargs["client_args"]["http_client"] is owned_endpoint_sdk_client(LLMProvider.OPENAI)
    # any-llm forwards every kwarg but client_args to the completion call, so
    # anything left at the top level reaches the model provider.
    assert ENDPOINT_AUTH_HEADERS_KEY not in kwargs
    assert "edge-s3cret" not in str({k: v for k, v in kwargs.items() if k != "client_args"})


# The gateway builds this header itself from a validated field; a provider whose
# SDK refuses the name would make the endpoint credential undeliverable there.
@pytest.mark.asyncio
@pytest.mark.parametrize("provider", sorted(TENANT_ENDPOINT_PROVIDERS))
async def test_every_pinnable_provider_sends_a_gateway_built_default_header(provider: str) -> None:
    """The header reaches the request, and never the client every organization on this provider shares.

    An SDK that dropped the keyword would send the request to the tenant's
    proxy with no credential, and one that set it on the injected client would
    put one organization's credential on a pool keyed by address alone.
    """
    llm = AnyLLM.create(
        provider,
        api_key="k",
        api_base="https://203.0.114.1/v1",
        http_client=owned_endpoint_sdk_client(LLMProvider(provider)),
        default_headers={"Fastly-Key": "edge-s3cret"},
    )

    clients = _sdk_clients(llm)
    assert clients, f"{provider} holds no SDK client this test can see"
    assert any("Fastly-Key" in headers for headers in _sdk_default_headers(llm)), (
        f"{provider} drops a gateway-built default header"
    )
    assert all("Fastly-Key" not in client.headers for client in clients), (
        f"{provider} puts a tenant's header on the client every organization shares"
    )
