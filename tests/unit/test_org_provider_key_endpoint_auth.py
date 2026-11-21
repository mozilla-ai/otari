"""An organization's endpoint credential is encrypted on the way in and survives a form that never saw it."""

import uuid
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from gateway.exceptions.providers_exceptions import OrgProviderKeyEndpointAuthRefusedError
from gateway.models.provider_keys import OrgProviderKey
from gateway.schemas.providers import EndpointAuthRequest
from gateway.services.catalog._served import _read_engine_settings
from gateway.services.model_discovery_service import _default_headers
from gateway.services.providers._org_provider_model_service import OrgProviderModelService
from gateway.services.secret_box import decrypt_secret, encrypt_secret, generate_secret_key
from gateway.services.tenancy import org_provider_key_service as store
from gateway.types.endpoint_auth import EndpointAuth, EndpointAuthMethod

LATER = datetime.now(UTC) + timedelta(days=90)


@pytest.fixture(autouse=True)
def _secret_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())


def _sent(**overrides: object) -> EndpointAuthRequest:
    fields: dict[str, object] = {"method": "static_header", "header_name": "Fastly-Key", "expires_at": LATER}
    return EndpointAuthRequest(**{**fields, **overrides})  # type: ignore[arg-type]


def test_a_sent_value_is_encrypted_and_only_its_last_four_are_kept_readable() -> None:
    auth = store._endpoint_auth_for_storage(_sent(secret="abcd-wxyz"), stored=EndpointAuth())

    assert auth.encrypted_secret is not None
    assert auth.encrypted_secret != "abcd-wxyz"
    assert decrypt_secret(auth.encrypted_secret) == "abcd-wxyz"
    assert auth.last4 == "wxyz"


def test_a_form_resubmitted_without_the_value_keeps_the_stored_one() -> None:
    stored = EndpointAuth(
        method=EndpointAuthMethod.STATIC_HEADER,
        header_name="Fastly-Key",
        encrypted_secret="gAAAA-stored",
        last4="9f2a",
        expires_at=LATER,
    )

    auth = store._endpoint_auth_for_storage(_sent(expires_at=LATER + timedelta(days=1)), stored=stored)

    assert auth.encrypted_secret == "gAAAA-stored"
    assert auth.last4 == "9f2a"
    assert auth.expires_at == LATER + timedelta(days=1)


def test_sending_nothing_clears_the_credential() -> None:
    stored = EndpointAuth(method=EndpointAuthMethod.STATIC_HEADER, header_name="Fastly-Key", encrypted_secret="x")

    assert store._endpoint_auth_for_storage(None, stored=stored) == EndpointAuth()


def _gate(**overrides: object) -> None:
    fields: dict[str, object] = {
        "method": EndpointAuthMethod.STATIC_HEADER,
        "header_name": "Fastly-Key",
        "encrypted_secret": "gAAAA-x",
        "expires_at": LATER,
    }
    store._gate_endpoint_auth(EndpointAuth(**{**fields, **overrides}), api_base="https://proxy.example/v1")  # type: ignore[arg-type]


def test_a_whole_credential_on_a_tenant_endpoint_is_accepted() -> None:
    _gate()


@pytest.mark.parametrize("header_name", ["authorization", "x-api-key", "Otari-Key", "transfer-encoding", "Host"])
def test_a_header_name_a_tenant_may_not_choose_is_refused_at_write(header_name: str) -> None:
    with pytest.raises(OrgProviderKeyEndpointAuthRefusedError):
        _gate(header_name=header_name)


def test_an_expiry_in_the_past_is_refused_at_write() -> None:
    with pytest.raises(OrgProviderKeyEndpointAuthRefusedError):
        _gate(expires_at=datetime.now(UTC) - timedelta(seconds=1))


def test_an_endpoint_credential_is_refused_on_a_key_with_no_base_url() -> None:
    auth = EndpointAuth(
        method=EndpointAuthMethod.STATIC_HEADER,
        header_name="Fastly-Key",
        encrypted_secret="gAAAA-x",
        expires_at=LATER,
    )

    with pytest.raises(OrgProviderKeyEndpointAuthRefusedError):
        store._gate_endpoint_auth(auth, api_base=None)


def test_a_stored_row_reports_the_credential_it_carries() -> None:
    row = OrgProviderKey(
        organization_id=uuid.uuid4(),
        provider="openai",
        name="k",
        created_at=datetime.now(UTC),
        endpoint_auth_method="static_header",
        endpoint_auth_header_name="Fastly-Key",
        encrypted_endpoint_auth_secret="gAAAA-x",
        endpoint_auth_last4="9f2a",
        endpoint_auth_expires_at=LATER,
    )

    assert store._endpoint_auth_of(row) == EndpointAuth(
        method="static_header",
        header_name="Fastly-Key",
        encrypted_secret="gAAAA-x",
        last4="9f2a",
        expires_at=LATER,
    )


def test_a_row_carrying_no_credential_reports_none() -> None:
    row = OrgProviderKey(organization_id=uuid.uuid4(), provider="openai", name="k", created_at=datetime.now(UTC))

    assert store._endpoint_auth_of(row).declared is False


@pytest.mark.asyncio
async def test_the_engine_probe_sends_the_endpoint_credential_beside_the_api_key() -> None:
    seen: list[httpx.Headers] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers)
        return httpx.Response(httpx.codes.OK, json={})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await _read_engine_settings(
            client,
            api_base="https://proxy.example/v1",
            api_key="sk-1",
            extra_headers={"Fastly-Key": "edge-s3cret"},
        )

    assert seen
    assert seen[0]["Fastly-Key"] == "edge-s3cret"
    assert seen[0]["Authorization"] == "Bearer sk-1"


def test_the_engine_probe_reads_its_headers_from_the_client_arguments() -> None:
    assert _default_headers({"default_headers": {"Fastly-Key": "edge-s3cret"}}) == {"Fastly-Key": "edge-s3cret"}
    assert _default_headers({"timeout": 5}) is None
    assert _default_headers(None) is None


def test_discovery_sends_the_endpoint_credential() -> None:
    row = OrgProviderKey(
        organization_id=uuid.uuid4(),
        provider="openai",
        name="k",
        created_at=datetime.now(UTC),
        api_base="https://proxy.example/v1",
        endpoint_auth_method="static_header",
        endpoint_auth_header_name="Fastly-Key",
        encrypted_endpoint_auth_secret=encrypt_secret("edge-s3cret"),
    )

    assert OrgProviderModelService._endpoint_headers(row) == {"Fastly-Key": "edge-s3cret"}


def test_discovery_sends_no_credential_it_cannot_decrypt(monkeypatch: pytest.MonkeyPatch) -> None:
    row = OrgProviderKey(
        organization_id=uuid.uuid4(),
        provider="openai",
        name="k",
        created_at=datetime.now(UTC),
        api_base="https://proxy.example/v1",
        endpoint_auth_method="static_header",
        endpoint_auth_header_name="Fastly-Key",
        encrypted_endpoint_auth_secret=encrypt_secret("edge-s3cret"),
    )
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())

    assert OrgProviderModelService._endpoint_headers(row) == {}
