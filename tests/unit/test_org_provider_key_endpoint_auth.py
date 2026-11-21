"""An organization's endpoint credential is encrypted on the way in and survives a form that never saw it."""

import uuid
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from pydantic import ValidationError

from gateway.exceptions.providers_exceptions import OrgProviderKeyEndpointAuthRefusedError
from gateway.models.provider_keys import OrgProviderKey
from gateway.repositories.tenancy.org_provider_key_repository import OrgProviderKeyRepository
from gateway.schemas.providers import EndpointAuthRequest
from gateway.services import model_discovery_service
from gateway.services.catalog._served import _read_engine_settings
from gateway.services.model_discovery_service import (
    ProviderDiscovery,
    _copy_discovery,
    _default_headers,
    _is_credential_rejection,
    _short_error,
)
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
    store._gate_endpoint_auth(
        EndpointAuth(**{**fields, **overrides}),  # type: ignore[arg-type]
        api_base="https://proxy.example/v1",
        stored=EndpointAuth(),
    )


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
        store._gate_endpoint_auth(auth, api_base=None, stored=EndpointAuth())


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


def _rejected(status_code: int) -> ProviderDiscovery:
    """A discovery as ``test_provider_credentials`` builds it from an upstream status."""

    class _Refused(Exception):
        def __init__(self) -> None:
            super().__init__("Unauthorized")
            self.status_code = status_code

    exc = _Refused()
    return ProviderDiscovery(
        provider="openai",
        models=[],
        error=_short_error(exc, provider="openai"),
        credentials_rejected=_is_credential_rejection(exc),
    )


def _key(*, endpoint_auth: bool) -> OrgProviderKey:
    return OrgProviderKey(
        organization_id=uuid.uuid4(),
        provider="openai",
        name="k",
        created_at=datetime.now(UTC),
        api_base="https://proxy.example/v1",
        endpoint_auth_method="static_header" if endpoint_auth else None,
        endpoint_auth_header_name="Fastly-Key" if endpoint_auth else None,
        encrypted_endpoint_auth_secret=encrypt_secret("edge-s3cret") if endpoint_auth else None,
    )


@pytest.mark.parametrize("status_code", [401, 403])
def test_a_refusal_is_read_off_the_status_the_address_returned(status_code: int) -> None:
    assert _rejected(status_code).credentials_rejected is True


@pytest.mark.parametrize("status_code", [404, 429, 500])
def test_a_failure_that_is_not_a_refusal_is_not_read_as_one(status_code: int) -> None:
    assert _rejected(status_code).credentials_rejected is False


def test_a_refusal_of_a_key_with_an_endpoint_credential_names_both() -> None:
    named = OrgProviderModelService._named_refusal(_rejected(401), _key(endpoint_auth=True))

    assert named.error is not None
    assert "endpoint credential" in named.error
    assert "API key" in named.error


def test_a_refusal_of_a_key_with_one_credential_keeps_the_address_s_own_wording() -> None:
    discovery = _rejected(403)
    named = OrgProviderModelService._named_refusal(discovery, _key(endpoint_auth=False))

    assert named.error == discovery.error


# A key whose stored endpoint secret will not decrypt still declares one, so the
# admin is told to check it rather than sent to rotate the API key.
def test_a_refusal_names_an_endpoint_credential_that_will_not_decrypt(monkeypatch: pytest.MonkeyPatch) -> None:
    key = _key(endpoint_auth=True)
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())

    named = OrgProviderModelService._named_refusal(_rejected(401), key)

    assert named.error is not None
    assert "endpoint credential" in named.error


def test_a_failure_that_is_not_a_refusal_keeps_its_wording_on_a_key_with_two_credentials() -> None:
    discovery = _rejected(500)
    named = OrgProviderModelService._named_refusal(discovery, _key(endpoint_auth=True))

    assert named.error == discovery.error


def test_a_named_refusal_echoes_no_part_of_either_credential() -> None:
    named = OrgProviderModelService._named_refusal(_rejected(401), _key(endpoint_auth=True))

    assert named.error is not None
    assert "edge-s3cret" not in named.error
    assert "Fastly-Key" not in named.error


@pytest.mark.asyncio
async def test_a_dial_that_is_refused_reports_it_as_a_refusal(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Refused(Exception):
        status_code = 401

    async def _refuse(**_: object) -> list[object]:
        raise _Refused

    monkeypatch.setattr("gateway.services.model_discovery_service.alist_models", _refuse)

    discovery = await model_discovery_service.test_provider_credentials("openai", api_key="sk-1", declared=False)

    assert discovery.credentials_rejected is True


def test_a_copied_discovery_keeps_the_refusal() -> None:
    assert _copy_discovery(_rejected(401)).credentials_rejected is True


def test_an_expiry_with_no_offset_is_refused_at_the_boundary() -> None:
    """A naive value cannot be compared with the clock, so it never reaches one."""
    with pytest.raises(ValidationError):
        _sent(expires_at="2027-03-01T00:00:00")


def test_a_stored_endpoint_secret_follows_a_key_the_request_leaves_alone() -> None:
    assert store._endpoint_secret_survives("gAAAA-x", None, sent=False) is True


def test_a_stored_endpoint_secret_does_not_follow_a_key_whose_credential_is_removed() -> None:
    assert store._endpoint_secret_survives("gAAAA-x", None, sent=True) is False


def test_a_stored_endpoint_secret_does_not_follow_a_key_given_a_new_value() -> None:
    assert store._endpoint_secret_survives("gAAAA-x", _sent(secret="edge-s3cret"), sent=True) is False


def test_a_stored_endpoint_secret_follows_a_form_resubmitted_without_the_value() -> None:
    assert store._endpoint_secret_survives("gAAAA-x", _sent(), sent=True) is True


def test_a_key_holding_no_endpoint_secret_carries_nothing_to_another_address() -> None:
    assert store._endpoint_secret_survives(None, None, sent=False) is False


def _expired() -> EndpointAuth:
    return EndpointAuth(
        method=EndpointAuthMethod.STATIC_HEADER,
        header_name="Fastly-Key",
        encrypted_secret="gAAAA-x",
        expires_at=datetime.now(UTC) - timedelta(days=1),
    )


def test_a_rotation_date_that_passed_does_not_refuse_an_edit_that_leaves_it_alone() -> None:
    stored = _expired()

    store._gate_endpoint_auth(stored, api_base="https://proxy.example/v1", stored=stored)


# The dashboard resends the whole credential on every save, so a date that has
# passed would otherwise make the key impossible to rename.
def test_a_rotation_date_that_passed_does_not_refuse_being_sent_back_unchanged() -> None:
    stored = _expired()
    resent = EndpointAuth(
        method=stored.method,
        header_name=stored.header_name,
        encrypted_secret=stored.encrypted_secret,
        expires_at=stored.expires_at,
    )

    store._gate_endpoint_auth(resent, api_base="https://proxy.example/v1", stored=stored)


def test_a_rotation_date_moved_further_into_the_past_is_still_refused() -> None:
    stored = _expired()
    moved = EndpointAuth(
        method=stored.method,
        header_name=stored.header_name,
        encrypted_secret=stored.encrypted_secret,
        expires_at=datetime.now(UTC) - timedelta(days=30),
    )

    with pytest.raises(OrgProviderKeyEndpointAuthRefusedError):
        store._gate_endpoint_auth(moved, api_base="https://proxy.example/v1", stored=stored)


def test_removing_a_credential_whose_rotation_date_passed_is_allowed() -> None:
    store._gate_endpoint_auth(EndpointAuth(), api_base="https://proxy.example/v1", stored=_expired())


def test_a_credential_a_key_may_not_carry_is_refused_even_where_the_request_sent_none() -> None:
    stored = EndpointAuth(
        method=EndpointAuthMethod.STATIC_HEADER,
        header_name="Fastly-Key",
        encrypted_secret="gAAAA-x",
        expires_at=LATER,
    )

    with pytest.raises(OrgProviderKeyEndpointAuthRefusedError):
        store._gate_endpoint_auth(stored, api_base=None, stored=stored)


def test_a_key_skipped_for_its_provider_says_that_is_why() -> None:
    row = _key(endpoint_auth=False)
    row.provider = "mistral"

    with pytest.raises(store.UnservableEndpointError) as refusal:
        store._row_to_entry(row)

    assert "api_base" in str(refusal.value)


@pytest.mark.asyncio
async def test_forgetting_an_organization_s_secrets_clears_the_endpoint_credential_whole() -> None:
    """Every column of the credential goes, so no declaration is left with no value behind it."""
    statements: list[object] = []

    class _Session:
        async def execute(self, statement: object) -> None:
            statements.append(statement)

    repository = OrgProviderKeyRepository(_Session())  # type: ignore[arg-type]

    await repository.forget_secrets([uuid.uuid4()], [])

    assignments = str(statements[0])
    cleared = [name for name in OrgProviderKey.model_fields if "endpoint_auth" in name]
    assert cleared
    assert [name for name in cleared if name not in assignments] == []
