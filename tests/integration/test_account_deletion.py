"""Deleting one's own account (`DELETE /api/v1/auth/account`), end to end against PostgreSQL."""

import asyncio
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from sqlmodel import col

from gateway.adapters.file_storage_adapter import LocalDirFileStore
from gateway.api.deps import get_telemetry_storage_port
from gateway.core.config import API_ROOT
from gateway.models.api_keys import APIKey, DisabledReason
from gateway.models.files import FileObject
from gateway.models.guardrails import OrganizationGuardrail, OrganizationGuardrailDefinition
from gateway.models.identity import ErasedAccount, OIDCConnection
from gateway.models.organizations import OrganizationInvitation
from gateway.models.platform import RuntimeSetting
from gateway.models.provider_keys import OrgProviderKey
from gateway.models.providers import ProviderEndpoint
from gateway.models.tenancy import DashboardSession, Organization, OrganizationMember, User, Workspace
from gateway.models.tools import OrgWebSearchKey, WorkspaceMcpServer
from gateway.models.usage import UsageLog
from gateway.models.users import User as BillingUser
from gateway.services.dashboard_session_service import SESSION_COOKIE_NAME, hash_session_token
from gateway.services.password_service import hash_password
from gateway.services.tenancy.provisioning_service import BOOTSTRAP_IDENTITY_KEY

PASSWORD = "a-real-password"  # pragma: allowlist secret
EMAIL = "ada@example.com"


@dataclass(frozen=True)
class Account:
    identity_id: uuid.UUID
    organization_id: uuid.UUID
    workspace_id: uuid.UUID
    service_key_id: str
    usage_id: str
    file_id: str
    session_token: str


@pytest.fixture
def tmp_file_store(client: TestClient, tmp_path: Path) -> None:
    """Bind a local blob store, so the deletion is built with the files service."""
    cast(Any, client.app).state.file_store = LocalDirFileStore(str(tmp_path))


def _add_account(
    session_factory: Callable[[], Session],
    *,
    email: str = EMAIL,
    password: str | None = PASSWORD,
    organization_id: uuid.UUID | None = None,
    role: str = "owner",
) -> Account:
    """An account with an organization of its own (or a membership in ``organization_id``), a key, usage and a file."""
    session = session_factory()
    try:
        if organization_id is None:
            organization = Organization(name="Ada Lovelace's organization", slug=f"ada-{uuid.uuid4().hex[:8]}")
            session.add(organization)
            session.flush()
            organization_id = organization.id
        workspace = Workspace(organization_id=organization_id, name=f"Ada's workspace {uuid.uuid4().hex[:6]}")
        identity = User(
            email=email,
            full_name="Ada Lovelace",
            hashed_password=hash_password(password) if password is not None else None,
            active_organization_id=organization_id,
        )
        session.add_all([workspace, identity])
        session.flush()
        session.add(OrganizationMember(organization_id=organization_id, user_id=identity.id, role=role))
        billing_id = str(identity.id)
        session.add(BillingUser(user_id=billing_id, alias=email))
        session.flush()
        session.add(BillingUser(user_id=str(uuid.uuid4()), parent_user_id=billing_id, external_id="customer-7"))
        # A key another owner holds in the same workspace, as an organization's service key does.
        service_owner = BillingUser(user_id=f"svc_{uuid.uuid4().hex}", alias="Service")
        session.add(service_owner)
        session.flush()
        own_key = APIKey(id=str(uuid.uuid4()), key_hash=uuid.uuid4().hex, workspace_id=workspace.id, user_id=billing_id)
        service_key = APIKey(
            id=str(uuid.uuid4()), key_hash=uuid.uuid4().hex, workspace_id=workspace.id, user_id=service_owner.user_id
        )
        session.add_all([own_key, service_key])
        session.flush()
        usage = UsageLog(
            workspace_id=workspace.id,
            api_key_id=own_key.id,
            user_id=billing_id,
            model="gpt-4o-mini",
            endpoint="/v1/chat/completions",
            total_tokens=42,
            status="success",
            tags={"customer": "Ada"},
            source_label="ada-laptop",
            safety_identifier="ada-at-example",
            error_message="Unknown user ada@example.com",
        )
        file = FileObject(
            user_id=billing_id,
            workspace_id=workspace.id,
            filename="ada-medical-notes.pdf",
            mime_type="application/pdf",
            bytes=3,
            storage_ref="ref-ada",
        )
        token = f"otari-sess-{uuid.uuid4().hex}"
        session.add_all(
            [
                usage,
                file,
                DashboardSession(
                    token_hash=hash_session_token(token),
                    user_id=identity.id,
                    created_at=datetime.now(UTC),
                    expires_at=datetime.now(UTC) + timedelta(hours=12),
                ),
            ]
        )
        session.commit()
        return Account(
            identity_id=identity.id,
            organization_id=organization_id,
            workspace_id=workspace.id,
            service_key_id=service_key.id,
            usage_id=usage.id,
            file_id=file.id,
            session_token=token,
        )
    finally:
        session.close()


def _delete(client: TestClient, account: Account, **body: Any) -> Any:
    client.cookies.set(SESSION_COOKIE_NAME, account.session_token)
    return client.request("DELETE", f"{API_ROOT}/auth/account", json={"confirm_email": EMAIL, **body})


def _identity_exists(session_factory: Callable[[], Session], identity_id: uuid.UUID) -> bool:
    session = session_factory()
    try:
        return session.get(User, identity_id) is not None
    finally:
        session.close()


def test_deleting_an_account_erases_the_person_and_keeps_anonymous_usage(
    client: TestClient,
    db_session_factory: Callable[[], Session],
    tmp_file_store: None,
) -> None:
    account = _add_account(db_session_factory)

    response = _delete(client, account, current_password=PASSWORD)

    assert response.status_code == 204, response.text
    cleared = response.headers["set-cookie"]
    assert cleared.startswith(f"{SESSION_COOKIE_NAME}=")
    assert "Max-Age=0" in cleared
    session = db_session_factory()
    try:
        assert session.get(User, account.identity_id) is None
        assert session.query(DashboardSession).filter(col(DashboardSession.user_id) == account.identity_id).count() == 0
        assert session.get(BillingUser, str(account.identity_id)) is None
        assert session.query(BillingUser).filter(BillingUser.parent_user_id == str(account.identity_id)).count() == 0

        usage = session.get(UsageLog, account.usage_id)
        assert usage is not None
        assert usage.total_tokens == 42
        assert usage.user_id is not None and usage.user_id != str(account.identity_id)
        assert usage.tags is None
        assert usage.source_label is None
        assert usage.safety_identifier is None
        assert usage.error_message is None
        tombstone = session.get(BillingUser, usage.user_id)
        assert tombstone is not None
        assert tombstone.alias is None
        assert tombstone.deleted_at is not None

        file = session.get(FileObject, account.file_id)
        assert file is not None
        assert file.user_id == usage.user_id
        assert file.filename == "deleted"
        assert file.deleted_at is not None

        organization = session.get(Organization, account.organization_id)
        assert organization is not None
        assert organization.name == "Deleted organization"
        assert organization.slug == f"deleted-{account.organization_id.hex}"
        service_key = session.get(APIKey, account.service_key_id)
        assert service_key is not None
        assert service_key.is_active is False
        assert service_key.disabled_reason == DisabledReason.REVOKED

        assert session.get(ErasedAccount, account.identity_id) is not None
    finally:
        session.close()


def test_the_deleted_session_no_longer_authenticates(
    client: TestClient, db_session_factory: Callable[[], Session]
) -> None:
    account = _add_account(db_session_factory)
    assert _delete(client, account, current_password=PASSWORD).status_code == 204

    client.cookies.set(SESSION_COOKIE_NAME, account.session_token)
    assert client.get(f"{API_ROOT}/auth/profile").status_code == 401


@pytest.mark.parametrize(
    ("body", "detail"),
    [
        ({"confirm_email": "someone@example.com", "current_password": PASSWORD}, "Type the account's email"),
        ({}, "current password is required"),
        ({"current_password": "not-the-password"}, "Current password is incorrect"),
    ],
)
def test_an_unconfirmed_deletion_deletes_nothing(
    client: TestClient, db_session_factory: Callable[[], Session], body: dict[str, str], detail: str
) -> None:
    account = _add_account(db_session_factory)

    response = _delete(client, account, **body)

    assert response.status_code == 400, response.text
    assert detail in response.json()["detail"]
    assert _identity_exists(db_session_factory, account.identity_id)


def test_an_account_without_a_password_confirms_with_its_address_alone(
    client: TestClient, db_session_factory: Callable[[], Session]
) -> None:
    account = _add_account(db_session_factory, password=None)

    assert _delete(client, account).status_code == 204
    assert not _identity_exists(db_session_factory, account.identity_id)


def test_the_last_owner_of_an_organization_others_use_is_refused(
    client: TestClient, db_session_factory: Callable[[], Session]
) -> None:
    account = _add_account(db_session_factory)
    _add_account(db_session_factory, email="grace@example.com", organization_id=account.organization_id, role="member")

    response = _delete(client, account, current_password=PASSWORD)

    assert response.status_code == 409, response.text
    assert "Ada Lovelace's organization" in response.json()["detail"]
    assert _identity_exists(db_session_factory, account.identity_id)


def test_a_member_leaves_an_organization_others_use_as_it_was(
    client: TestClient, db_session_factory: Callable[[], Session]
) -> None:
    owner = _add_account(db_session_factory, email="grace@example.com")
    account = _add_account(db_session_factory, organization_id=owner.organization_id, role="member")

    assert _delete(client, account, current_password=PASSWORD).status_code == 204

    session = db_session_factory()
    try:
        organization = session.get(Organization, owner.organization_id)
        assert organization is not None
        assert organization.name == "Ada Lovelace's organization"
        assert (
            session.query(OrganizationMember)
            .filter(col(OrganizationMember.organization_id) == owner.organization_id)
            .count()
            == 1
        )
        service_key = session.get(APIKey, account.service_key_id)
        assert service_key is not None
        assert service_key.is_active is True
    finally:
        session.close()


def _add_secrets(session_factory: Callable[[], Session], account: Account) -> None:
    """A credential of every kind an organization or its workspaces store, in ``account``'s organization."""
    session = session_factory()
    try:
        session.add_all(
            [
                OrgProviderKey(
                    organization_id=account.organization_id,
                    provider="openai",
                    name="Ada's OpenAI",
                    encrypted_api_key="encrypted",
                    last4="abcd",
                    client_args={"organization": "org-ada"},
                ),
                ProviderEndpoint(
                    workspace_id=account.workspace_id,
                    name="ada-vllm",
                    provider="openai",
                    api_base="https://vllm.example.com/v1",
                    encrypted_api_key="encrypted",
                    last4="abcd",
                ),
                OrgWebSearchKey(
                    organization_id=account.organization_id,
                    provider="tavily",
                    name="Ada's Tavily",
                    encrypted_api_key="encrypted",
                    last4="abcd",
                ),
                WorkspaceMcpServer(
                    workspace_id=account.workspace_id,
                    name="ada-mcp",
                    url="https://mcp.example.com",
                    encrypted_token="encrypted",
                ),
                OrganizationGuardrailDefinition(
                    organization_id=account.organization_id,
                    name="ada-check",
                    guardrail_name="pydantic_ai",
                    encrypted_create_secrets="encrypted",
                ),
                OrganizationGuardrail(
                    organization_id=account.organization_id,
                    profile="ada-profile",
                    url="https://guardrails.example.com",
                    encrypted_credential="encrypted",
                ),
                OIDCConnection(
                    organization_id=account.organization_id,
                    issuer="https://idp.example.com",
                    resolved_issuer="https://idp.example.com",
                    client_id="ada-client",
                    encrypted_client_secret="encrypted",
                    client_secret_last4="cret",
                    display_name="Ada SSO",
                    discovery_document={},
                    discovered_at=datetime.now(UTC),
                    enabled=True,
                ),
            ]
        )
        session.commit()
    finally:
        session.close()


def _stored_secrets(session_factory: Callable[[], Session], account: Account) -> list[Any]:
    """Every credential still stored in ``account``'s organization and workspace."""
    session = session_factory()
    try:
        return [
            *session.query(col(OrgProviderKey.encrypted_api_key), col(OrgProviderKey.client_args))
            .filter(col(OrgProviderKey.organization_id) == account.organization_id)
            .all(),
            *session.query(ProviderEndpoint.encrypted_api_key)
            .filter(ProviderEndpoint.workspace_id == account.workspace_id)
            .all(),
            *session.query(OrgWebSearchKey.encrypted_api_key)
            .filter(OrgWebSearchKey.organization_id == account.organization_id)
            .all(),
            *session.query(WorkspaceMcpServer.encrypted_token)
            .filter(WorkspaceMcpServer.workspace_id == account.workspace_id)
            .all(),
            *session.query(OrganizationGuardrailDefinition.encrypted_create_secrets)
            .filter(OrganizationGuardrailDefinition.organization_id == account.organization_id)
            .all(),
            *session.query(OrganizationGuardrail.encrypted_credential)
            .filter(OrganizationGuardrail.organization_id == account.organization_id)
            .all(),
            *session.query(OIDCConnection.encrypted_client_secret)
            .filter(OIDCConnection.organization_id == account.organization_id)
            .all(),
        ]
    finally:
        session.close()


def test_an_organization_left_to_nobody_loses_its_stored_credentials(
    client: TestClient, db_session_factory: Callable[[], Session]
) -> None:
    account = _add_account(db_session_factory)
    _add_secrets(db_session_factory, account)
    unrelated = _add_account(db_session_factory, email="grace@example.com")
    _add_secrets(db_session_factory, unrelated)

    assert _delete(client, account, current_password=PASSWORD).status_code == 204

    assert all(value is None for row in _stored_secrets(db_session_factory, account) for value in row)
    unrelated_secrets = _stored_secrets(db_session_factory, unrelated)
    assert len(unrelated_secrets) == 7
    assert all(row[0] is not None for row in unrelated_secrets)
    session = db_session_factory()
    try:
        assert (
            session.query(OIDCConnection).filter(OIDCConnection.organization_id == account.organization_id).count() == 0
        )
    finally:
        session.close()


def test_an_organization_others_use_keeps_its_stored_credentials(
    client: TestClient, db_session_factory: Callable[[], Session]
) -> None:
    owner = _add_account(db_session_factory, email="grace@example.com")
    account = _add_account(db_session_factory, organization_id=owner.organization_id, role="member")
    _add_secrets(db_session_factory, owner)

    assert _delete(client, account, current_password=PASSWORD).status_code == 204

    secrets = _stored_secrets(db_session_factory, owner)
    assert len(secrets) == 7
    assert all(row[0] is not None for row in secrets)


def _invite(session_factory: Callable[[], Session], organization_id: uuid.UUID, email: str) -> uuid.UUID:
    session = session_factory()
    try:
        now = datetime.now(UTC)
        invitation = OrganizationInvitation(
            organization_id=organization_id,
            email=email,
            role="member",
            token_hash=uuid.uuid4().hex,
            expires_at=now + timedelta(days=7),
            send_count=1,
            last_send_requested_at=now,
        )
        session.add(invitation)
        session.commit()
        return invitation.id
    finally:
        session.close()


def test_invitations_to_the_deleted_address_are_withdrawn_everywhere(
    client: TestClient, db_session_factory: Callable[[], Session]
) -> None:
    other = _add_account(db_session_factory, email="grace@example.com")
    account = _add_account(db_session_factory)
    to_account = _invite(db_session_factory, other.organization_id, EMAIL)
    to_someone_else = _invite(db_session_factory, other.organization_id, "linus@example.com")

    assert _delete(client, account, current_password=PASSWORD).status_code == 204

    session = db_session_factory()
    try:
        assert session.get(OrganizationInvitation, to_account) is None
        assert session.get(OrganizationInvitation, to_someone_else) is not None
    finally:
        session.close()


def test_the_bootstrap_operator_cannot_delete_its_account(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    response = client.request(
        "DELETE", f"{API_ROOT}/auth/account", headers=master_key_header, json={"confirm_email": "operator@example.com"}
    )

    assert response.status_code == 400, response.text
    assert "bootstrap operator" in response.json()["detail"]


class _AddsEndUserOnFirstPurge:
    """A telemetry store that records whom it purges, and lets a service key name a new end user during the first."""

    def __init__(self, session_factory: Callable[[], Session], owner_id: str) -> None:
        self._session_factory = session_factory
        self._owner_id = owner_id
        self.late_id: str | None = None
        self.purged: list[str] = []
        self.late_existed_when_purged: bool | None = None

    async def purge_user(self, *, user_id: str) -> None:
        self.purged.append(user_id)
        if user_id == self.late_id:
            self.late_existed_when_purged = await asyncio.to_thread(self._exists, user_id)
        if self.late_id is None:
            self.late_id = str(uuid.uuid4())
            # On a thread of its own: the test's session is synchronous and this runs on the app's event loop.
            await asyncio.to_thread(self._add_end_user, self.late_id)

    def _exists(self, user_id: str) -> bool:
        session = self._session_factory()
        try:
            return session.get(BillingUser, user_id) is not None
        finally:
            session.close()

    def _add_end_user(self, user_id: str) -> None:
        session = self._session_factory()
        try:
            session.add(BillingUser(user_id=user_id, parent_user_id=self._owner_id, external_id="customer-8"))
            session.commit()
        finally:
            session.close()


def test_an_end_user_named_during_the_erasure_is_erased_before_it_is_deleted(
    client: TestClient, db_session_factory: Callable[[], Session]
) -> None:
    account = _add_account(db_session_factory)
    telemetry = _AddsEndUserOnFirstPurge(db_session_factory, str(account.identity_id))
    overrides = cast(Any, client.app).dependency_overrides
    overrides[get_telemetry_storage_port] = lambda: telemetry
    try:
        assert _delete(client, account, current_password=PASSWORD).status_code == 204
    finally:
        overrides.pop(get_telemetry_storage_port, None)

    assert telemetry.late_id is not None
    assert telemetry.purged.count(telemetry.late_id) == 1
    assert telemetry.late_existed_when_purged is True
    session = db_session_factory()
    try:
        assert session.get(BillingUser, telemetry.late_id) is None
    finally:
        session.close()


class _HandsOverDuringPurge:
    """A telemetry store that makes the account being deleted the bootstrap operator while it is purged."""

    def __init__(self, session_factory: Callable[[], Session], identity_id: uuid.UUID) -> None:
        self._session_factory = session_factory
        self._identity_id = identity_id

    async def purge_user(self, *, user_id: str) -> None:
        # On a thread of its own: the test's session is synchronous and this runs on the app's event loop.
        await asyncio.to_thread(self._mark)

    def _mark(self) -> None:
        session = self._session_factory()
        try:
            marker = session.get(RuntimeSetting, BOOTSTRAP_IDENTITY_KEY)
            assert marker is not None
            marker.value = str(self._identity_id)
            session.commit()
        finally:
            session.close()


def test_an_account_handed_the_bootstrap_role_mid_deletion_is_kept(
    client: TestClient, master_key_header: dict[str, str], db_session_factory: Callable[[], Session]
) -> None:
    """The role can move after the request checked it, so deletion checks again under the account's lock."""
    assert client.get(f"{API_ROOT}/organizations/me", headers=master_key_header).status_code == 200
    account = _add_account(db_session_factory)
    overrides = cast(Any, client.app).dependency_overrides
    overrides[get_telemetry_storage_port] = lambda: _HandsOverDuringPurge(db_session_factory, account.identity_id)
    try:
        response = _delete(client, account, current_password=PASSWORD)
    finally:
        overrides.pop(get_telemetry_storage_port, None)
        client.cookies.clear()

    assert response.status_code == 400, response.text
    assert "bootstrap operator" in response.json()["detail"]
    assert _identity_exists(db_session_factory, account.identity_id)
