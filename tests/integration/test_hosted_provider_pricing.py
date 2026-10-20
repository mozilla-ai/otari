"""A request served on a hosted provider is priced at a rate only the deployment operator sets.

The write-time guard (``OrganizationPricingService.raise_if_deployment_supplied``)
refuses an organization's override for a model the deployment serves on its own
credential, but it runs once, when the override is written. An override written
while the organization's own key covered the model, or before the operator added
the hosted provider, outlives that state. These tests pin that settlement reads
no such override for a deployment-paid request, while a rate the operator set for
the organization still applies and stays the operator's to change.
"""

import uuid
from collections.abc import Callable, Generator, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from any_llm.types.completion import ChatCompletion, ChatCompletionMessage, Choice, CompletionUsage
from any_llm.types.model import Model
from fastapi import status
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from gateway.core.config import API_KEY_HEADER, API_ROOT, GatewayConfig
from gateway.models.provider_keys import OrgProviderKey
from gateway.models.tenancy import DashboardSession, Organization, OrganizationMember, User, Workspace
from gateway.models.usage import UsageLog
from gateway.services.dashboard_session_service import SESSION_COOKIE_NAME, hash_session_token
from gateway.services.model_discovery_service import ProviderDiscovery
from gateway.services.secret_box import encrypt_secret, generate_secret_key

from .conftest import build_test_client, shared_test_config


@pytest.fixture(scope="module")
def test_config(postgres_url: str) -> GatewayConfig:
    """The shared configuration declaring no provider, so nothing configured shadows a hosted one."""
    return shared_test_config(postgres_url, {})


HOSTED_KEY = "sk-live-hosted-1234"
MODEL = "openai:gpt-4o"
# 1000 prompt and 500 completion tokens at the deployment's 2.5 / 10 per million.
DEPLOYMENT_COST = Decimal("0.0075")


@pytest.fixture(autouse=True)
def _keys(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """No ambient provider key, which would credential the ladder ahead of the port, and a real secret box."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    yield


@pytest.fixture(autouse=True)
def _discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    """The hosted key lists the one model these tests ask for, so the roster offers it."""

    async def _stub(impl_name: str, **_: object) -> ProviderDiscovery:
        return ProviderDiscovery(
            provider=impl_name, models=[Model(id="gpt-4o", object="model", created=0, owned_by=impl_name)]
        )

    monkeypatch.setattr("gateway.services.providers._hosted_provider_service.test_provider_credentials", _stub)


@dataclass
class _Tenant:
    organization_id: uuid.UUID
    owner: str
    operator: str
    byo_key_id: uuid.UUID | None = None


def _session(session: Session, *, email: str, organization_id: uuid.UUID, is_superuser: bool = False) -> str:
    user = User(email=email, full_name=email, active_organization_id=organization_id, is_superuser=is_superuser)
    session.add(user)
    session.commit()
    session.refresh(user)
    session.add(OrganizationMember(organization_id=organization_id, user_id=user.id, role="owner", status="active"))
    token = f"otari-sess-{email}"
    session.add(
        DashboardSession(
            token_hash=hash_session_token(token),
            user_id=user.id,
            created_at=datetime.now(UTC),
            expires_at=datetime.now(UTC) + timedelta(hours=12),
        )
    )
    session.commit()
    return token


def _provision(client: TestClient, master: dict[str, str]) -> None:
    """Let the deployment provision its own organization before a second one exists beside it."""
    assert client.get(f"{API_ROOT}/organizations/me", headers=master).status_code == status.HTTP_200_OK


@pytest.fixture
def tenant(client: TestClient, master_key_header: dict[str, str], db_session_factory: Callable[[], Session]) -> _Tenant:
    """An organization whose owner does not operate the deployment, and an operator acting in it."""
    _provision(client, master_key_header)
    session = db_session_factory()
    try:
        organization = Organization(name="Carol", slug="carol")
        session.add(organization)
        session.commit()
        session.refresh(organization)
        session.add(Workspace(name="Carol one", organization_id=organization.id))
        session.commit()
        return _Tenant(
            organization_id=organization.id,
            owner=_session(session, email="carol@carol.test", organization_id=organization.id),
            operator=_session(session, email="root@carol.test", organization_id=organization.id, is_superuser=True),
        )
    finally:
        session.close()


def _as(client: TestClient, token: str, method: str, path: str, **kwargs: Any) -> Any:
    client.cookies.set(SESSION_COOKIE_NAME, token)
    try:
        return client.request(method, f"{API_ROOT}{path}", **kwargs)
    finally:
        client.cookies.clear()


def _byo_key(client: TestClient, db_session_factory: Callable[[], Session], tenant: _Tenant) -> None:
    """The organization's own openai key, stored directly so creating it dials nothing."""
    session = db_session_factory()
    try:
        key = OrgProviderKey(
            organization_id=tenant.organization_id,
            provider="openai",
            name="openai-own",
            encrypted_api_key=encrypt_secret("sk-carol-own"),
            last4="1234",
            is_org_default=True,
        )
        session.add(key)
        session.commit()
        tenant.byo_key_id = key.id
    finally:
        session.close()
    # Written behind the service's back, so a write through it reloads the request path's key cache.
    marked = _as(client, tenant.owner, "POST", f"/organizations/me/provider-keys/{tenant.byo_key_id}/default")
    assert marked.status_code == status.HTTP_200_OK, marked.text


def _deployment_price(client: TestClient, headers: dict[str, str], model_key: str = MODEL) -> None:
    response = client.post(
        f"{API_ROOT}/pricing",
        json={"model_key": model_key, "input_price_per_million": 2.5, "output_price_per_million": 10},
        headers=headers,
    )
    assert response.status_code == status.HTTP_200_OK, response.text


def _hosted_openai(client: TestClient, headers: dict[str, str]) -> None:
    response = client.post(
        f"{API_ROOT}/hosted-providers", json={"provider": "openai", "api_key": HOSTED_KEY}, headers=headers
    )
    assert response.status_code == status.HTTP_201_CREATED, response.text


def _override(client: TestClient, token: str, rate: float, model_key: str = MODEL) -> Any:
    return _as(
        client,
        token,
        "POST",
        "/organizations/me/pricing",
        json={"model_key": model_key, "input_price_per_million": rate, "output_price_per_million": rate},
    )


def _api_key(client: TestClient, tenant: _Tenant) -> dict[str, str]:
    minted = _as(client, tenant.owner, "POST", "/organizations/me/keys", json={"key_name": "carol key"})
    assert minted.status_code == status.HTTP_200_OK, minted.text
    return {API_KEY_HEADER: f"Bearer {minted.json()['key']}"}


def _complete(client: TestClient, headers: dict[str, str], model: str = MODEL) -> tuple[Any, dict[str, object]]:
    captured: dict[str, object] = {}
    completion = ChatCompletion(
        id="chatcmpl-hosted",
        object="chat.completion",
        created=0,
        model=model.split(":", 1)[1],
        choices=[Choice(index=0, message=ChatCompletionMessage(role="assistant", content="hi"), finish_reason="stop")],
        usage=CompletionUsage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500),
    )

    async def fake_acompletion(**kwargs: object) -> ChatCompletion:
        captured.update(kwargs)
        return completion

    with patch("gateway.api.routes.chat.acompletion", new=AsyncMock(side_effect=fake_acompletion)):
        response = client.post(
            f"{API_ROOT}/chat/completions",
            json={"model": model, "messages": [{"role": "user", "content": "Hi"}]},
            headers=headers,
        )
    return response, captured


def _usage_rows(db_session_factory: Callable[[], Session]) -> list[UsageLog]:
    with db_session_factory() as db:
        return list(db.scalars(select(UsageLog).where(UsageLog.model == "gpt-4o")))


def test_an_override_written_under_a_byo_key_does_not_price_a_hosted_request(
    client: TestClient,
    master_key_header: dict[str, str],
    tenant: _Tenant,
    db_session_factory: Callable[[], Session],
) -> None:
    """The reproduction: refused, then written under the organization's own key, then the key archived."""
    _deployment_price(client, master_key_header)
    _hosted_openai(client, master_key_header)

    refused = _override(client, tenant.owner, 0)
    assert refused.status_code == status.HTTP_403_FORBIDDEN, refused.text

    _byo_key(client, db_session_factory, tenant)
    written = _override(client, tenant.owner, 0)
    assert written.status_code == status.HTTP_201_CREATED, written.text
    archived = _as(client, tenant.owner, "POST", f"/organizations/me/provider-keys/{tenant.byo_key_id}/archive")
    assert archived.status_code == status.HTTP_200_OK, archived.text

    response, captured = _complete(client, _api_key(client, tenant))

    assert response.status_code == status.HTTP_200_OK, response.text
    assert captured["api_key"] == HOSTED_KEY, "served on the deployment's credential"
    [row] = _usage_rows(db_session_factory)
    assert row.deployment_paid is True
    assert row.cost == DEPLOYMENT_COST
    assert row.pricing_source == "deployment"


def test_an_override_written_before_the_provider_was_hosted_does_not_price_a_hosted_request(
    client: TestClient,
    master_key_header: dict[str, str],
    tenant: _Tenant,
    db_session_factory: Callable[[], Session],
) -> None:
    """Nothing served the model when the override was written, so the guard had nothing to refuse."""
    _deployment_price(client, master_key_header)
    written = _override(client, tenant.owner, 0)
    assert written.status_code == status.HTTP_201_CREATED, written.text
    _hosted_openai(client, master_key_header)

    response, captured = _complete(client, _api_key(client, tenant))

    assert response.status_code == status.HTTP_200_OK, response.text
    assert captured["api_key"] == HOSTED_KEY
    [row] = _usage_rows(db_session_factory)
    assert row.cost == DEPLOYMENT_COST
    assert row.pricing_source == "deployment"


def test_the_organizations_own_override_still_prices_a_request_on_its_own_key(
    client: TestClient,
    master_key_header: dict[str, str],
    tenant: _Tenant,
    db_session_factory: Callable[[], Session],
) -> None:
    """The control: the organization's rate is what it pays for a model it supplies the key for."""
    _deployment_price(client, master_key_header)
    _hosted_openai(client, master_key_header)
    _byo_key(client, db_session_factory, tenant)
    assert _override(client, tenant.owner, 1).status_code == status.HTTP_201_CREATED

    response, captured = _complete(client, _api_key(client, tenant))

    assert response.status_code == status.HTTP_200_OK, response.text
    assert captured["api_key"] == "sk-carol-own"
    [row] = _usage_rows(db_session_factory)
    assert row.deployment_paid is False
    assert row.cost == Decimal("0.0015")
    assert row.pricing_source == "organization"


def test_a_rate_the_operator_set_prices_a_hosted_request_and_stays_the_operators(
    client: TestClient,
    master_key_header: dict[str, str],
    tenant: _Tenant,
    db_session_factory: Callable[[], Session],
) -> None:
    """A negotiated rate is the operator's to set for an organization, and the organization's to read, not change."""
    _deployment_price(client, master_key_header)
    _hosted_openai(client, master_key_header)
    written = _override(client, tenant.operator, 1)
    assert written.status_code == status.HTTP_201_CREATED, written.text
    pricing_id = written.json()["id"]

    response, _ = _complete(client, _api_key(client, tenant))

    assert response.status_code == status.HTTP_200_OK, response.text
    [row] = _usage_rows(db_session_factory)
    assert row.cost == Decimal("0.0015")
    assert row.pricing_source == "organization"

    body = {
        "input_price_per_million": 0,
        "output_price_per_million": 0,
        "effective_from": written.json()["effective_from"],
    }
    edited = _as(client, tenant.owner, "PUT", f"/organizations/me/pricing/{pricing_id}", json=body)
    assert edited.status_code == status.HTTP_403_FORBIDDEN, edited.text
    deleted = _as(client, tenant.owner, "DELETE", f"/organizations/me/pricing/{pricing_id}")
    assert deleted.status_code == status.HTTP_403_FORBIDDEN, deleted.text
    # Still in force, so the next request is priced exactly as the first.
    response, _ = _complete(client, _api_key(client, tenant))
    assert response.status_code == status.HTTP_200_OK, response.text
    assert {row.cost for row in _usage_rows(db_session_factory)} == {Decimal("0.0015")}


@pytest.fixture
def strict_client(postgres_url: str) -> Generator[TestClient]:
    """A gateway that refuses to serve what it cannot price, and consults no community dataset."""
    config = GatewayConfig(
        database_url=postgres_url,
        master_key="test-master-key",
        auto_migrate=False,
        require_pricing=True,
        default_pricing=False,
        model_discovery=False,
    )
    yield from build_test_client(config)


def test_a_zero_override_cannot_carry_an_unpriced_hosted_model_past_require_pricing(
    strict_client: TestClient, db_session_factory: Callable[[], Session]
) -> None:
    """Admission priced the selector at the organization's zero; the hosted request settles at no rate at all."""
    master = {API_KEY_HEADER: "Bearer test-master-key"}
    _provision(strict_client, master)
    session = db_session_factory()
    try:
        organization = Organization(name="Dave", slug="dave")
        session.add(organization)
        session.commit()
        session.refresh(organization)
        session.add(Workspace(name="Dave one", organization_id=organization.id))
        session.commit()
        tenant = _Tenant(
            organization_id=organization.id,
            owner=_session(session, email="dave@dave.test", organization_id=organization.id),
            operator="",
        )
    finally:
        session.close()
    unpriced = "openai:gpt-4o"
    assert _override(strict_client, tenant.owner, 0, unpriced).status_code == status.HTTP_201_CREATED
    _hosted_openai(strict_client, master)
    # Nothing prices it, so it arrives switched off; the operator serves it anyway.
    [offered] = strict_client.get(f"{API_ROOT}/hosted-providers/openai/models", headers=master).json()["data"]
    switched_on = strict_client.patch(
        f"{API_ROOT}/hosted-providers/openai/models/{offered['id']}", json={"enabled": True}, headers=master
    )
    assert switched_on.status_code == status.HTTP_200_OK, switched_on.text

    response, captured = _complete(strict_client, _api_key(strict_client, tenant), unpriced)

    assert response.status_code == status.HTTP_402_PAYMENT_REQUIRED, response.text
    assert not captured, "the provider was called for a request nothing prices"
    assert [row.status_code for row in _usage_rows(db_session_factory)] == [status.HTTP_402_PAYMENT_REQUIRED]
