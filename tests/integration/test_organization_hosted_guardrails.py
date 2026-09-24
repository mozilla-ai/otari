"""The hosted guardrails an organization may pick: who sees them, and what they see."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.config import API_ROOT
from gateway.models.tenancy import Organization, User
from gateway.repositories.tenancy import OrganizationMemberRepository, OrganizationRepository, UserRepository
from gateway.services.tenancy.errors import NotAuthorizedError
from gateway.services.tenancy.organization_hosted_guardrail_service import OrganizationHostedGuardrailService
from gateway.services.tenancy.organization_service import OrganizationService

from .hosted_guardrail_helpers import LAKERA, HostedGuardrails


async def _member(db: AsyncSession, organization: Organization, *, role: str) -> User:
    user = await UserRepository(db).create_local_identity(full_name=role, active_organization_id=organization.id)
    await OrganizationMemberRepository(db).create_membership(
        organization_id=organization.id, user_id=user.id, role=role
    )
    return user


def _service(db: AsyncSession, port: HostedGuardrails) -> OrganizationHostedGuardrailService:
    return OrganizationHostedGuardrailService(
        organizations=OrganizationService(db, membership_listener=None), hosted_guardrails=port
    )


@pytest.mark.asyncio
async def test_an_admin_sees_what_is_offered_to_their_own_organization(async_db: AsyncSession) -> None:
    organization = await OrganizationRepository(async_db).create_organization(
        name="Acme", slug="acme", created_by_user_id=None
    )
    admin = await _member(async_db, organization, role="admin")
    port = HostedGuardrails(offered_to=organization.id)

    listed = await _service(async_db, port).list_for(user=admin)

    assert port.asked == [organization.id]
    assert [guardrail.model_dump() for guardrail in listed.data] == [
        {
            "id": LAKERA.id,
            "name": "Prompt injection",
            "guardrail_name": "lakera_guard",
            "description": "Lakera Guard, run by the deployment",
            "price_per_check": 0.0005,
        }
    ]


@pytest.mark.asyncio
async def test_another_organization_sees_nothing_offered_elsewhere(async_db: AsyncSession) -> None:
    repository = OrganizationRepository(async_db)
    offered = await repository.create_organization(name="Acme", slug="acme", created_by_user_id=None)
    other = await repository.create_organization(name="Other", slug="other", created_by_user_id=None)
    owner = await _member(async_db, other, role="owner")

    listed = await _service(async_db, HostedGuardrails(offered_to=offered.id)).list_for(user=owner)

    assert listed.data == []


@pytest.mark.asyncio
async def test_a_plain_member_may_not_see_the_list(async_db: AsyncSession) -> None:
    organization = await OrganizationRepository(async_db).create_organization(
        name="Acme", slug="acme", created_by_user_id=None
    )
    member = await _member(async_db, organization, role="member")
    port = HostedGuardrails(offered_to=organization.id)

    with pytest.raises(NotAuthorizedError):
        await _service(async_db, port).list_for(user=member)
    assert port.asked == []


def test_the_plain_build_offers_nothing_over_http(client: TestClient, master_key_header: dict[str, str]) -> None:
    response = client.get(f"{API_ROOT}/organizations/me/hosted-guardrails", headers=master_key_header)

    assert response.status_code == 200, response.text
    assert response.json() == {"data": []}


def test_the_list_needs_a_credential(client: TestClient) -> None:
    for headers in ({}, {"Authorization": "Bearer not-a-key"}):
        assert client.get(f"{API_ROOT}/organizations/me/hosted-guardrails", headers=headers).status_code == 401
