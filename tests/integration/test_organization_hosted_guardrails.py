"""The hosted guardrails an organization may pick: who sees them, and what they see."""

import uuid
from collections.abc import Mapping, Sequence
from decimal import Decimal
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.config import API_ROOT
from gateway.models.tenancy import Organization, User
from gateway.ports.hosted_guardrail_port import (
    HostedGuardrail,
    HostedGuardrailUnavailableError,
    HostedGuardrailVerdict,
)
from gateway.repositories.tenancy import OrganizationMemberRepository, OrganizationRepository, UserRepository
from gateway.services.tenancy.errors import NotAuthorizedError
from gateway.services.tenancy.organization_hosted_guardrail_service import OrganizationHostedGuardrailService
from gateway.services.tenancy.organization_service import OrganizationService

LAKERA = HostedGuardrail(
    id=uuid.UUID("33333333-3333-3333-3333-333333333333"),
    name="Prompt injection",
    guardrail_name="lakera_guard",
    description="Lakera Guard, run by the deployment",
    price_per_check=Decimal("0.0005"),
)


class OfferingPort:
    """An overlay-bound adapter offering one guardrail to one organization."""

    def __init__(self, offered_to: uuid.UUID | None) -> None:
        self.offered_to = offered_to
        self.asked: list[uuid.UUID | None] = []

    async def list_hosted_guardrails(self, *, organization_id: uuid.UUID | None) -> Sequence[HostedGuardrail]:
        self.asked.append(organization_id)
        return [LAKERA] if organization_id == self.offered_to else []

    async def get_hosted_guardrail(
        self, *, organization_id: uuid.UUID, hosted_guardrail_id: uuid.UUID
    ) -> HostedGuardrail | None:
        return LAKERA if organization_id == self.offered_to and hosted_guardrail_id == LAKERA.id else None

    async def evaluate(
        self,
        *,
        organization_id: uuid.UUID,
        workspace_id: uuid.UUID | None,
        hosted_guardrail_id: uuid.UUID,
        text: str,
        validate_kwargs: Mapping[str, Any],
        idempotency_key: str,
    ) -> HostedGuardrailVerdict:
        raise HostedGuardrailUnavailableError("not under test")


async def _member(db: AsyncSession, organization: Organization, *, role: str) -> User:
    user = await UserRepository(db).create_local_identity(full_name=role, active_organization_id=organization.id)
    await OrganizationMemberRepository(db).create_membership(
        organization_id=organization.id, user_id=user.id, role=role
    )
    return user


def _service(db: AsyncSession, port: OfferingPort) -> OrganizationHostedGuardrailService:
    return OrganizationHostedGuardrailService(
        organizations=OrganizationService(db, membership_listener=None), hosted_guardrails=port
    )


@pytest.mark.asyncio
async def test_an_admin_sees_what_is_offered_to_their_own_organization(async_db: AsyncSession) -> None:
    organization = await OrganizationRepository(async_db).create_organization(
        name="Acme", slug="acme", created_by_user_id=None
    )
    admin = await _member(async_db, organization, role="admin")
    port = OfferingPort(offered_to=organization.id)

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

    listed = await _service(async_db, OfferingPort(offered_to=offered.id)).list_for(user=owner)

    assert listed.data == []


@pytest.mark.asyncio
async def test_a_plain_member_may_not_see_the_list(async_db: AsyncSession) -> None:
    organization = await OrganizationRepository(async_db).create_organization(
        name="Acme", slug="acme", created_by_user_id=None
    )
    member = await _member(async_db, organization, role="member")
    port = OfferingPort(offered_to=organization.id)

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
