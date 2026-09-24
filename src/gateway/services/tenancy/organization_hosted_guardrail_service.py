"""The hosted guardrails an organization may pick for its mandates.

A hosted guardrail belongs to the deployment, not the organization, so the
rows live behind ``HostedGuardrailPort`` and nothing here reads a table. What
this adds is the organization's side: whose view of the list the caller gets,
and who may see it at all.
"""

import uuid

from pydantic import BaseModel

from gateway.models.tenancy import User
from gateway.ports.hosted_guardrail_port import HostedGuardrail, HostedGuardrailPort
from gateway.services.tenancy.organization_service import OrganizationService


class HostedGuardrailPublic(BaseModel):
    """A hosted guardrail as the picker shows it. Never a secret or an argument."""

    id: uuid.UUID
    name: str
    guardrail_name: str
    description: str | None
    price_per_check: float | None

    @classmethod
    def from_port(cls, guardrail: HostedGuardrail) -> "HostedGuardrailPublic":
        price = guardrail.price_per_check
        return cls(
            id=guardrail.id,
            name=guardrail.name,
            guardrail_name=guardrail.guardrail_name,
            description=guardrail.description,
            price_per_check=None if price is None else float(price),
        )


class HostedGuardrailsPublic(BaseModel):
    """Every hosted guardrail the caller's organization may pick."""

    data: list[HostedGuardrailPublic]


class OrganizationHostedGuardrailService:
    """Lists hosted guardrails for the people who may mandate one."""

    def __init__(self, *, organizations: OrganizationService, hosted_guardrails: HostedGuardrailPort) -> None:
        self.organizations = organizations
        self.hosted_guardrails = hosted_guardrails

    async def list_for(self, *, user: User) -> HostedGuardrailsPublic:
        """The hosted guardrails the caller's organization may pick.

        Gated like the mandates themselves: only someone who may mandate a
        guardrail needs to see what can be mandated.
        """
        organization = await self.organizations.get_active_organization_for_user(user)
        await self.organizations.require_active_organization_management_access(user=user, organization=organization)
        offered = await self.hosted_guardrails.list_hosted_guardrails(organization_id=organization.id)
        return HostedGuardrailsPublic(data=[HostedGuardrailPublic.from_port(guardrail) for guardrail in offered])
