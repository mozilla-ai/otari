"""The hosted guardrails the caller's organization may pick for a mandate.

A build with nothing hosted answers an empty list, which is how the dashboard
knows to offer no hosted choice. So the route needs no capability gate: the
core adapter's answer is the plain build's whole behavior.
"""

from fastapi import APIRouter, Depends

from gateway.api.deps import CurrentIdentity, OrganizationHostedGuardrailServiceDep, verify_master_key
from gateway.services.tenancy.organization_hosted_guardrail_service import HostedGuardrailsPublic

router = APIRouter(
    prefix="/organizations/me/hosted-guardrails",
    tags=["organization-guardrails"],
    dependencies=[Depends(verify_master_key)],
)


@router.get("")
async def list_organization_hosted_guardrails(
    service: OrganizationHostedGuardrailServiceDep,
    current_identity: CurrentIdentity,
) -> HostedGuardrailsPublic:
    """List the deployment's hosted guardrails the caller's organization may pick.

    Organization owners and admins only, like the mandates they are picked for.
    Each entry names the guardrail and its price per check, never its secret
    or arguments.
    """
    return await service.list_for(user=current_identity)
