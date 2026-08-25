"""The caller's organization's guardrails (standalone mode only).

Thin composition over `gateway.services.tenancy.organization_guardrail_service`:
resolve the caller's identity, call the service, return its typed result. The
role gate and the scope rules live there, and the domain errors it raises carry
their own statuses (see `gateway.services.tenancy.errors`), so nothing here
catches them.

Scoped to ``/me`` for the reason `routes/organization_pricing.py` and
`routes/organizations.py` are: a standalone deployment has exactly one
organization and the caller's identity already points at it, so a request cannot
name one. Multi-organization administration is the overlay's to contribute.

These entries sit *above* ``/v1/tool-settings``, which stays the deployment's own
guardrail configuration. ``guardrails_url`` there is still what a guardrail
without an endpoint of its own is sent to, and an organization with no entries
changes nothing about how a request is checked.
"""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import CurrentIdentity, get_db, verify_master_key
from gateway.api.routes.organizations import Message
from gateway.services.tenancy.organization_guardrail_service import (
    OrganizationGuardrailCreate,
    OrganizationGuardrailPublic,
    OrganizationGuardrailService,
    OrganizationGuardrailsPublic,
    OrganizationGuardrailUpdate,
)

# Master key on the router, as every standalone management router declares it.
# The role gate is a separate question answered in the service: the credential
# says a request is the operator's, the membership says whether that identity may
# change what every workspace of the organization is checked against.
router = APIRouter(
    prefix="/v1/organizations/me/guardrails",
    tags=["organization-guardrails"],
    dependencies=[Depends(verify_master_key)],
)


def get_organization_guardrail_service(db: Annotated[AsyncSession, Depends(get_db)]) -> OrganizationGuardrailService:
    """Build the service on the request's session."""
    return OrganizationGuardrailService(db)


OrganizationGuardrailServiceDep = Annotated[OrganizationGuardrailService, Depends(get_organization_guardrail_service)]


@router.get("")
async def list_organization_guardrails(
    service: OrganizationGuardrailServiceDep,
    current_identity: CurrentIdentity,
    skip: Annotated[int, Query(ge=0, description="Number of records to skip")] = 0,
    limit: Annotated[int, Query(ge=1, le=1000, description="Maximum number of records to return")] = 100,
) -> OrganizationGuardrailsPublic:
    """List the guardrails the caller's organization mandates.

    Organization owners and admins only, unlike the pricing overrides next door
    that any member may read: these rows name the endpoints this gateway
    connects to and say which of them carry a credential. A credential is never
    returned, only whether one is set.
    """
    return await service.list_guardrails(user=current_identity, skip=skip, limit=limit)


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_organization_guardrail(
    service: OrganizationGuardrailServiceDep,
    current_identity: CurrentIdentity,
    body: OrganizationGuardrailCreate,
) -> OrganizationGuardrailPublic:
    """Mandate a guardrail across the organization. Organization owners and admins only.

    The guardrail runs on every request from the workspaces it is scoped to, in
    addition to whatever the caller asked for, with the stricter of the two
    settings applying to a profile both name. Set
    ``applies_to_all_workspaces`` for it to cover workspaces created later;
    otherwise a new workspace inherits nothing and the entry runs only in the
    workspaces ``workspace_ids`` lists.
    """
    return await service.create_guardrail(user=current_identity, request=body)


@router.patch("/{guardrail_id}")
async def update_organization_guardrail(
    service: OrganizationGuardrailServiceDep,
    current_identity: CurrentIdentity,
    guardrail_id: uuid.UUID,
    body: OrganizationGuardrailUpdate,
) -> OrganizationGuardrailPublic:
    """Change a guardrail's profile, endpoint, credential, modes, or scope.

    Organization owners and admins only. Omitted fields are left as they are;
    ``workspace_ids`` replaces the scope whole when sent, and ``url`` and
    ``credential`` are cleared by sending an empty string rather than null.
    """
    return await service.update_guardrail(user=current_identity, guardrail_id=guardrail_id, request=body)


@router.delete("/{guardrail_id}")
async def delete_organization_guardrail(
    service: OrganizationGuardrailServiceDep,
    current_identity: CurrentIdentity,
    guardrail_id: uuid.UUID,
) -> Message:
    """Stop mandating a guardrail, discarding its credential and scope.

    Organization owners and admins only. Use ``enabled: false`` instead to stop
    it everywhere while keeping both.
    """
    await service.delete_guardrail(user=current_identity, guardrail_id=guardrail_id)
    return Message(message="Organization guardrail deleted")
