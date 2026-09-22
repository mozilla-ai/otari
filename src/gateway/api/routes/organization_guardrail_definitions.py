"""The guardrails the caller's organization defined for Otari to run itself (standalone mode only).

Thin composition over
`gateway.services.tenancy.organization_guardrail_definition_service`: resolve the
caller's identity, call the service, return its typed result. The role gate, the
catalog rules and the secret handling live there, and the domain errors it raises
carry their own statuses (see `gateway.services.tenancy.errors`), so nothing here
catches them.

A separate surface from ``/api/v1/organizations/me/guardrails`` because the two
say different things. A definition is *what* a check is, and one of them can be
mandated under several profiles; a mandate is *where* a check runs and how hard
it bites. The keys differ with them, which is why the tables do.

Scoped to ``/me`` for the reason the mandates next door are: a standalone
deployment has one organization and the caller's identity already points at it,
so a request cannot name one.
"""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Query, status

from gateway.api.deps import (
    CurrentIdentity,
    OrganizationGuardrailDefinitionServiceDep,
    verify_master_key,
)
from gateway.api.routes.organizations import Message
from gateway.services.tenancy.organization_guardrail_definition_service import (
    OrganizationGuardrailDefinitionCreate,
    OrganizationGuardrailDefinitionPublic,
    OrganizationGuardrailDefinitionsPublic,
    OrganizationGuardrailDefinitionUpdate,
)

# Master key on the router, as every standalone management router declares it.
# The role gate is a separate question answered in the service: the credential
# says a request is the operator's, the membership says whether that identity may
# spend the organization's vendor accounts.
router = APIRouter(
    prefix="/organizations/me/guardrail-definitions",
    tags=["organization-guardrail-definitions"],
    dependencies=[Depends(verify_master_key)],
)


@router.get("")
async def list_organization_guardrail_definitions(
    service: OrganizationGuardrailDefinitionServiceDep,
    current_identity: CurrentIdentity,
    skip: Annotated[int, Query(ge=0, description="Number of records to skip")] = 0,
    limit: Annotated[int, Query(ge=1, le=1000, description="Maximum number of records to return")] = 100,
) -> OrganizationGuardrailDefinitionsPublic:
    """List the guardrails the caller's organization has defined.

    Organization owners and admins only. A stored vendor credential is never
    returned: each one comes back as ``***`` under its own name, which is what a
    form resubmits to keep it.
    """
    return await service.list_definitions(user=current_identity, skip=skip, limit=limit)


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_organization_guardrail_definition(
    service: OrganizationGuardrailDefinitionServiceDep,
    current_identity: CurrentIdentity,
    body: OrganizationGuardrailDefinitionCreate,
) -> OrganizationGuardrailDefinitionPublic:
    """Define a guardrail this deployment will build and call itself. Organization owners and admins only.

    ``guardrail_name`` must be one the built-in guardrail catalog lists
    (``GET /api/v1/tool-settings/guardrails/catalog``), and
    ``create_kwargs`` must satisfy that guardrail's constructor as the catalog
    describes it: no argument it does not declare, nothing it types as a live
    object rather than configuration, and every required argument no environment
    variable can supply. Arguments the catalog marks secret are encrypted at
    rest.

    A definition on its own changes no request. Mandate it through
    ``/api/v1/organizations/me/guardrails`` for it to run.
    """
    return await service.create_definition(user=current_identity, request=body)


@router.patch("/{definition_id}")
async def update_organization_guardrail_definition(
    service: OrganizationGuardrailDefinitionServiceDep,
    current_identity: CurrentIdentity,
    definition_id: uuid.UUID,
    body: OrganizationGuardrailDefinitionUpdate,
) -> OrganizationGuardrailDefinitionPublic:
    """Change a definition's name, guardrail, build arguments, or enabled flag.

    Organization owners and admins only. Omitted fields are left as they are.
    ``create_kwargs`` replaces the arguments whole when sent, an argument sent as
    ``***`` keeps the value stored under that name, and omitting the field
    entirely leaves the stored credentials untouched and unread.
    """
    return await service.update_definition(user=current_identity, definition_id=definition_id, request=body)


@router.delete("/{definition_id}")
async def delete_organization_guardrail_definition(
    service: OrganizationGuardrailDefinitionServiceDep,
    current_identity: CurrentIdentity,
    definition_id: uuid.UUID,
) -> Message:
    """Drop a definition and the credentials it holds.

    Organization owners and admins only. Use ``enabled: false`` instead to stop
    the guardrail everywhere while keeping both.
    """
    await service.delete_definition(user=current_identity, definition_id=definition_id)
    return Message(message="Organization guardrail definition deleted")
