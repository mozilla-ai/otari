"""Builders for the routing domain's services."""

import uuid
from typing import Annotated, Any

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps._base import PortSessionDep, get_unit_of_work
from gateway.core.unit_of_work import UnitOfWork
from gateway.repositories.routing import SubagentDecisionRepository
from gateway.services.pricing_service import resolve_model_pricing
from gateway.services.routing import SubagentDecisionService, SubagentRateLookup
from gateway.services.workspace_scope import organization_for_workspace_id


def _subagent_rate_lookup(db: AsyncSession) -> SubagentRateLookup:
    """The rate a model is priced at for a workspace's organization, read on the request's session."""

    async def rate(model_key: str, workspace_id: uuid.UUID) -> Any:
        provider, _, model = model_key.partition(":")
        organization_id = await organization_for_workspace_id(db, workspace_id)
        resolved = await resolve_model_pricing(db, provider, model, organization_id=organization_id)
        return resolved.pricing if resolved is not None else None

    return rate


def get_subagent_decision_service(
    uow: Annotated[UnitOfWork, Depends(get_unit_of_work)], db: PortSessionDep
) -> SubagentDecisionService:
    """Build the subagent decision record on the request's Unit of Work."""
    return SubagentDecisionService(uow, SubagentDecisionRepository(uow), _subagent_rate_lookup(db))


SubagentDecisionServiceDep = Annotated[SubagentDecisionService, Depends(get_subagent_decision_service)]
