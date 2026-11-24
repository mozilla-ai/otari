"""Builders for the routing domain's services."""

from typing import Annotated

from fastapi import Depends

from gateway.api.deps._base import get_unit_of_work
from gateway.core.unit_of_work import UnitOfWork
from gateway.repositories.routing import SubagentDecisionRepository
from gateway.services.routing import SubagentDecisionService


def get_subagent_decision_service(
    uow: Annotated[UnitOfWork, Depends(get_unit_of_work)],
) -> SubagentDecisionService:
    """Build the subagent decision record on the request's Unit of Work."""
    return SubagentDecisionService(uow, SubagentDecisionRepository(uow))


SubagentDecisionServiceDep = Annotated[SubagentDecisionService, Depends(get_subagent_decision_service)]
