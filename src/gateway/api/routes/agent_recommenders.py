"""The Subagent router operator API.

The stored decision models that recommend a subagent's model when a coding
agent asks ``POST /api/v1/routing/recommend``: which provider and model, on
what key, and the two token rates the caller is charged at. One is switched
on at a time.

Two gates on the router, as on hosted providers: ``verify_master_key`` asks
whether the caller authenticated, and the operator gate whether that caller
may act deployment-wide, refusing everyone else with the 404 ``/api/v1/admin``
answers. Declared on the router so a route added later inherits both.

**This surface never returns a key.** A router is known by its provider, its
model and the last four characters of its key, and that is all a client ever
learns about it.
"""

import uuid

from fastapi import APIRouter, Depends, status

from gateway.api.deps import AgentRecommenderServiceDep, require_deployment_operator_or_absent, verify_master_key
from gateway.core.surface import Surface
from gateway.schemas.routing import (
    AgentRecommenderCreate,
    AgentRecommenderPublic,
    AgentRecommendersPublic,
    AgentRecommenderUpdate,
)

router = APIRouter(
    prefix="/routing/recommenders",
    tags=["agent-recommenders"],
    dependencies=[Depends(verify_master_key), Depends(require_deployment_operator_or_absent)],
)

# The decision models that pick a subagent's model. The page is operator-only.
SURFACE = Surface("agent_recommenders")


@router.get("")
async def list_agent_recommenders(service: AgentRecommenderServiceDep) -> AgentRecommendersPublic:
    """List the subagent routers, and the configured decision model that answers while none is stored."""
    return await service.list()


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_agent_recommender(
    service: AgentRecommenderServiceDep, body: AgentRecommenderCreate
) -> AgentRecommenderPublic:
    """Add a subagent router: a decision model, its key, and the prices a recommendation is charged at.

    The first router added is switched on; a later one waits to be switched
    on in place of the one that is.
    """
    return await service.create(body)


@router.patch("/{recommender_id}")
async def update_agent_recommender(
    service: AgentRecommenderServiceDep, recommender_id: uuid.UUID, body: AgentRecommenderUpdate
) -> AgentRecommenderPublic:
    """Change a subagent router, or switch it on in place of the one that is.

    An omitted field keeps its stored value, the key included while the
    provider and the API base stay the same.
    """
    return await service.update(recommender_id, body)


@router.delete("/{recommender_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_agent_recommender(service: AgentRecommenderServiceDep, recommender_id: uuid.UUID) -> None:
    """Remove a subagent router and its key. With none left, the configured decision model answers again."""
    await service.delete(recommender_id)
