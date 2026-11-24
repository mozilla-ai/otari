"""The record of each subagent model recommendation the gateway answered.

A row is what a later reading of the recommendation's worth starts from: the
model the subagent would have started on, the model recommended, and what the
decision call cost.
"""

import uuid
from decimal import Decimal

from gateway.core.database import DATABASE_ERRORS, describe_database_error
from gateway.core.unit_of_work import UnitOfWork
from gateway.log_config import logger
from gateway.models.routing import BASELINE_PARENT, BASELINE_REQUESTED
from gateway.ports.agent_model_recommender_port import SubagentSpawn
from gateway.repositories.routing import SubagentDecisionRepository

# Claude Code's spelling of "the parent's model" in an agent definition.
_INHERIT = "inherit"


def _baseline(spawn: SubagentSpawn) -> tuple[str, str]:
    """The model the subagent would have started on with no recommendation, and where that came from."""
    requested = spawn.requested_model
    if requested and requested != _INHERIT:
        return requested, BASELINE_REQUESTED
    return spawn.parent_model, BASELINE_PARENT


class SubagentDecisionService:
    """Keep what each answered recommendation decided."""

    def __init__(self, uow: UnitOfWork, decisions: SubagentDecisionRepository) -> None:
        self._uow = uow
        self._rows = decisions

    async def record(
        self,
        *,
        decision_id: uuid.UUID,
        workspace_id: uuid.UUID,
        user_id: str | None,
        api_key_id: str | None,
        spawn: SubagentSpawn,
        recommended_model: str,
        router_cost: Decimal | None,
    ) -> uuid.UUID | None:
        """Store one answered recommendation and return its id, or None where it could not be stored.

        The recommendation was already answered and charged, so a failed write
        costs the caller nothing but this row. The task and its description are
        left out.
        """
        baseline_model, baseline_source = _baseline(spawn)
        try:
            async with self._uow:
                await self._rows.insert(
                    id=decision_id,
                    workspace_id=workspace_id,
                    user_id=user_id,
                    api_key_id=api_key_id,
                    harness=spawn.harness,
                    agent_type=spawn.agent_type,
                    requested_model=spawn.requested_model,
                    parent_model=spawn.parent_model,
                    baseline_model=baseline_model,
                    baseline_source=baseline_source,
                    recommended_model=recommended_model,
                    router_cost=router_cost,
                )
        except DATABASE_ERRORS as exc:
            logger.warning("Subagent decision %s was not recorded: %s", decision_id, describe_database_error(exc))
            return None
        return decision_id
