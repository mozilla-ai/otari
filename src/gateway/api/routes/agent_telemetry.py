"""Operator purge of previously-captured behavioral-event rows."""

from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import get_db, verify_master_key
from gateway.services.agent_telemetry_admin_service import (
    AgentTelemetryDeleteRequest,
    AgentTelemetryDeleteResult,
    delete_agent_telemetry,
)

router = APIRouter(prefix="/v1/agent-telemetry", tags=["agent-telemetry"])


@router.delete("", dependencies=[Depends(verify_master_key)])
async def delete_agent_telemetry_rows(
    request: AgentTelemetryDeleteRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> AgentTelemetryDeleteResult:
    """Delete agent_telemetry rows by explicit ids or by filter (standalone).

    Target either an explicit selection (`ids`) or everything matching a filter
    (`by_filter: true` plus optional `user_id` / `api_key_id` / `name` / date
    range). A selection matching zero rows succeeds with `deleted: 0`.
    Master-key only.
    """
    return await delete_agent_telemetry(db, request)
