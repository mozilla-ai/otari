"""Operator purge of previously-captured behavioral-event rows.

`agent_telemetry` never counts toward budget or spend, so this needs none of
`usage_admin_service`'s imported-only safety scoping: a fresh, narrower
selection schema (mirroring its `ids`-or-`by_filter` shape) is simpler than
threading a generic filter through fields that would not apply here. The
filter set is deliberately scoped to fields a behavioral-event row actually
has: no `model`, `provider`, `status`, `source_label`, `priced`, or `tool`.
"""

from datetime import datetime
from typing import Annotated, Any, cast

from pydantic import BaseModel, Field, model_validator
from sqlalchemy import ColumnElement, delete
from sqlalchemy.engine import CursorResult
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.sql import MAX_FILTER_VALUES, match_any
from gateway.log_config import logger
from gateway.models.entities import AgentTelemetry

# Matches UsageSelection's ids cap: page selections drive this path, and 1000
# leaves headroom without letting a single request name an unbounded set.
_MAX_IDS = 1000

_CappedValues = Annotated[list[str], Field(max_length=MAX_FILTER_VALUES)]


class AgentTelemetrySelection(BaseModel):
    """Which `agent_telemetry` rows an operation targets.

    Exactly one of two modes: a non-empty `ids` list, or `by_filter=True` with
    optional filter fields. `by_filter` is required for the filter path, so an
    empty body is a 422 rather than a match of every row.
    """

    ids: list[str] | None = Field(default=None, max_length=_MAX_IDS)
    by_filter: bool = False
    user_id: str | _CappedValues | None = None
    api_key_id: str | _CappedValues | None = None
    # Event type / tool name (AgentTelemetry.name), e.g. "tool_result".
    name: str | None = None
    start_date: datetime | None = None
    end_date: datetime | None = None

    @model_validator(mode="after")
    def _require_exactly_one_mode(self) -> "AgentTelemetrySelection":
        has_ids = bool(self.ids)
        if has_ids == self.by_filter:
            raise ValueError("provide a non-empty `ids` list or set `by_filter` true (exactly one)")
        return self


class AgentTelemetryDeleteRequest(AgentTelemetrySelection):
    """Selection of agent_telemetry rows to delete."""


class AgentTelemetryDeleteResult(BaseModel):
    """How many rows the delete removed."""

    deleted: int = 0


def _selection_conditions(selection: AgentTelemetrySelection) -> list[ColumnElement[bool]]:
    if selection.ids:
        return [AgentTelemetry.id.in_(selection.ids)]
    conditions: list[ColumnElement[bool]] = []
    if selection.user_id is not None and selection.user_id != []:
        conditions.append(match_any(AgentTelemetry.user_id, selection.user_id))
    if selection.api_key_id is not None and selection.api_key_id != []:
        conditions.append(match_any(AgentTelemetry.api_key_id, selection.api_key_id))
    if selection.name is not None:
        conditions.append(AgentTelemetry.name == selection.name)
    if selection.start_date is not None:
        conditions.append(AgentTelemetry.timestamp >= selection.start_date)
    if selection.end_date is not None:
        conditions.append(AgentTelemetry.timestamp < selection.end_date)
    return conditions


async def delete_agent_telemetry(db: AsyncSession, request: AgentTelemetryDeleteRequest) -> AgentTelemetryDeleteResult:
    """Delete the agent_telemetry rows a selection matches.

    A selection matching zero rows succeeds with `deleted: 0`, not an error.
    No anti-replay guarantee: a purged event re-exported later is treated as
    new by `POST /v1/logs` and may be re-stored.
    """
    conditions = _selection_conditions(request)
    try:
        result = cast("CursorResult[Any]", await db.execute(delete(AgentTelemetry).where(*conditions)))
        await db.commit()
    except SQLAlchemyError:
        await db.rollback()
        logger.exception("agent_telemetry delete failed")
        raise
    deleted = result.rowcount or 0
    logger.info("agent_telemetry delete: removed=%d by_filter=%s", deleted, request.by_filter)
    return AgentTelemetryDeleteResult(deleted=deleted)
