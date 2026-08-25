"""Operator purge of previously-captured telemetry rows.

Captured telemetry never counts toward budget or spend, so this needs none of
`usage_admin_service`'s imported-only safety scoping: a fresh, narrower
selection schema (mirroring its `ids`-or-`by_filter` shape) is simpler than
threading a generic filter through fields that would not apply here. The
filter set is deliberately scoped to fields a captured row actually has: no
`model`, `provider`, `status`, `source_label`, `priced`, or `tool`.

Which rows are removed is decided here; removing them is `TelemetryStoragePort`'s
job, so a deployment storing telemetry elsewhere purges from that store instead.
"""

from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, Field, model_validator

from gateway.core.sql import MAX_FILTER_VALUES
from gateway.ports.telemetry_storage_port import TelemetryFilter, TelemetryStoragePort

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


def _values(value: str | list[str] | None) -> tuple[str, ...]:
    """One filter field as a tuple, treating an explicit empty list as unset."""
    if value is None:
        return ()
    return (value,) if isinstance(value, str) else tuple(value)


def _selection_filter(selection: AgentTelemetrySelection) -> TelemetryFilter:
    """The filter half of a selection, ignored by storage when ids are given."""
    return TelemetryFilter(
        start=selection.start_date,
        end=selection.end_date,
        user_ids=_values(selection.user_id),
        api_key_ids=_values(selection.api_key_id),
        name=selection.name,
    )


async def delete_agent_telemetry(
    request: AgentTelemetryDeleteRequest,
    *,
    storage: TelemetryStoragePort,
) -> AgentTelemetryDeleteResult:
    """Delete the telemetry rows a selection matches.

    A selection matching zero rows succeeds with `deleted: 0`, not an error.
    No anti-replay guarantee: a purged event re-exported later is treated as
    new by `POST /v1/logs` and may be re-stored.
    """
    deleted = await storage.purge(ids=tuple(request.ids or ()), filters=_selection_filter(request))
    return AgentTelemetryDeleteResult(deleted=deleted)
