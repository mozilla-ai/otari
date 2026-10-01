"""What a learning router backend is told about a request it decided, built from the request's usage row."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any

from gateway.models.usage import UsageLog
from gateway.ports.routing_port import RoutingOutcome

__all__ = ["outcome_from_usage_row"]

# Charge lines that price the prompt side of a row. Everything else the token
# pricing writes is ``output``; a gateway-run tool's line is neither, and counts
# only toward the row's total.
_PROMPT_METERS = frozenset({"input", "cache_read", "cache_write_5m", "cache_write_1h"})
_COMPLETION_METERS = frozenset({"output"})


def _meter_cost(row: UsageLog, meters: frozenset[str]) -> Decimal | None:
    """The summed cost of ``row``'s charge lines for ``meters``, or ``None`` when the row was not priced by token."""
    lines = [line for line in row.pricing_breakdown or [] if isinstance(line, dict)]
    if not any(line.get("meter") in _PROMPT_METERS | _COMPLETION_METERS for line in lines):
        return None
    return sum(
        (Decimal(str(line.get("cost") or 0)) for line in lines if line.get("meter") in meters),
        start=Decimal(0),
    )


def _error(row: UsageLog) -> dict[str, Any] | None:
    """The failure as the row recorded it: the redacted message and the classifying status."""
    if row.status == "success":
        return None
    return {"message": row.error_message, "status_code": row.status_code}


def outcome_from_usage_row(row: UsageLog, *, model: str, started_at: datetime) -> RoutingOutcome:
    """The outcome of the request ``row`` settled, for the backend that routed it.

    ``model`` is the served candidate as the policy spelled it, and ``started_at``
    the wall-clock time the gateway began handling the request; the row carries
    neither. The row's own ``timestamp`` is when the request completed.
    """
    return RoutingOutcome(
        model=model,
        success=row.status == "success",
        error=_error(row),
        started_at=started_at,
        completed_at=row.timestamp,
        prompt_tokens=row.prompt_tokens,
        completion_tokens=row.completion_tokens,
        total_tokens=row.total_tokens,
        prompt_cost_usd=_meter_cost(row, _PROMPT_METERS),
        completion_cost_usd=_meter_cost(row, _COMPLETION_METERS),
        total_cost_usd=row.cost,
    )
