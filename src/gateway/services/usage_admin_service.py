"""Operator mutations over imported usage rows: bulk delete and manual repricing.

The dashboard's usage tables let an operator clean up or correct *imported* usage
(rows brought in through ``POST /v1/usage/external-events``), the routine work that
otherwise needs direct DB access. Two operations are exposed:

- **delete**: drop a set of imported rows.
- **set-price**: recompute ``cost`` / ``billing_meters`` / ``pricing_breakdown`` from
  manual per-1M rates and each row's own token counts.

Both target the same set two ways: an explicit ``ids`` list (the current UI selection)
or a filter (for "everything matching this filter"). ``by_filter`` must be set
explicitly for the filter path so an empty request body can never match, and thus never
delete or reprice, every imported row by accident.

**Safety invariant:** every query is scoped to ``counts_toward_budget = False`` (imported
rows). Enforced gateway rows and the spend ledger (``users.spend``) are never touched, so
neither operation can desync a budget, matching the boundary the ingest path establishes.
"""

from datetime import datetime
from typing import Any, cast

from pydantic import BaseModel, Field, model_validator
from sqlalchemy import ColumnElement, delete, func, select
from sqlalchemy.engine import CursorResult
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.usage import GatewayUsage
from gateway.log_config import logger
from gateway.models.entities import ModelPricing, UsageLog
from gateway.services.metered_pricing import calculate_metered_cost

# Cap on an explicit id list. Page selections drive the id path and the largest
# rows-per-page the UI offers is 500; 1000 leaves headroom without letting a single
# request name an unbounded set of ids.
_MAX_IDS = 1000
# Load matched rows for repricing in chunks under SQLite's default bound on bind
# variables in one IN() (999), mirroring the ingest path.
_REPRICE_CHUNK = 500


class UsageSelection(BaseModel):
    """Which imported usage rows an operation targets.

    Exactly one of two modes: a non-empty ``ids`` list (the current UI selection) or
    ``by_filter=True`` with optional filter fields (everything matching). ``by_filter``
    is required for the filter path, so an empty body is a 422 rather than a match of
    every imported row.
    """

    ids: list[str] | None = Field(default=None, max_length=_MAX_IDS)
    by_filter: bool = False
    source: str | None = None
    model: str | None = None
    user_id: str | None = None
    api_key_id: str | None = None
    status: str | None = None
    endpoint: str | None = None
    start_date: datetime | None = None
    end_date: datetime | None = None
    # None: any; True: only rows with a cost; False: only rows with no cost yet.
    priced: bool | None = None

    @model_validator(mode="after")
    def _require_exactly_one_mode(self) -> "UsageSelection":
        has_ids = bool(self.ids)
        if has_ids == self.by_filter:
            raise ValueError("provide a non-empty `ids` list or set `by_filter` true (exactly one)")
        return self


class UsageDeleteRequest(UsageSelection):
    """Selection of imported usage rows to delete."""


class UsageDeleteResult(BaseModel):
    """How many imported rows the delete removed."""

    deleted: int = 0


class UsageSetPriceRequest(UsageSelection):
    """Selection of imported usage rows plus the manual per-1M rates to price them at.

    ``input`` and ``output`` are required (every row is charged for them); the cache
    rates are optional and, when omitted, those tokens fold into the fresh-input charge
    exactly as an unpriced cache rate does in normal metered pricing.
    """

    input_price_per_million: float = Field(ge=0)
    output_price_per_million: float = Field(ge=0)
    cache_read_price_per_million: float | None = Field(default=None, ge=0)
    cache_write_price_per_million: float | None = Field(default=None, ge=0)


class UsageSetPriceResult(BaseModel):
    """Outcome of a manual repricing pass.

    ``matched`` imported rows were recomputed; ``updated`` had their stored cost
    changed and ``unchanged`` already matched the recomputed value.
    """

    matched: int = 0
    updated: int = 0
    unchanged: int = 0


def _selection_conditions(selection: UsageSelection) -> list[ColumnElement[bool]]:
    """WHERE conditions for a selection, always scoped to imported rows.

    Two fixed conditions pin the target set to imported usage regardless of the
    caller's input:

    - ``source != "gateway"`` is the provenance invariant: imported rows carry a
      source slug (e.g. ``claude_code``), and "gateway" is reserved for usage Otari
      served itself. This is the load-bearing guard, because ``counts_toward_budget``
      alone is *not* an imported-only flag: gateway traffic on a budget-exempt API
      key (``exclude_from_budget``) is also ``counts_toward_budget = False``, and
      those are real gateway rows a cleanup / reprice must never touch.
    - ``counts_toward_budget = False`` is kept as a defense-in-depth budget guard, so
      the spend ledger can never be affected even if the provenance guard ever slips.

    Together they mean even an ``ids`` list naming enforced or budget-exempt gateway
    rows cannot reach them: they simply do not match.
    """
    conditions: list[ColumnElement[bool]] = [
        UsageLog.source != "gateway",
        UsageLog.counts_toward_budget.is_(False),
    ]
    if selection.ids:
        conditions.append(UsageLog.id.in_(selection.ids))
        return conditions
    if selection.source is not None:
        conditions.append(UsageLog.source == selection.source)
    if selection.model is not None:
        conditions.append(UsageLog.model == selection.model)
    if selection.user_id is not None:
        conditions.append(UsageLog.user_id == selection.user_id)
    if selection.api_key_id is not None:
        conditions.append(UsageLog.api_key_id == selection.api_key_id)
    if selection.status is not None:
        conditions.append(UsageLog.status == selection.status)
    if selection.endpoint is not None:
        conditions.append(UsageLog.endpoint == selection.endpoint)
    if selection.start_date is not None:
        conditions.append(UsageLog.timestamp >= selection.start_date)
    if selection.end_date is not None:
        conditions.append(UsageLog.timestamp < selection.end_date)
    if selection.priced is True:
        conditions.append(UsageLog.cost.is_not(None))
    elif selection.priced is False:
        conditions.append(UsageLog.cost.is_(None))
    return conditions


async def delete_usage(db: AsyncSession, request: UsageDeleteRequest) -> UsageDeleteResult:
    """Delete the imported usage rows a selection matches.

    Only ``counts_toward_budget=False`` rows are ever removed, so the enforcement
    ledger is untouched. Nothing references ``usage_logs``, so a plain bulk delete
    leaves no orphans.
    """
    conditions = _selection_conditions(request)
    try:
        result = cast("CursorResult[Any]", await db.execute(delete(UsageLog).where(*conditions)))
        await db.commit()
    except SQLAlchemyError:
        await db.rollback()
        logger.exception("usage delete failed")
        raise
    deleted = result.rowcount or 0
    logger.info("usage delete: removed=%d by_filter=%s", deleted, request.by_filter)
    return UsageDeleteResult(deleted=deleted)


def _row_usage(row: UsageLog) -> GatewayUsage:
    """Rebuild billing usage from a stored row's token counts.

    The per-event cache convention (``cache_tokens_in_prompt``) is not persisted, so
    this assumes the additive shape (cache buckets sit outside ``prompt_tokens``), the
    ingest default and the Anthropic / Claude Code convention imported usage carries.
    Rows with no cache tokens price identically under either shape.
    """
    return GatewayUsage(
        prompt_tokens=row.prompt_tokens or 0,
        completion_tokens=row.completion_tokens or 0,
        total_tokens=row.total_tokens or 0,
        cache_read_tokens=row.cache_read_tokens or 0,
        cache_write_tokens=row.cache_write_tokens or 0,
        cache_write_1h_tokens=row.cache_write_1h_tokens or 0,
        cache_tokens_in_prompt=False,
    )


async def set_usage_price(db: AsyncSession, request: UsageSetPriceRequest) -> UsageSetPriceResult:
    """Recompute cost for matched imported rows from manual per-1M rates.

    Builds a transient ``ModelPricing`` from the supplied rates and reprices each
    matched row against its own token counts, writing ``cost`` / ``billing_meters`` /
    ``pricing_breakdown`` back. Only imported rows are touched (see
    ``_selection_conditions``), so recomputing cost can never desync ``users.spend``.
    Rows whose recomputed cost equals the stored value are reported ``unchanged``.

    Rows are walked in bounded keyset pages ordered by id, so memory stays flat even
    for a large ``by_filter`` set (each page is detached after it is flushed). The
    whole pass commits once at the end: like ``delete_usage`` it is all-or-nothing, so
    an error partway through leaves no half-repriced set behind. Keyset paging is safe
    while mutating because it only moves forward past ``last_id``; already-repriced
    rows sit behind the cursor and are never revisited, even when a ``priced=False``
    filter would no longer match them.
    """
    pricing = ModelPricing(
        model_key="__manual__",
        input_price_per_million=request.input_price_per_million,
        output_price_per_million=request.output_price_per_million,
        cache_read_price_per_million=request.cache_read_price_per_million,
        cache_write_price_per_million=request.cache_write_price_per_million,
        cache_write_1h_price_per_million=None,
        pricing_tiers=[],
    )
    conditions = _selection_conditions(request)
    result = UsageSetPriceResult()

    try:
        last_id = ""
        while True:
            rows = (
                await db.execute(
                    select(UsageLog)
                    .where(*conditions, UsageLog.id > last_id)
                    .order_by(UsageLog.id)
                    .limit(_REPRICE_CHUNK)
                )
            ).scalars().all()
            if not rows:
                break
            for row in rows:
                result.matched += 1
                cost, meters, breakdown = calculate_metered_cost(pricing, _row_usage(row))
                if cost == row.cost:
                    result.unchanged += 1
                    continue
                row.cost = cost
                row.billing_meters = meters
                row.pricing_breakdown = breakdown
                result.updated += 1
            last_id = rows[-1].id
            # Flush this page's UPDATEs into the (still open) transaction, then detach
            # the objects so the session does not accumulate the whole matched set.
            await db.flush()
            db.expunge_all()
        await db.commit()
    except SQLAlchemyError:
        await db.rollback()
        logger.exception("usage set-price failed")
        raise

    logger.info(
        "usage set-price: matched=%d updated=%d unchanged=%d by_filter=%s",
        result.matched,
        result.updated,
        result.unchanged,
        request.by_filter,
    )
    return result


# Count query used by the dashboard's "select all N matching this filter" affordance
# and the delete/set-price confirm dialogs, so an operator sees how many imported rows
# a filter touches before committing to the mutation.
async def count_imported_matches(db: AsyncSession, selection: UsageSelection) -> int:
    conditions = _selection_conditions(selection)
    stmt = select(func.count()).select_from(UsageLog).where(*conditions)
    return int((await db.execute(stmt)).scalar_one())
