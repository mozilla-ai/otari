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

import uuid
from datetime import datetime
from typing import Annotated, Any, cast

from pydantic import BaseModel, Field, model_validator
from sqlalchemy import ColumnElement, delete, func, select
from sqlalchemy.engine import CursorResult
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.metered_pricing import BillableUsage, billable_usage, price_billable_usage
from gateway.core.sql import MAX_FILTER_VALUES, match_any, utc_bound
from gateway.log_config import logger
from gateway.models.entities import ModelPricing, UsageLog
from gateway.services.tool_usage import TOOL_METER_NAMESPACE

# Cap on an explicit id list. Page selections drive the id path and the largest
# rows-per-page the UI offers is 500; 1000 leaves headroom without letting a single
# request name an unbounded set of ids.
_MAX_IDS = 1000
# Load matched rows for repricing in chunks under SQLite's default bound on bind
# variables in one IN() (999), mirroring the ingest path.
_REPRICE_CHUNK = 500

# A repeatable entity filter's values, bounded the way the read endpoints bound
# theirs (see MAX_FILTER_VALUES). The bound is annotated on the list itself rather
# than on the ``str | list[str]`` field: on the union it would also cap a single
# value's character length, rejecting a long provider-qualified model name.
_CappedValues = Annotated[list[str], Field(max_length=MAX_FILTER_VALUES)]


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
    # The three entity filters accept several values, matching the repeatable form
    # the read endpoints take. They have to: "all N matching" is counted from the
    # filters the operator was shown and re-derived here, so a filter this body
    # could not express would target more rows than the table displayed. They carry
    # the read endpoints' ceiling for the mirror of that reason: a value set /count
    # rejects (422) but a delete accepted would run destructively over rows no count
    # could have been shown for.
    model: str | _CappedValues | None = None
    user_id: str | _CappedValues | None = None
    api_key_id: str | _CappedValues | None = None
    status: str | None = None
    endpoint: str | None = None
    provider: str | None = None
    # Session/project attribution, so a bulk op driven from a session drill-down
    # targets that session rather than every imported row in the window.
    source_label: str | None = None
    start_date: datetime | None = None
    end_date: datetime | None = None
    # None: any; True: only rows with a cost; False: only rows with no cost yet.
    priced: bool | None = None
    # Gateway-run tool usage, forwarded so a bulk op driven from a tool-filtered
    # Activity view targets exactly the rows the operator was shown. In practice it
    # matches nothing here, because only gateway rows carry tool meters and this
    # selection is hard-scoped to imported rows; it is still forwarded so the
    # "every scoping filter must be repeatable" invariant holds without exception.
    tool: str | None = None
    # Workspace scoping. On this body for the same reason every other scoping
    # filter is: an operator who filtered the table to one workspace and then
    # chose "all N matching" must not delete another workspace's rows.
    workspace_id: uuid.UUID | None = None

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
    if selection.workspace_id is not None:
        conditions.append(UsageLog.workspace_id == selection.workspace_id)
    if selection.source is not None:
        conditions.append(UsageLog.source == selection.source)
    # An empty list is no filter at all, the same reading the count endpoint applies.
    # That agreement is the point: the "N matching" an operator confirms comes from
    # /v1/usage/count over this same filter set, so a dimension the two endpoints
    # scoped differently would delete a different number of rows than the dialog
    # promised. (The dashboard sends the field absent, never empty.)
    if selection.model is not None and selection.model != []:
        conditions.append(match_any(UsageLog.model, selection.model))
    if selection.user_id is not None and selection.user_id != []:
        conditions.append(match_any(UsageLog.user_id, selection.user_id))
    if selection.api_key_id is not None and selection.api_key_id != []:
        conditions.append(match_any(UsageLog.api_key_id, selection.api_key_id))
    if selection.status is not None:
        conditions.append(UsageLog.status == selection.status)
    if selection.endpoint is not None:
        conditions.append(UsageLog.endpoint == selection.endpoint)
    if selection.provider is not None:
        conditions.append(UsageLog.provider == selection.provider)
    if selection.source_label is not None:
        conditions.append(UsageLog.source_label == selection.source_label)
    if selection.start_date is not None:
        conditions.append(UsageLog.timestamp >= utc_bound(selection.start_date))
    if selection.end_date is not None:
        conditions.append(UsageLog.timestamp < utc_bound(selection.end_date))
    if selection.priced is True:
        conditions.append(UsageLog.cost.is_not(None))
    elif selection.priced is False:
        conditions.append(UsageLog.cost.is_(None))
    if selection.tool is not None:
        namespace = UsageLog.billing_meters[TOOL_METER_NAMESPACE]
        conditions.append(
            # See routes/usage._tool_used_expr: the text coercion is what makes a
            # missing key compare as SQL NULL rather than JSON null.
            namespace.as_string().is_not(None)
            if selection.tool == "any"
            else namespace[selection.tool]["billed"].as_integer().is_not(None)
        )
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


def _row_cache_tokens_included(row: UsageLog) -> bool:
    """Which cached-token convention a stored row's token counts were reported under.

    ``cache_tokens_in_prompt`` answers it directly when the row has it: ingest
    writes the convention the submitter stated and settlement writes the one the
    provider reported, so a row repriced later is priced the way it arrived.
    That covers the case the recovery below cannot, a row that was never priced
    at all (no rate row for its model at ingest) and so has no meters to read.

    The column is nullable, so a row written before it existed reads NULL, and
    "not recorded" is answered the way it always was: recovered from the meters
    the pricing wrote, where ``total_input_tokens`` equals ``prompt_tokens``
    under the inclusive shape and exceeds it by the cache buckets under the
    additive one. That is a fact recorded at settlement, not a guess.

    A row with neither (never priced and written before the column) falls back
    to the additive shape, which is the ingest default and the convention every
    Claude Code import carries. A row with no cache tokens prices identically
    under either shape, so the fallback only decides rows that were both
    unpriced and cached.
    """
    if row.cache_tokens_in_prompt is not None:
        return row.cache_tokens_in_prompt
    meters = row.billing_meters or {}
    total_input = meters.get("total_input_tokens")
    if not isinstance(total_input, int):
        return False
    return total_input == (row.prompt_tokens or 0)


def _row_usage(row: UsageLog) -> BillableUsage:
    """Rebuild billing meters from a stored row's token counts."""
    return billable_usage(
        input_tokens=row.prompt_tokens or 0,
        output_tokens=row.completion_tokens or 0,
        cache_read_tokens=row.cache_read_tokens or 0,
        cache_write_tokens=row.cache_write_tokens or 0,
        cache_write_1h_tokens=row.cache_write_1h_tokens or 0,
        cache_tokens_included=_row_cache_tokens_included(row),
    )


async def set_usage_price(db: AsyncSession, request: UsageSetPriceRequest) -> UsageSetPriceResult:
    """Recompute cost for matched imported rows from manual per-1M rates.

    Builds a transient ``ModelPricing`` from the supplied rates and reprices each
    matched row against its own token counts, writing ``cost`` / ``billing_meters`` /
    ``pricing_breakdown`` back and clearing the row's pricing provenance, which the
    old amount's source no longer explains. Only imported rows are touched (see
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
                cost, meters, breakdown = price_billable_usage(pricing, _row_usage(row))
                if cost == row.cost:
                    result.unchanged += 1
                    continue
                row.cost = cost
                row.billing_meters = meters
                row.pricing_breakdown = breakdown
                # The amount no longer comes from whatever priced it before, so the
                # provenance recorded against it would now be a lie. Cleared rather
                # than rewritten: a manual per-1M rate an operator typed is not an
                # entry in any price list, so there is no source to name and NULL is
                # the honest answer (see ``UsageLog.pricing_source``).
                row.pricing_source = None
                row.pricing_reference = None
                row.pricing_effective_at = None
                row.pricing_version = None
                row.calculated_at = None
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
