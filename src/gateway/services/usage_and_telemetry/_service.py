"""The usage domain's reads, each in a block of its own on the Unit of Work it was built with."""

import uuid
from collections.abc import Collection
from dataclasses import replace
from datetime import datetime, timedelta

from gateway.core.metered_pricing import quantize_cost
from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions.usage_and_telemetry_exceptions import UsageSeriesTooLongError
from gateway.repositories.usage_and_telemetry import ServedTraffic, UsageFilters, UsageLogRepository, UsageScope
from gateway.schemas.usage_and_telemetry import (
    Bucket,
    RequestSettlement,
    SeriesGroupBy,
    SummaryDimension,
    UsageEntry,
    UsageErrorCodeRow,
    UsageGroupedSeries,
    UsageGroupRow,
    UsageSummary,
    UsageTotals,
)
from gateway.services.api_keys import ApiKeyService
from gateway.services.organizations import UserAccess
from gateway.services.usage_and_telemetry._window import (
    MAX_SERIES_POINTS,
    SERIES_TOP_N,
    dense_series,
    error_class_for,
    resolve_window,
)

# How many rows each breakdown returns before the remainder is folded into a
# single synthesized "other" row (so the tables still reconcile with the totals).
_BREAKDOWN_TOP_N = 100

# Sessions (``source_label``) are an order of magnitude higher-cardinality than
# models or users: one agent workload can open hundreds of them in a month, and
# the interesting signal is a long-ish head ("which tasks burned the budget"),
# not just the top few. Give that dimension a deeper cap so the head is not
# swallowed by the "other" fold.
_SESSION_BREAKDOWN_TOP_N = 250

# Every breakdown a summary can compute, mapped to its top-N cap. A dimension name
# is the ``by_<name>`` response field it fills, so a caller reads the selector and
# the payload with one vocabulary.
_SUMMARY_DIMENSIONS: dict[str, int] = {
    "model": _BREAKDOWN_TOP_N,
    "user": _BREAKDOWN_TOP_N,
    "api_key": _BREAKDOWN_TOP_N,
    "source": _BREAKDOWN_TOP_N,
    "source_label": _SESSION_BREAKDOWN_TOP_N,
    "endpoint": _BREAKDOWN_TOP_N,
    "provider": _BREAKDOWN_TOP_N,
}

# The failure taxonomy (``errors_by_status_code``) is a GROUP BY pass like the
# breakdowns above, but it groups failures by status code rather than spend by a
# dimension, so it is selectable by name without living in _SUMMARY_DIMENSIONS.
# It is the one dimension whose response field is not ``by_<name>``.
_ERROR_TAXONOMY_DIMENSION = "status_code"
_TOOL_DIMENSION = "tool"
_ALL_SUMMARY_DIMENSIONS: set[str] = set(_SUMMARY_DIMENSIONS) | {_ERROR_TAXONOMY_DIMENSION, _TOOL_DIMENSION}

# A grouped series names its dimension by the column; a breakdown by the response field.
_SERIES_DIMENSIONS: dict[str, str] = {"model": "model", "user_id": "user", "api_key_id": "api_key", "source": "source"}


class UsageService:
    """Usage rows, read inside a scope the caller derived.

    A breakdown by user or by key is labeled with the user's alias or the key's
    name, read through the domains that own them.
    """

    def __init__(
        self, uow: UnitOfWork, usage_logs: UsageLogRepository, *, keys: ApiKeyService, users: UserAccess
    ) -> None:
        self._uow = uow
        self._usage_logs = usage_logs
        self._keys = keys
        self._users = users

    async def served_traffic(self, *, workspace_ids: Collection[uuid.UUID], since: datetime) -> list[ServedTraffic]:
        """The requests served here since ``since`` in these workspaces, by name sent, offering reached and key."""
        async with self._uow:
            return await self._usage_logs.served_traffic(workspace_ids=workspace_ids, since=since)

    async def page(self, filters: UsageFilters, scope: UsageScope, *, skip: int, limit: int) -> list[UsageEntry]:
        """The rows matching these filters inside this scope, most recent first, each labeled with its user and key.

        The labels are read for the whole page at once rather than per row, and
        rather than left to the client: naming a page of rows must not cost a round
        trip each, nor oblige a dashboard to hold every user and every key in memory
        to label 100 rows. A row whose owner was deleted still comes back, with a
        null label.
        """
        async with self._uow:
            rows = await self._usage_logs.page(filters, scope, skip=skip, limit=limit)
            aliases = await self._users.aliases({row.user_id for row in rows if row.user_id is not None})
            names = await self._keys.key_names(
                {row.api_key_id for row in rows if row.api_key_id is not None}, workspace_ids=None
            )
        return [
            UsageEntry.from_model(
                row,
                user_alias=aliases.get(row.user_id) if row.user_id is not None else None,
                api_key_name=names.get(row.api_key_id) if row.api_key_id is not None else None,
            )
            for row in rows
        ]

    async def count(self, filters: UsageFilters, scope: UsageScope, *, imported_only: bool = False) -> int:
        """How many rows match these filters inside this scope; with ``imported_only``, only imported rows."""
        async with self._uow:
            return await self._usage_logs.count_matching(filters, scope, imported_only=imported_only)

    async def request_settlement(self, request_id: str, *, api_key_id: str | None) -> RequestSettlement | None:
        """What one request settled at, or ``None`` until the row that settles it is written.

        With ``api_key_id`` only that key's rows count, so another key's request
        reads the same as one that does not exist. Absorbed rows and a vision
        side-call alone mean the row that settles the request has not been written
        yet.
        """
        async with self._uow:
            rows = await self._usage_logs.request_rows(request_id, api_key_id=api_key_id)
        if rows.served == 0 and rows.errors == 0 and rows.incomplete == 0:
            return None
        cost_usd = f"{quantize_cost(rows.cost):.6f}" if rows.priced_rows and rows.cost is not None else None
        # Only where every priced row recorded its tool part: a partial sum would understate it.
        tool_cost_known = cost_usd is not None and rows.tool_costed_rows == rows.priced_rows
        return RequestSettlement(
            request_id=request_id,
            status="error" if rows.errors else "incomplete" if rows.incomplete else "success",
            cost_usd=cost_usd,
            tool_cost_usd=(
                f"{quantize_cost(rows.tool_cost):.6f}" if tool_cost_known and rows.tool_cost is not None else None
            ),
            prompt_tokens=rows.prompt_tokens,
            completion_tokens=rows.completion_tokens,
            total_tokens=rows.total_tokens,
            row_count=rows.row_count,
        )

    async def cost_and_requests(self, filters: UsageFilters, scope: UsageScope) -> tuple[float, int]:
        """The spend and the request count over the matching rows, counting requests as every summary does."""
        async with self._uow:
            return await self._usage_logs.cost_and_requests(filters, scope)

    async def summary(
        self,
        filters: UsageFilters,
        scope: UsageScope,
        *,
        bucket: Bucket,
        dimensions: list[SummaryDimension] | None,
        group_by_tag: str | None = None,
    ) -> UsageSummary:
        """Totals, the requested breakdowns and a zero-filled series over a bounded window.

        Each breakdown is its own ``GROUP BY`` pass, so ``dimensions`` names the
        ones to compute; ``None`` computes them all.
        """
        start, end = resolve_window(filters.start_date, filters.end_date)
        filters = replace(filters, start_date=start, end_date=end)
        # ``none`` is dropped rather than rejected: it exists only so a caller can send
        # an empty selection, and it never contributes a dimension of its own.
        requested = _ALL_SUMMARY_DIMENSIONS if dimensions is None else {d for d in dimensions if d != "none"}
        async with self._uow:
            totals = await self._usage_logs.totals(filters, scope)
            breakdowns = {
                name: await self._breakdown(filters, scope, totals, dimension=name, limit=cap)
                for name, cap in _SUMMARY_DIMENSIONS.items()
                if name in requested
            }
            # The failure taxonomy is a GROUP BY pass like the others, so it answers to the
            # same selector rather than being charged to every caller: the tiles and the
            # timeline ask for no dimensions at all. No window reaches the cap (HTTP has
            # far fewer codes), so unlike the cost breakdowns there is no fold row.
            errors_by_status_code = (
                [
                    UsageErrorCodeRow(status_code=code, error_class=error_class_for(code), requests=requests)
                    for code, requests in await self._usage_logs.errors_by_status_code(
                        filters, scope, limit=_BREAKDOWN_TOP_N
                    )
                ]
                if _ERROR_TAXONOMY_DIMENSION in requested
                else []
            )
            by_tool = await self._usage_logs.tool_usage(filters, scope) if _TOOL_DIMENSION in requested else []
            by_tag = (
                _with_fold(
                    await self._usage_logs.groups(filters, scope, tag_key=group_by_tag, limit=_BREAKDOWN_TOP_N),
                    totals,
                )
                if group_by_tag
                else []
            )
            points = await self._usage_logs.series(filters, scope, bucket=bucket)
        # Zero-fill empty buckets so the chart is time-linear (GROUP BY drops gaps).
        series = dense_series(start, end, bucket, {point.bucket_start: point for point in points})
        return UsageSummary(
            start_date=start.isoformat(),
            end_date=end.isoformat(),
            bucket=bucket,
            totals=totals,
            by_model=breakdowns.get("model", []),
            by_user=breakdowns.get("user", []),
            by_api_key=breakdowns.get("api_key", []),
            by_source=breakdowns.get("source", []),
            by_source_label=breakdowns.get("source_label", []),
            by_endpoint=breakdowns.get("endpoint", []),
            by_provider=breakdowns.get("provider", []),
            by_tool=by_tool,
            by_tag=by_tag,
            errors_by_status_code=errors_by_status_code,
            series=series,
        )

    async def grouped_series(
        self, filters: UsageFilters, scope: UsageScope, *, bucket: Bucket, group_by: SeriesGroupBy
    ) -> UsageGroupedSeries:
        """A series split by one dimension: the window's top groups by spend, the rest folded into one.

        The points are sparse, so it is the bucket grid that is bounded: an hourly
        bucket over the widest window would otherwise be ~8.8k buckets for each group.
        """
        start, end = resolve_window(filters.start_date, filters.end_date)
        filters = replace(filters, start_date=start, end_date=end)
        step = timedelta(hours=1) if bucket == "hour" else timedelta(days=1)
        if (end - start) / step > MAX_SERIES_POINTS:
            raise UsageSeriesTooLongError(MAX_SERIES_POINTS, bucket)
        dimension = _SERIES_DIMENSIONS[group_by]
        async with self._uow:
            totals = await self._usage_logs.totals(filters, scope)
            groups = await self._breakdown(filters, scope, totals, dimension=dimension, limit=SERIES_TOP_N)
            points = await self._usage_logs.grouped_series(
                filters,
                scope,
                bucket=bucket,
                dimension=dimension,
                named={group.key for group in groups if group.key is not None},
                keeps_null=any(group.key is None and not group.is_other for group in groups),
            )
        return UsageGroupedSeries(
            start_date=start.isoformat(),
            end_date=end.isoformat(),
            bucket=bucket,
            group_by=group_by,
            groups=groups,
            points=sorted(points, key=lambda point: point.bucket_start),
        )

    async def _breakdown(
        self, filters: UsageFilters, scope: UsageScope, totals: UsageTotals, *, dimension: str, limit: int
    ) -> list[UsageGroupRow]:
        """One dimension's top groups, labeled where the key is an opaque id, with the remainder folded.

        Only the user and key dimensions need a label: a model, source or endpoint
        already reads as its own name. Resolving it here is what lets a client
        offer a user or key filter without holding those whole tables.
        """
        rows = await self._usage_logs.groups(filters, scope, dimension=dimension, limit=limit)
        keys = {row.key for row in rows if row.key is not None}
        if dimension == "user":
            labels = await self._users.aliases(keys)
        elif dimension == "api_key":
            labels = await self._keys.key_names(keys, workspace_ids=None)
        else:
            labels = {}
        if labels:
            rows = [
                row.model_copy(update={"label": labels.get(row.key)}) if row.key is not None else row for row in rows
            ]
        return _with_fold(rows, totals)


def _with_fold(rows: list[UsageGroupRow], totals: UsageTotals) -> list[UsageGroupRow]:
    """Append the groups past the cap as one synthesized ``other`` row, so a breakdown reconciles with the totals."""
    # request_count is an exact integer, so a positive residual is the reliable
    # signal that groups were folded; cost/tokens residuals follow from totals.
    residual_requests = totals.request_count - sum(row.requests for row in rows)
    if residual_requests <= 0:
        return rows
    billed_total = totals.billed_input_tokens + totals.billed_output_tokens
    return [
        *rows,
        UsageGroupRow(
            key=None,
            cost=totals.cost - sum(row.cost for row in rows),
            tokens=billed_total - sum(row.tokens for row in rows),
            requests=residual_requests,
            is_other=True,
        ),
    ]
