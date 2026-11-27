"""The record of each subagent model recommendation the gateway answered.

A row is what a later reading of the recommendation's worth starts from: the
model the subagent would have started on, the model recommended, and what the
decision call cost. When the harness reports what the subagent consumed, the
same tokens are priced twice, at the models it ran on and at the baseline, and
both amounts are stored.
"""

import uuid
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Literal

from gateway.core.database import DATABASE_ERRORS, describe_database_error
from gateway.core.metered_pricing import RATE_FIELDS, calculate_token_cost, priced_per_request, quantize_cost
from gateway.core.sql import WorkspaceIds
from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions.routing_exceptions import RecommenderNotConfiguredError, SubagentDecisionNotFoundError
from gateway.log_config import logger
from gateway.models.routing import (
    BASELINE_PARENT,
    BASELINE_REQUESTED,
    SUBAGENT_OUTCOME_ESTIMATED,
    SUBAGENT_OUTCOME_MEASURED,
    UNPRICED_ACTUAL,
    UNPRICED_BASELINE,
    UNPRICED_BASELINE_UNRESOLVED,
    UNPRICED_PER_REQUEST,
    SubagentDecision,
)
from gateway.models.usage import DECISIONS_ENDPOINT
from gateway.ports.agent_model_recommender_port import AgentModelRecommenderPort, SubagentSpawn
from gateway.repositories.routing import DecisionSums, DecisionWindow, SubagentDecisionRepository
from gateway.schemas.routing import (
    SavingsAmounts,
    SavingsByAgentType,
    SavingsBySwap,
    SubagentSavingsPoint,
    SubagentSavingsPublic,
    SubagentSavingsTotals,
    SubagentUsageLine,
)
from gateway.services.routing._subagent_baseline import baseline_key, model_key
from gateway.services.routing.recommend import ROUTER_LABEL_MODEL, ROUTER_LABEL_PROVIDER
from gateway.services.usage_and_telemetry import UsageFilters, UsageScope, UsageService, dense_series, resolve_window

SAVINGS_TOP_N = 20
_NOTHING = DecisionSums((), 0, 0, 0, 0, Decimal(0), Decimal(0), Decimal(0))

SubagentRateLookup = Callable[[str, uuid.UUID], Awaitable[Any]]
"""The rate a ``<provider>:<model>`` is priced at for a workspace's organization, or None."""

# Claude Code's spelling of "the parent's model" in an agent definition.
_INHERIT = "inherit"


def _baseline(spawn: SubagentSpawn) -> tuple[str, str]:
    """The model the subagent would have started on with no recommendation, and where that came from."""
    requested = spawn.requested_model
    if requested and requested != _INHERIT:
        return requested, BASELINE_REQUESTED
    return spawn.parent_model, BASELINE_PARENT


@dataclass(frozen=True)
class _BaseRate:
    """A rate's base prices, without its context tiers.

    A report sums a whole run, whose total would reach a long-context tier no
    single request in it reached, so both sides are priced at the base rates.
    """

    model_key: str
    input_price_per_million: Any
    output_price_per_million: Any
    cache_read_price_per_million: Any
    cache_write_price_per_million: Any


@dataclass(frozen=True)
class _Priced:
    baseline_model_key: str | None = None
    baseline_cost: Decimal | None = None
    actual_cost: Decimal | None = None
    unpriced_reason: str | None = None


def _base_rate(key: str, rate: Any) -> _BaseRate:
    return _BaseRate(key, *(getattr(rate, field, None) for field in RATE_FIELDS))


def _cost(rate: _BaseRate, lines: Sequence[SubagentUsageLine]) -> Decimal:
    return sum(
        (
            calculate_token_cost(
                rate,
                input_tokens=line.input_tokens,
                output_tokens=line.output_tokens,
                cache_tokens_included=False,
                cache_read_tokens=line.cache_read_tokens,
                cache_write_tokens=line.cache_write_tokens,
                quantize=False,
            )
            for line in lines
        ),
        Decimal(0),
    )


def _has_tokens(line: SubagentUsageLine) -> bool:
    return bool(line.input_tokens or line.output_tokens or line.cache_read_tokens or line.cache_write_tokens)


def _amounts(sums: DecisionSums) -> dict[str, Any]:
    gross = sums.baseline_cost - sums.actual_cost
    return {
        "decisions": sums.decisions,
        "with_outcome": sums.with_outcome,
        "measured": sums.measured,
        "estimated": sums.estimated,
        "baseline_cost": float(sums.baseline_cost),
        "actual_cost": float(sums.actual_cost),
        "router_cost": float(sums.router_cost),
        "gross_saved": float(gross),
        # Every decision's router cost comes off, also where no saving could be read.
        "net_saved": float(gross - sums.router_cost),
    }


class SubagentDecisionService:
    """Keep what each answered recommendation decided, and what came of it."""

    def __init__(
        self,
        uow: UnitOfWork,
        decisions: SubagentDecisionRepository,
        rate: SubagentRateLookup,
        *,
        usage: UsageService,
        recommender: AgentModelRecommenderPort,
    ) -> None:
        self._uow = uow
        self._rows = decisions
        self._rate = rate
        self._usage = usage
        self._recommender = recommender

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

    async def report_outcome(
        self,
        decision_id: uuid.UUID,
        *,
        user_id: str | None,
        workspace_id: uuid.UUID | None,
        api_key_id: str | None = None,
        lines: Sequence[SubagentUsageLine],
    ) -> None:
        """Store what the subagent a decision started consumed, as its harness counted it, and price it.

        ``lines`` are running totals and replace an earlier report. A caller
        named by ``user_id`` and ``workspace_id`` reaches only its own
        decisions, and one with no user is named by ``api_key_id`` too, so it
        reaches only the decisions its own key made. Metered usage is never replaced by a harness's count.

        Raises:
            SubagentDecisionNotFoundError: no decision of the caller's has this id.

        """
        async with self._uow:
            row = await self._rows.get_for_caller(
                decision_id, user_id=user_id, workspace_id=workspace_id, api_key_id=api_key_id
            )
            if row is None:
                raise SubagentDecisionNotFoundError
            if row.outcome_kind == SUBAGENT_OUTCOME_MEASURED:
                return
            priced = await self._price(row, lines)
            row.outcome_kind = SUBAGENT_OUTCOME_ESTIMATED
            row.outcome_at = datetime.now(UTC)
            row.ran_on = max(lines, key=lambda line: (_has_tokens(line), line.output_tokens)).model
            row.usage = [line.model_dump() for line in lines]
            row.input_tokens = sum(line.input_tokens for line in lines)
            row.output_tokens = sum(line.output_tokens for line in lines)
            row.cache_read_tokens = sum(line.cache_read_tokens for line in lines)
            row.cache_write_tokens = sum(line.cache_write_tokens for line in lines)
            row.baseline_model_key = priced.baseline_model_key
            row.baseline_cost = priced.baseline_cost
            row.actual_cost = priced.actual_cost
            row.unpriced_reason = priced.unpriced_reason
            await self._rows.save(row)

    async def _price(self, row: SubagentDecision, lines: Sequence[SubagentUsageLine]) -> _Priced:
        """Price ``lines`` at the models they name and at the decision's baseline, or say why not.

        A line with no tokens is a placeholder a harness lists and is left out.
        """
        lines = [line for line in lines if _has_tokens(line)]
        key = baseline_key(
            row.harness, row.baseline_model, parent_model=row.parent_model, ran_on=[line.model for line in lines]
        )
        if key is None:
            return _Priced(unpriced_reason=UNPRICED_BASELINE_UNRESOLVED)
        baseline = await self._rate(key, row.workspace_id)
        if baseline is None:
            return _Priced(baseline_model_key=key, unpriced_reason=UNPRICED_BASELINE)
        if priced_per_request(baseline):
            return _Priced(baseline_model_key=key, unpriced_reason=UNPRICED_PER_REQUEST)
        actual = Decimal(0)
        for line in lines:
            line_key = model_key(row.harness, line.model)
            rate = await self._rate(line_key, row.workspace_id) if line_key is not None else None
            if line_key is None or rate is None:
                return _Priced(baseline_model_key=key, unpriced_reason=UNPRICED_ACTUAL)
            if priced_per_request(rate):
                return _Priced(baseline_model_key=key, unpriced_reason=UNPRICED_PER_REQUEST)
            actual += _cost(_base_rate(line_key, rate), [line])
        return _Priced(
            baseline_model_key=key,
            baseline_cost=quantize_cost(_cost(_base_rate(key, baseline), lines)),
            actual_cost=quantize_cost(actual),
        )

    async def savings(
        self,
        *,
        workspaces: WorkspaceIds,
        start_date: datetime | None,
        end_date: datetime | None,
        user_ids: list[str] | None,
        bucket: Literal["hour", "day"],
    ) -> SubagentSavingsPublic:
        """What the recommendations made in the window saved, with no dates the last 30 days.

        In all, over time, by subagent type and by model swap. ``workspaces``
        is the caller's scope and ``user_ids`` narrows it.
        """
        start, end = resolve_window(start_date, end_date)
        try:
            self._recommender.quote()
            recommender_available = True
        except RecommenderNotConfiguredError:
            recommender_available = False
        router_unanswered = await self._usage.count(
            UsageFilters(
                start_date=start,
                end_date=end,
                user_id=user_ids,
                status="error",
                model=[ROUTER_LABEL_MODEL],
                provider=ROUTER_LABEL_PROVIDER,
                endpoint=DECISIONS_ENDPOINT,
            ),
            UsageScope(workspaces=workspaces),
        )
        window = DecisionWindow(workspaces=workspaces, start=start, end=end, user_ids=user_ids)
        async with self._uow:
            totals = await self._rows.totals(window)
            kinds = {sums.key[0]: sums for sums in await self._rows.by_outcome_kind(window)}
            series = await self._rows.series(window, bucket)
            agent_types = await self._rows.by_agent_type(window, limit=SAVINGS_TOP_N)
            swaps = await self._rows.by_swap(window, limit=SAVINGS_TOP_N)
        points = {start: SubagentSavingsPoint(bucket_start=start, **_amounts(sums)) for start, sums in series.items()}
        return SubagentSavingsPublic(
            recommender_available=recommender_available,
            totals=SubagentSavingsTotals(
                **_amounts(totals),
                no_outcome=totals.decisions - totals.with_outcome,
                unpriced=totals.with_outcome - totals.measured - totals.estimated,
                router_unanswered=router_unanswered,
            ),
            measured=SavingsAmounts(**_amounts(kinds.get(SUBAGENT_OUTCOME_MEASURED, _NOTHING))),
            estimated=SavingsAmounts(**_amounts(kinds.get(SUBAGENT_OUTCOME_ESTIMATED, _NOTHING))),
            series=dense_series(
                window.start,
                window.end,
                bucket,
                points,
                lambda start: SubagentSavingsPoint(bucket_start=start, **_amounts(_NOTHING)),
            ),
            by_agent_type=[SavingsByAgentType(agent_type=sums.key[0], **_amounts(sums)) for sums in agent_types],
            by_swap=[SavingsBySwap(baseline_model=sums.key[0], ran_on=sums.key[1], **_amounts(sums)) for sums in swaps],
        )
