"""kNN routing memory: the learned router backend.

Implements the premise that similar prompts behave similarly, so a
nearest-neighbor vote over one user's ``(embedding, {model: quality})`` records
can send each request to the cheapest candidate that is still good enough.

Scoring, per candidate ``m`` over the ``k`` nearest neighbors:

    score(m) = mean_quality(m | neighbors) - alpha * normalized_cost(m)

``alpha`` is the single cost-vs-quality dial and the highest score wins. Every
uncertain case declines instead of guessing (cold pool, sparse neighborhood,
sub-floor confidence, tool-bearing request, unpriced candidate), and a decline
serves the policy's default target, so a learned policy is never worse than the
plain failover policy it was written from.

The store is a linear cosine scan over the user's records held in the gateway DB
(:class:`gateway.models.entities.RoutingMemory`). That holds into the low
thousands of records per user (the ``router_max_records_per_user`` cap); pgvector
or an ANN index is the next step past that, and the reasoning is in
`docs/routing-scaling.md`. Records carry an ``embedding_model`` tag so changing
the embedding model invalidates stale vectors rather than mixing incomparable
spaces.

Known v1 limits (tracked on #187): capability gating is minimal (a request
carrying tools keeps the default target; vision and context-window gating need a
capability registry), learning happens only through the preference API rather
than from live traffic, cost is list price rather than cache-aware, and the
router is standalone-only because hybrid mode resolves models upstream.
"""

from __future__ import annotations

import hashlib
import math
from collections import OrderedDict
from typing import TYPE_CHECKING

from any_llm import aembedding
from any_llm.exceptions import AnyLLMError
from sqlalchemy import delete, func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.database import create_session
from gateway.log_config import logger
from gateway.models.entities import RoutingMemory
from gateway.services.pricing_service import find_model_pricing
from gateway.services.provider_kwargs import resolve_provider_selector
from gateway.services.routing.backends import RoutingContext, RoutingDecision

if TYPE_CHECKING:
    from gateway.core.config import GatewayConfig

__all__ = ["KnnRoutingMemory", "RouterPricingError", "unpriced_router_candidates"]

_TRACE_CACHE_MAX = 10_000


class RouterPricingError(ValueError):
    """A learned policy's candidate has no configured pricing.

    The router scores by cost, so an unpriced candidate has nothing to weigh.
    Rather than guess a price, pricing is required for every candidate: refused
    when a policy is written or loaded, and declined (never a failed request) if
    it goes missing under a running gateway.
    """


def _unit(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return vec
    return [x / norm for x in vec]


def _cosine(a: list[float], b: list[float]) -> float:
    # Both vectors are stored and queried L2-normalized, so the dot product is the
    # cosine similarity.
    n = min(len(a), len(b))
    return sum(a[i] * b[i] for i in range(n))


class KnnRoutingMemory:
    """Learned router backend: kNN over one user's scored examples."""

    def __init__(self, config: GatewayConfig) -> None:
        self.config = config
        self.k = max(1, int(config.router_k))
        self.alpha = float(config.router_alpha)
        self.confidence_floor = float(config.router_confidence_floor)
        self.seed_count = max(0, int(config.router_seed_count))
        self.granularity = config.router_granularity.strip().lower()
        self.embedding_model = config.router_embedding_model
        self.max_records = max(0, int(config.router_max_records_per_user))
        # trace_key -> chosen model, so the turns of one conversation reuse its
        # first decision. Bounded LRU, in-process only: a restart simply re-decides
        # at the next turn, which is safe because every candidate can serve.
        self._trace_decisions: OrderedDict[str, str] = OrderedDict()

    # -- RouterBackend ------------------------------------------------------

    async def rank(self, ctx: RoutingContext) -> RoutingDecision:
        pool = self._pool(ctx)
        if len(pool) <= 1:
            return RoutingDecision.decline("one usable candidate: nothing to route among")

        # Capability gate, minimal and conservative: a tool-bearing request keeps
        # the default target. Full vision and context-window gating needs a
        # capability registry, so until then the strong model handles tool calls.
        if ctx.has_tools:
            return RoutingDecision.decline("tools present: capability-gated to the default target")

        trace_key = self._trace_key(ctx)
        if self.granularity == "trace_sticky" and ctx.is_trace_continuation:
            prior = self._trace_decisions.get(trace_key)
            if prior is not None and prior in pool:
                self._touch_trace(trace_key)
                return RoutingDecision(
                    ordered_models=self._ordered_with_fallthrough([prior], ctx.default_model, pool),
                    confidence=1.0,
                    rationale="trace-sticky reuse of this conversation's first decision",
                )

        # `step` reads the current turn; `trace_sticky` anchors on the opening task
        # so a cache miss (a restart, another replica) reproduces the conversation's
        # first decision instead of drifting turn to turn.
        signal = ctx.task_signal if self.granularity == "step" else ctx.trace_signal
        if not signal.strip():
            return RoutingDecision.decline("no embeddable task signal")

        try:
            query = await self._embed(signal)
        except Exception as exc:  # embedding is best-effort; it never fails a request
            logger.warning("Router embedding failed (%s); serving the policy default", type(exc).__name__)
            return RoutingDecision.decline(f"embedding error ({type(exc).__name__})")

        records = await self._load_records(ctx.user_id, ctx.task_id)
        if len(records) < self.seed_count:
            partition = f" and task '{ctx.task_id}'" if ctx.task_id else ""
            return RoutingDecision.decline(
                f"cold pool: {len(records)}/{self.seed_count} records for this user{partition}"
            )

        neighbors = self._neighbors(query, records)
        if len(neighbors) < self.k:
            return RoutingDecision.decline(f"sparse neighborhood: {len(neighbors)}/{self.k} comparable records")

        try:
            prices = await self._candidate_prices(pool)
        except RouterPricingError as exc:
            # Startup and write-time validation should have caught this, so getting
            # here means pricing was removed under a running gateway. Declining
            # keeps the request served, on the default target.
            logger.warning("%s Serving the policy default.", exc)
            return RoutingDecision.decline(str(exc))

        decision = self._score(ctx, pool, neighbors, prices)
        if self.granularity == "trace_sticky" and decision.ordered_models:
            self._remember_trace(trace_key, decision.ordered_models[0])
        return decision

    # -- preference-collection write path -----------------------------------

    async def record_preference(
        self,
        *,
        user_id: str,
        prompt: str,
        scores: dict[str, float],
        task_id: str | None,
        label_source: str = "human",
    ) -> int:
        """Persist one routing-memory record for this example.

        ``scores`` maps each candidate model to its quality in ``[0.0, 1.0]``
        (1.0 = great, 0.0 = bad). One example is one record, the prompt embedding
        plus the per-model scores, so the kNN later votes over distinct prompts.
        Returns the number of records written (1, or 0 for an empty submission).
        """
        if not prompt.strip() or not scores:
            return 0
        embedding = await self._embed(prompt)
        async with create_session() as db:
            db.add(
                RoutingMemory(
                    user_id=user_id,
                    embedding_model=self.embedding_model,
                    embedding=embedding,
                    qualities={model: float(score) for model, score in scores.items()},
                    task_id=task_id,
                    label_source=label_source,
                )
            )
            try:
                await db.commit()
            except SQLAlchemyError:
                await db.rollback()
                raise
        await self._evict_if_needed(user_id)
        return 1

    # -- scoring -----------------------------------------------------------

    def _score(
        self,
        ctx: RoutingContext,
        pool: list[str],
        neighbors: list[tuple[float, RoutingMemory]],
        prices: dict[str, float],
    ) -> RoutingDecision:
        lo, hi = min(prices.values()), max(prices.values())
        span = hi - lo

        def norm_cost(model: str) -> float:
            return 0.0 if span == 0 else (prices[model] - lo) / span

        scores: dict[str, float] = {}
        for model in pool:
            qualities = [record.qualities[model] for _, record in neighbors if model in record.qualities]
            if not qualities:
                continue
            scores[model] = sum(qualities) / len(qualities) - self.alpha * norm_cost(model)

        if not scores:
            return RoutingDecision.decline("no neighbor scored any candidate in this pool")

        ordered = sorted(scores, key=lambda model: scores[model], reverse=True)
        best = ordered[0]

        # Confidence is the share of the k neighbors whose own best candidate is the
        # winner. A clear local consensus reads as high confidence; if the nearby
        # prompts mostly preferred something else, the pick is weakly supported.
        # Each neighbor is one prompt, so this is a vote over distinct prompts, and
        # only pool candidates count: a neighbor's favorite that this policy cannot
        # dispatch is irrelevant.
        def neighbor_best(qualities: dict[str, float]) -> str | None:
            among = {model: qualities[model] for model in pool if model in qualities}
            return max(among, key=lambda model: among[model]) if among else None

        agree = sum(1 for _, record in neighbors if neighbor_best(record.qualities) == best)
        confidence = agree / len(neighbors)

        if confidence < self.confidence_floor:
            # Lead with the safe (default) model, keeping the rest in score order as
            # the failover chain.
            led = [ctx.default_model, *[model for model in ordered if model != ctx.default_model]]
            return RoutingDecision(
                ordered_models=self._ordered_with_fallthrough(led, ctx.default_model, pool),
                confidence=confidence,
                rationale=(
                    f"confidence {confidence:.2f} below floor {self.confidence_floor:.2f}; led with the default"
                ),
            )
        return RoutingDecision(
            ordered_models=self._ordered_with_fallthrough(ordered, ctx.default_model, pool),
            confidence=confidence,
            rationale=f"kNN cost-biased argmax (alpha={self.alpha:g}, k={self.k})",
        )

    # -- candidate pool / ordering ------------------------------------------

    @staticmethod
    def _pool(ctx: RoutingContext) -> list[str]:
        pool: list[str] = []
        for model in [*ctx.candidate_pool, ctx.default_model]:
            if model and model not in pool:
                pool.append(model)
        return pool

    @staticmethod
    def _ordered_with_fallthrough(ordered: list[str], default_model: str, pool: list[str]) -> list[str]:
        """Score order, with the default target guaranteed present as the last resort.

        The chosen model leads. The default target is the cascade's final
        fallthrough, so it moves to the end unless it is itself the choice, in which
        case it stays first and is not demoted.
        """
        result = [model for model in ordered if model in pool]
        for model in pool:
            if model not in result:
                result.append(model)
        if result and result[0] != default_model and default_model in result:
            result.remove(default_model)
            result.append(default_model)
        return result

    # -- storage / retrieval ------------------------------------------------

    async def _load_records(self, user_id: str, task_id: str | None) -> list[RoutingMemory]:
        """Load a user's records for the current embedding model.

        A ``task_id`` is a hard partition: only records carrying that label load,
        so the cold-start gate counts that partition alone and a request stays on
        the default target until its own task is warm. Records from other tasks
        never influence it. With no task, every record the user has is in play.
        """
        async with create_session() as db:
            stmt = select(RoutingMemory).where(
                RoutingMemory.user_id == user_id,
                RoutingMemory.embedding_model == self.embedding_model,
            )
            if task_id is not None:
                stmt = stmt.where(RoutingMemory.task_id == task_id)
            return list((await db.execute(stmt)).scalars().all())

    def _neighbors(self, query: list[float], records: list[RoutingMemory]) -> list[tuple[float, RoutingMemory]]:
        sims = [(_cosine(query, record.embedding), record) for record in records]
        sims.sort(key=lambda pair: pair[0], reverse=True)
        return sims[: self.k]

    async def _evict_if_needed(self, user_id: str) -> None:
        """Keep at most ``max_records`` of the newest records per user."""
        if self.max_records <= 0:
            return
        async with create_session() as db:
            count = (
                await db.execute(
                    select(func.count()).select_from(RoutingMemory).where(RoutingMemory.user_id == user_id)
                )
            ).scalar_one()
            if count <= self.max_records:
                return
            keep_ids = (
                (
                    await db.execute(
                        select(RoutingMemory.id)
                        .where(RoutingMemory.user_id == user_id)
                        .order_by(RoutingMemory.created_at.desc())
                        .limit(self.max_records)
                    )
                )
                .scalars()
                .all()
            )
            await db.execute(
                delete(RoutingMemory).where(
                    RoutingMemory.user_id == user_id,
                    RoutingMemory.id.notin_(keep_ids),
                )
            )
            try:
                await db.commit()
            except SQLAlchemyError:
                await db.rollback()
                raise

    # -- pricing -----------------------------------------------------------

    async def _candidate_prices(self, pool: list[str]) -> dict[str, float]:
        async with create_session() as db:
            return {model: await self._input_price(db, model) for model in pool}

    async def _input_price(self, db: AsyncSession, selector: str) -> float:
        """Input price per million tokens for one candidate.

        Resolved through ``resolve_provider_selector`` rather than split by hand so
        the lookup keys on the same ``instance:model`` the request will be billed
        under. A candidate naming a provider *instance* would otherwise be priced
        against its implementation name and look unpriced.
        """
        try:
            resolved = resolve_provider_selector(self.config, selector)
        except (ValueError, AnyLLMError) as exc:
            raise RouterPricingError(f"Router candidate '{selector}' does not resolve to a provider.") from exc
        pricing = await find_model_pricing(db, resolved.instance, resolved.model)
        if pricing is None:
            raise RouterPricingError(f"Router candidate '{selector}' has no configured pricing.")
        return float(pricing.input_price_per_million)

    # -- embedding ---------------------------------------------------------

    async def _embed(self, text: str) -> list[float]:
        resolved = resolve_provider_selector(self.config, self.embedding_model)
        result = await aembedding(
            model=resolved.model, inputs=text, provider=resolved.provider, **resolved.kwargs
        )
        vector = list(result.data[0].embedding)
        return _unit([float(x) for x in vector])

    # -- trace memory ------------------------------------------------------

    def _trace_key(self, ctx: RoutingContext) -> str:
        """Per-(user, task) trace identity for trace-sticky reuse.

        The client-supplied conversation id when present, otherwise a hash of the
        conversation's opening text. Both are namespaced by user and task, so one
        conversation id never collides across users or across partitions.
        """
        explicit = ctx.trace_key.strip() if ctx.trace_key else ""
        anchor = explicit or ctx.trace_anchor
        raw = f"{ctx.user_id}\x00{ctx.task_id or ''}\x00{anchor}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _remember_trace(self, trace_key: str, model: str) -> None:
        self._trace_decisions[trace_key] = model
        self._trace_decisions.move_to_end(trace_key)
        while len(self._trace_decisions) > _TRACE_CACHE_MAX:
            self._trace_decisions.popitem(last=False)

    def _touch_trace(self, trace_key: str) -> None:
        if trace_key in self._trace_decisions:
            self._trace_decisions.move_to_end(trace_key)


async def unpriced_router_candidates(
    config: GatewayConfig, db: AsyncSession, candidates: list[str]
) -> list[str]:
    """Which of ``candidates`` have no configured pricing.

    Shared by the two places that must refuse rather than decline: startup
    validation over the config policies, and the write path for a stored one. The
    router scores by cost, so a candidate with no price could never be compared,
    and a policy that silently never routes is worse than one that fails to load.
    """
    missing: list[str] = []
    for selector in candidates:
        try:
            resolved = resolve_provider_selector(config, selector)
        except (ValueError, AnyLLMError):
            # An unresolvable selector is refused elsewhere, with a clearer message.
            continue
        if await find_model_pricing(db, resolved.instance, resolved.model) is None:
            missing.append(selector)
    return missing
