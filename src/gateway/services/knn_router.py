"""kNN routing-memory backend: the default learned router.

Implements the design's central premise: similar prompts behave similarly, so a
nearest-neighbor vote over per-tenant ``(embedding, {model: quality})`` records,
one per scored example, can send each request to the cheapest model that is still
good enough.

Scoring (per candidate model ``m`` over the ``k`` nearest neighbors):

    score(m) = mean_quality(m | neighbors) - alpha * normalized_cost(m)

``alpha`` is the single cost-vs-quality dial; the candidate with the highest
score wins. Cold start, a sparse neighbor set, or sub-floor confidence all fall
back to the requested (strong) model so the safe default is never worse than no
routing.

The store is per-tenant cosine kNN over vectors held in the gateway DB
(``RoutingMemory``), scanned linearly in Python. That is fine into the low
thousands of vectors per tenant (the ``router_max_vectors_per_tenant`` cap);
pgvector / ANN is the documented next step past that. Records carry an
``embedding_model`` tag so changing the embedding model invalidates stale
vectors rather than mixing incomparable spaces.

Known v1 limitations (tracked on the design doc / issue #187): capability gating
is minimal (requests with tools stay on the requested model; full vision /
context-window gating needs a capability registry), passive learning from live
traffic and the cache-aware / switch-penalty cost terms are fast-follows, and
platform-mode routing memory is out of scope.
"""

from __future__ import annotations

import hashlib
import math
from collections import OrderedDict
from typing import TYPE_CHECKING

from any_llm import AnyLLM, aembedding
from sqlalchemy import delete, func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.database import create_session
from gateway.log_config import logger
from gateway.models.entities import RoutingMemory
from gateway.services.pricing_service import find_model_pricing
from gateway.services.provider_kwargs import get_provider_kwargs
from gateway.services.router_backend import RoutingContext, RoutingDecision

if TYPE_CHECKING:
    from gateway.core.config import GatewayConfig

_TRACE_CACHE_MAX = 10_000


class RouterPricingError(ValueError):
    """A model in the kNN router candidate pool has no configured pricing.

    The router scores candidates by cost, so an unpriced candidate has nothing to
    weigh. Rather than guess a price, the router requires pricing for every model
    in the pool, validated at startup and enforced again at request time.
    """


def _unit(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return vec
    return [x / norm for x in vec]


def _cosine(a: list[float], b: list[float]) -> float:
    # Both vectors are stored/queried L2-normalized, so the dot product is the
    # cosine similarity.
    n = min(len(a), len(b))
    return sum(a[i] * b[i] for i in range(n))


def _latest_user_text(messages: list[dict[str, object]]) -> str:
    """Best-effort task signal: the last user turn's text.

    Handles both string content and OpenAI content-part lists; ignores non-text
    parts (images, files). Falls back to the last message of any role when no
    user turn is present.
    """
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        text = _content_text(msg.get("content"))
        if text:
            return text
    if messages:
        return _content_text(messages[-1].get("content"))
    return ""


def _first_user_text(messages: list[dict[str, object]]) -> str:
    """The first user turn's text: the stable task signal for a trace."""
    for msg in messages:
        if msg.get("role") != "user":
            continue
        text = _content_text(msg.get("content"))
        if text:
            return text
    return _latest_user_text(messages)


def _trace_anchor_text(messages: list[dict[str, object]]) -> str:
    """The conversation's opening text: every turn before the first assistant reply.

    This is the trace-identity fallback when no conversation id is supplied. It is
    stable across a conversation's turns (the opening system + user turns do not
    change) and richer than the first user turn alone, so it disambiguates traces
    that share a first user message but differ in their system preamble. It still
    cannot tell apart two conversations whose entire opening is identical; only a
    client-supplied conversation id can. See docs/routing-scaling.md.
    """
    parts: list[str] = []
    for msg in messages:
        if msg.get("role") == "assistant":
            break
        text = _content_text(msg.get("content"))
        if text:
            parts.append(text)
    if parts:
        return "\n".join(parts)
    return _first_user_text(messages)


def _content_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return ""


class KnnRoutingMemory:
    """Default learned routing backend (kNN over per-tenant routing memory)."""

    def __init__(self, config: GatewayConfig) -> None:
        self.config = config
        self.k = max(1, int(config.router_k))
        self.alpha = float(config.router_alpha)
        self.confidence_floor = float(config.router_confidence_floor)
        self.seed_count = max(0, int(config.router_seed_count))
        self.granularity = config.router_granularity.strip().lower()
        self.embedding_model = config.router_embedding_model
        self.max_vectors = max(0, int(config.router_max_vectors_per_tenant))
        # trace_key -> chosen model, for trace-sticky reuse across the calls of
        # one agent trace. Bounded LRU; in-process only (a restart simply
        # re-routes at the next trace step, which is safe).
        self._trace_decisions: OrderedDict[str, str] = OrderedDict()

    # -- public RouterBackend protocol -------------------------------------

    async def route(self, ctx: RoutingContext) -> RoutingDecision:
        pool = self._candidate_pool(ctx)
        if len(pool) <= 1:
            return self._passthrough(ctx, "single-candidate pool: nothing to route among")

        # Capability gate (minimal, conservative): do not reroute tool-bearing
        # requests in v1. Full vision / context-window gating needs a capability
        # registry; until then the requested model is kept for tool calls.
        if ctx.has_tools:
            return self._passthrough(ctx, "tools present: capability-gated to requested model")

        trace_key = self._trace_key(ctx)
        if self.granularity == "trace_sticky" and ctx.is_trace_continuation:
            prior = self._trace_decisions.get(trace_key)
            if prior is not None:
                self._touch_trace(trace_key)
                return RoutingDecision(
                    ordered_models=self._order_with_fallthrough([prior], ctx.requested_model, pool),
                    confidence=1.0,
                    rationale="trace-sticky reuse of the trace's first decision",
                )

        # `step` routes on the current turn; `trace_sticky` anchors on the
        # conversation's opening task so a cache miss (e.g. a different replica)
        # reproduces the trace's first decision instead of drifting per turn.
        signal = _latest_user_text(ctx.messages) if self.granularity == "step" else _first_user_text(ctx.messages)
        if not signal.strip():
            return self._passthrough(ctx, "no embeddable task signal")

        try:
            query = await self._embed(signal)
        except Exception as exc:  # embedding is best-effort; never fail the request
            logger.warning("Router embedding failed (%s); passing through to requested model", type(exc).__name__)
            return self._passthrough(ctx, "embedding error: route requested model")

        records, total = await self._load_records(ctx.tenant_id, ctx.task_id)
        if total < self.seed_count:
            return self._passthrough(
                ctx, f"cold tenant: {total}/{self.seed_count} seed records, route requested model"
            )

        neighbors = self._neighbors(query, records)
        if len(neighbors) < self.k:
            return self._passthrough(ctx, "sparse neighborhood: route requested model")

        prices = await self._candidate_prices(pool)
        decision = self._score(ctx, pool, neighbors, prices)

        if self.granularity == "trace_sticky":
            self._remember_trace(trace_key, decision.ordered_models[0])
        return decision

    # -- preference-collection write path ----------------------------------

    async def record_preference(
        self,
        *,
        tenant_id: str,
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
                    tenant_id=tenant_id,
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
        await self._evict_if_needed(tenant_id)
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
            qualities = [r.qualities[model] for _, r in neighbors if model in r.qualities]
            if not qualities:
                continue
            scores[model] = sum(qualities) / len(qualities) - self.alpha * norm_cost(model)

        if not scores:
            return self._passthrough(ctx, "no neighbor signal for any candidate")

        ordered = sorted(scores, key=lambda m: scores[m], reverse=True)
        best = ordered[0]

        # Confidence = fraction of the k neighbor prompts whose own best-scoring
        # candidate is the chosen model. A clear local consensus reads as high
        # confidence; if the nearby prompts mostly preferred a different candidate
        # the pick is weakly supported. Each neighbor is one prompt, so this is a
        # vote over distinct prompts. Only pool candidates count: a neighbor's
        # favorite model that is not in the candidate pool is irrelevant here.
        def _neighbor_best(qualities: dict[str, float]) -> str | None:
            among = {m: qualities[m] for m in pool if m in qualities}
            return max(among, key=lambda m: among[m]) if among else None

        agree = sum(1 for _, r in neighbors if _neighbor_best(r.qualities) == best)
        confidence = agree / len(neighbors)

        if confidence < self.confidence_floor:
            # Lead with the safe (requested) model, keeping the rest in score order.
            led = [ctx.requested_model] + [m for m in ordered if m != ctx.requested_model]
            return RoutingDecision(
                ordered_models=self._order_with_fallthrough(led, ctx.requested_model, pool),
                confidence=confidence,
                rationale=f"confidence {confidence:.2f} below floor; lead with requested model",
            )
        return RoutingDecision(
            ordered_models=self._order_with_fallthrough(ordered, ctx.requested_model, pool),
            confidence=confidence,
            rationale=f"kNN cost-biased argmax (alpha={self.alpha}, k={self.k})",
        )

    # -- candidate pool / ordering -----------------------------------------

    def _candidate_pool(self, ctx: RoutingContext) -> list[str]:
        pool: list[str] = []
        for model in ctx.candidate_pool:
            if model and model not in pool:
                pool.append(model)
        if ctx.requested_model and ctx.requested_model not in pool:
            pool.append(ctx.requested_model)
        return pool

    @staticmethod
    def _order_with_fallthrough(ordered: list[str], requested_model: str, pool: list[str]) -> list[str]:
        """Score order, with the requested model guaranteed present as the tail safety net.

        The chosen model (``ordered[0]``) always leads. The requested model is
        the cascade's final fallthrough, so it is moved to the end UNLESS it is
        itself the chosen model (then it stays first and is not demoted).
        """
        result = [m for m in ordered if m in pool]
        for model in pool:
            if model not in result:
                result.append(model)
        if result and result[0] != requested_model and requested_model in result:
            result.remove(requested_model)
            result.append(requested_model)
        return result

    def _passthrough(self, ctx: RoutingContext, rationale: str) -> RoutingDecision:
        return RoutingDecision(
            ordered_models=[ctx.requested_model],
            confidence=0.0,
            rationale=rationale,
        )

    # -- storage / retrieval -----------------------------------------------

    async def _load_records(
        self, tenant_id: str, task_id: str | None
    ) -> tuple[list[RoutingMemory], int]:
        """Load a tenant's vectors for the current embedding model.

        When ``task_id`` is set the result is a hard partition: only records
        carrying that task label are loaded, and the returned count (which drives
        the cold-start seed gate) counts only that partition, so a request stays
        in pass-through until its own task is warm and records from other tasks
        never influence it. With no task the tenant's whole pool is used.
        """
        async with create_session() as db:
            stmt = select(RoutingMemory).where(
                RoutingMemory.tenant_id == tenant_id,
                RoutingMemory.embedding_model == self.embedding_model,
            )
            if task_id is not None:
                stmt = stmt.where(RoutingMemory.task_id == task_id)
            rows = list((await db.execute(stmt)).scalars().all())
        return rows, len(rows)

    def _neighbors(
        self, query: list[float], records: list[RoutingMemory]
    ) -> list[tuple[float, RoutingMemory]]:
        sims = [(_cosine(query, r.embedding), r) for r in records]
        sims.sort(key=lambda x: x[0], reverse=True)
        return sims[: self.k]

    async def _evict_if_needed(self, tenant_id: str) -> None:
        """Keep at most ``max_vectors`` of the newest records per tenant."""
        if self.max_vectors <= 0:
            return
        async with create_session() as db:
            count = (
                await db.execute(
                    select(func.count())
                    .select_from(RoutingMemory)
                    .where(RoutingMemory.tenant_id == tenant_id)
                )
            ).scalar_one()
            if count <= self.max_vectors:
                return
            keep_ids = (
                await db.execute(
                    select(RoutingMemory.id)
                    .where(RoutingMemory.tenant_id == tenant_id)
                    .order_by(RoutingMemory.created_at.desc())
                    .limit(self.max_vectors)
                )
            ).scalars().all()
            await db.execute(
                delete(RoutingMemory).where(
                    RoutingMemory.tenant_id == tenant_id,
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
            return {model: await self._model_input_price(db, model) for model in pool}

    @staticmethod
    async def _model_input_price(db: AsyncSession, model: str) -> float:
        provider, bare = AnyLLM.split_model_provider(model)
        pricing = await find_model_pricing(db, provider, bare)
        if pricing is None:
            raise RouterPricingError(f"Router candidate '{model}' has no configured pricing.")
        return float(pricing.input_price_per_million)

    # -- embedding ---------------------------------------------------------

    async def _embed(self, text: str) -> list[float]:
        provider, model = AnyLLM.split_model_provider(self.embedding_model)
        kwargs = get_provider_kwargs(self.config, provider)
        result = await aembedding(model=model, inputs=text, provider=provider, **kwargs)
        vector = list(result.data[0].embedding)
        return _unit([float(x) for x in vector])

    # -- trace memory ------------------------------------------------------

    def _trace_key(self, ctx: RoutingContext) -> str:
        """Per-(tenant, task) trace identity for trace-sticky reuse.

        Uses the client-supplied conversation id (``ctx.trace_key``) when present,
        otherwise a hash of the conversation's opening text. Both are namespaced by
        tenant and task so the same conversation id never collides across tenants
        or across routing-memory partitions.
        """
        explicit = ctx.trace_key.strip() if ctx.trace_key else ""
        anchor = explicit or _trace_anchor_text(ctx.messages)
        raw = f"{ctx.tenant_id}\x00{ctx.task_id or ''}\x00{anchor}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _remember_trace(self, trace_key: str, model: str) -> None:
        self._trace_decisions[trace_key] = model
        self._trace_decisions.move_to_end(trace_key)
        while len(self._trace_decisions) > _TRACE_CACHE_MAX:
            self._trace_decisions.popitem(last=False)

    def _touch_trace(self, trace_key: str) -> None:
        if trace_key in self._trace_decisions:
            self._trace_decisions.move_to_end(trace_key)


async def validate_router_pricing(config: GatewayConfig, db: AsyncSession) -> None:
    """Fail fast at startup if a kNN router candidate has no configured pricing.

    The router scores candidates by cost, so every model in
    ``OTARI_ROUTER_CANDIDATES`` must have pricing. No-op unless the kNN backend is
    configured with a candidate pool.
    """
    if config.router_backend.strip().lower() != "knn":
        return
    candidates = [m.strip() for m in config.router_candidates.split(",") if m.strip()]
    missing = [
        model for model in candidates if await find_model_pricing(db, *AnyLLM.split_model_provider(model)) is None
    ]
    if missing:
        raise RouterPricingError(
            f"Router candidates without configured pricing: {', '.join(missing)}. "
            "Add pricing for every OTARI_ROUTER_CANDIDATES model; the router scores by cost."
        )
