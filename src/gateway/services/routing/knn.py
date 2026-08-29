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

The store is a linear cosine scan over the records the requesting user has in the
requesting workspace, held in the gateway DB
(:class:`gateway.models.entities.RoutingMemory`). That holds into the low
thousands of records per partition (the ``router_max_records_per_user`` cap,
which bounds one user's records in one workspace, since that is what a decision
loads); larger pools need an indexed vector store. Records carry an
``embedding_model`` tag so changing
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
import uuid
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
# Read bound when eviction is disabled (`router_max_records_per_user = 0`). Matches
# the field's own default, so turning eviction off does not turn the read unbounded.
_DEFAULT_READ_LIMIT = 5000


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
        # What one decision may load. `max_records` is the operator's own answer to
        # "how many records should this router use"; 0 means eviction is off, which
        # is not a license for an unbounded select, so fall back to the default.
        self._read_limit = self.max_records or _DEFAULT_READ_LIMIT
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

        records = await self._load_records(ctx.user_id, ctx.task_id, ctx.workspace_id)
        if len(records) < self.seed_count:
            partition = f" and task '{ctx.task_id}'" if ctx.task_id else ""
            return RoutingDecision.decline(
                f"cold pool: {len(records)}/{self.seed_count} records for this user{partition}"
            )

        neighbors = self._neighbors(query, records)
        if len(neighbors) < self.k:
            return RoutingDecision.decline(f"sparse neighborhood: {len(neighbors)}/{self.k} comparable records")

        try:
            prices = await self._candidate_prices(pool, workspace_id=ctx.workspace_id)
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
        workspace_id: uuid.UUID,
        prompt: str,
        scores: dict[str, float],
        task_id: str | None,
        label_source: str = "human",
    ) -> int:
        """Persist one routing-memory record for this example.

        ``scores`` maps each candidate model to its quality in ``[0.0, 1.0]``
        (1.0 = great, 0.0 = bad), keyed on canonical ``instance:model``; the caller
        canonicalizes, and :meth:`_score` canonicalizes what it reads, so a
        candidate's spelling never decides whether it matches. One example is one
        record, the prompt embedding plus the per-model scores, so the kNN later
        votes over distinct prompts. Returns the number of records written (1, or 0
        for an empty submission).
        """
        if not prompt.strip() or not scores:
            return 0
        embedding = await self._embed(prompt)
        async with create_session() as db:
            db.add(
                RoutingMemory(
                    user_id=user_id,
                    workspace_id=workspace_id,
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
        await self._evict_if_needed(user_id, workspace_id)
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

        # Candidates and stored scores are matched on canonical `instance:model`, not
        # on the spelling either side happens to use. `openai/gpt-4o`,
        # `openai:gpt-4o`, and an alias pointing at it name one model, so a candidate
        # whose policy spells it differently from the example that taught it must
        # still match: a miss is silent, and leaves the cheap candidate scoreless
        # while the pool reports warm.
        cache: dict[str, str] = {}
        key_of = {
            model: self._canonical(model, ctx.user_id, cache, workspace_id=ctx.workspace_id) for model in pool
        }
        recorded = [
            self._canonical_qualities(record.qualities, ctx.user_id, cache, workspace_id=ctx.workspace_id)
            for _, record in neighbors
        ]

        scores: dict[str, float] = {}
        for model in pool:
            key = key_of[model]
            qualities = [record[key] for record in recorded if key in record]
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
            among = {model: qualities[key_of[model]] for model in pool if key_of[model] in qualities}
            return max(among, key=lambda model: among[model]) if among else None

        agree = sum(1 for qualities in recorded if neighbor_best(qualities) == best)
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

    # -- model identity ------------------------------------------------------

    def _canonical(
        self,
        selector: str,
        user_id: str | None,
        cache: dict[str, str],
        *,
        workspace_id: uuid.UUID | None = None,
    ) -> str:
        """``instance:model`` for a selector, or the selector unchanged if it resolves to nothing.

        An unresolvable selector keeps its own spelling rather than dropping out, so
        it can still match a stored key spelled the same way; scoring is not the
        place to decide a candidate is invalid.

        ``workspace_id`` is the request's, and has to be: a candidate selector can
        name an alias, and this workspace's alias may point somewhere the default
        workspace's does not. ``usable_candidates`` filtered the pool in this
        workspace and ``/rank`` canonicalized the stored keys in it, so resolving
        here in another one produces a key that matches neither, and the candidate
        drops out of scoring with no score and no error.

        ``cache`` is per decision: the pool and the neighbors' stored keys repeat the
        same handful of selectors, and each resolution walks the alias and
        static-policy tables. Keying it on the selector alone is safe because one
        decision has one workspace.
        """
        canonical = cache.get(selector)
        if canonical is None:
            try:
                resolved = resolve_provider_selector(
                    self.config, selector, user_id, workspace_id=workspace_id
                )
                canonical = f"{resolved.instance}:{resolved.model}"
            except (ValueError, AnyLLMError):
                canonical = selector
            cache[selector] = canonical
        return canonical

    def _canonical_qualities(
        self,
        qualities: dict[str, float],
        user_id: str | None,
        cache: dict[str, str],
        *,
        workspace_id: uuid.UUID | None = None,
    ) -> dict[str, float]:
        """One record's scores, rekeyed on canonical model identity.

        The write path canonicalizes too, so this matters for records written before
        it did: they hold whatever spelling was sent, and rekeying on read is what
        makes them usable instead of stranding them behind a schema migration.
        Two spellings of one model in one record (which ``/rank`` refuses) keep the
        first, so the collision resolves the same way on every request.
        """
        canonical: dict[str, float] = {}
        for model, quality in qualities.items():
            canonical.setdefault(self._canonical(model, user_id, cache, workspace_id=workspace_id), quality)
        return canonical

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

    async def _load_records(
        self, user_id: str, task_id: str | None, workspace_id: uuid.UUID | None = None
    ) -> list[RoutingMemory]:
        """Load a user's records for the current embedding model, in one workspace.

        A ``task_id`` is a hard partition: only records carrying that label load,
        so the cold-start gate counts that partition alone and a request stays on
        the default target until its own task is warm. Records from other tasks
        never influence it. With no task, every record the user has is in play.

        ``workspace_id`` is a hard partition too: examples labeled in one
        workspace never steer another's traffic, even for a user who holds keys in
        both. Omitted only where there is no request to route (no backend is asked
        to rank there), which reads every workspace the user has records in.
        """
        async with create_session() as db:
            stmt = select(RoutingMemory).where(
                RoutingMemory.user_id == user_id,
                RoutingMemory.embedding_model == self.embedding_model,
            )
            if workspace_id is not None:
                stmt = stmt.where(RoutingMemory.workspace_id == workspace_id)
            if task_id is not None:
                stmt = stmt.where(RoutingMemory.task_id == task_id)
            # Newest first, and bounded. Eviction is enforced lazily on write, and
            # only for the (user, workspace) set rather than per task, so nothing
            # else stops this select from growing without limit: a request would
            # then load and cosine-score every row it finds. The cap is what the
            # operator already configured as "how many records this router uses",
            # and taking the newest is the same rule eviction applies.
            stmt = stmt.order_by(RoutingMemory.created_at.desc()).limit(self._read_limit)
            return list((await db.execute(stmt)).scalars().all())

    def _neighbors(self, query: list[float], records: list[RoutingMemory]) -> list[tuple[float, RoutingMemory]]:
        sims = [(_cosine(query, record.embedding), record) for record in records]
        sims.sort(key=lambda pair: pair[0], reverse=True)
        return sims[: self.k]

    async def _evict_if_needed(self, user_id: str, workspace_id: uuid.UUID) -> None:
        """Keep at most ``max_records`` of the newest records per user and workspace.

        Applied within the partition the router reads, not across every workspace
        the user has records in. ``router_max_records_per_user`` bounds what one
        decision loads and scores, and a decision only ever loads one workspace,
        so evicting across them would have a busy workspace delete labels another
        one is still routing on.
        """
        if self.max_records <= 0:
            return
        partition = (RoutingMemory.user_id == user_id, RoutingMemory.workspace_id == workspace_id)
        async with create_session() as db:
            count = (
                await db.execute(select(func.count()).select_from(RoutingMemory).where(*partition))
            ).scalar_one()
            if count <= self.max_records:
                return
            # Delete by timestamp rather than by an id NOT IN list. The list would
            # hold `max_records` ids (5000 by default), and SQLite caps host
            # parameters per statement at 999 on builds before 3.32, so the eviction
            # that keeps the store bounded would itself fail on the default config.
            cutoff = (
                await db.execute(
                    select(RoutingMemory.created_at)
                    .where(*partition)
                    .order_by(RoutingMemory.created_at.desc())
                    .offset(self.max_records - 1)
                    .limit(1)
                )
            ).scalar_one_or_none()
            if cutoff is None:
                return
            # Strictly older than the oldest row being kept. Rows sharing that exact
            # timestamp are kept, so a batch written in one tick is never half
            # evicted; the count can sit slightly above the cap until the next write.
            await db.execute(
                delete(RoutingMemory).where(
                    *partition,
                    RoutingMemory.created_at < cutoff,
                )
            )
            try:
                await db.commit()
            except SQLAlchemyError:
                await db.rollback()
                raise

    # -- pricing -----------------------------------------------------------

    async def _candidate_prices(
        self, pool: list[str], *, workspace_id: uuid.UUID | None = None
    ) -> dict[str, float]:
        async with create_session() as db:
            return {model: await self._input_price(db, model, workspace_id=workspace_id) for model in pool}

    async def _input_price(
        self, db: AsyncSession, selector: str, *, workspace_id: uuid.UUID | None = None
    ) -> float:
        """Input price per million tokens for one candidate.

        Resolved through ``resolve_provider_selector`` rather than split by hand so
        the lookup keys on the same ``instance:model`` the request will be billed
        under. A candidate naming a provider *instance* would otherwise be priced
        against its implementation name and look unpriced.

        ``workspace_id`` decides *which model* the selector names, for the same
        reason it does in :meth:`_canonical`: an alias can point somewhere
        different per workspace, and pricing the default workspace's target while
        scoring this workspace's would rank on a price the request never pays.
        That is separate from *whose rates* apply, which the next paragraph is
        about and which this deliberately leaves alone.

        Deliberately reads the *deployment* price list, not an organization's rate
        overrides, so a router's ranking can differ from what the chosen request
        settles at. Three reasons it stays that way. This is a preference and not
        a charge, so nothing is billed wrong. ``decide_ordering`` carries no key
        and no session (``_candidate_prices`` opens its own), and the surface it
        shares with ``explain`` and the CLI is required to stay callable with no
        request at all, so an organization could only arrive by widening that
        boundary. And the refusal below cannot fire spuriously for an
        override-only price: ``unpriced_router_candidates`` validates a policy's
        candidates against the deployment list when the policy is written, so a
        candidate priced only by an override never becomes a stored policy.
        """
        try:
            resolved = resolve_provider_selector(self.config, selector, workspace_id=workspace_id)
        except (ValueError, AnyLLMError) as exc:
            raise RouterPricingError(f"Router candidate '{selector}' does not resolve to a provider.") from exc
        pricing = await find_model_pricing(db, resolved.instance, resolved.model)
        if pricing is None:
            raise RouterPricingError(f"Router candidate '{selector}' has no configured pricing.")
        return float(pricing.input_price_per_million)

    # -- embedding ---------------------------------------------------------

    async def _embed(self, text: str) -> list[float]:
        """Embed ``text`` with the deployment's configured embedding model.

        Resolved with no workspace, and deliberately, unlike every other selector
        this class resolves. ``router_embedding_model`` is operator configuration
        rather than caller input, which is the case ``resolve_provider_selector``
        documents as workspace-less. More than a convention here: every stored
        vector is tagged with this model name and compared against others carrying
        the same tag, so if two workspaces resolved that one name to different
        providers, their vectors would be silently incomparable while claiming to
        be the same space. Scoping this would create the bug that scoping the
        others fixes.
        """
        resolved = resolve_provider_selector(self.config, self.embedding_model)
        result = await aembedding(
            model=resolved.model, inputs=text, provider=resolved.provider, **resolved.kwargs
        )
        vector = list(result.data[0].embedding)
        return _unit([float(x) for x in vector])

    # -- trace memory ------------------------------------------------------

    def _trace_key(self, ctx: RoutingContext) -> str:
        """Per-(workspace, user, task) trace identity for trace-sticky reuse.

        The client-supplied conversation id when present, otherwise a hash of the
        conversation's opening text. Both are namespaced by workspace, user and
        task, so one conversation id never collides across any of the three.

        The workspace belongs here for the same reason it is on
        :meth:`_load_records`, and is load-bearing rather than tidy: this cache is
        consulted *before* any record loads, so without it a user sending one
        conversation id in two workspaces would have workspace A's decision
        replayed in workspace B, out of process memory, having never read
        workspace B's examples at all.
        """
        explicit = ctx.trace_key.strip() if ctx.trace_key else ""
        anchor = explicit or ctx.trace_anchor
        raw = f"{ctx.workspace_id or ''}\x00{ctx.user_id}\x00{ctx.task_id or ''}\x00{anchor}"
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
    config: GatewayConfig,
    db: AsyncSession,
    candidates: list[str],
    *,
    workspace_id: uuid.UUID | None = None,
) -> list[str]:
    """Which of ``candidates`` have no configured pricing.

    Shared by the two places that must refuse rather than decline: startup
    validation over the config policies, and the write path for a stored one. The
    router scores by cost, so a candidate with no price could never be compared,
    and a policy that silently never routes is worse than one that fails to load.

    ``workspace_id`` is the workspace the policy is being stored into, so a
    candidate naming an alias is validated as it will resolve for the requests
    that policy will serve. Omitted by startup validation, which reads the
    ``config.yml`` policies: those are deployment-wide, so the default workspace
    is the only workspace-shaped answer there is.
    """
    missing: list[str] = []
    for selector in candidates:
        try:
            resolved = resolve_provider_selector(config, selector, workspace_id=workspace_id)
        except (ValueError, AnyLLMError):
            # An unresolvable selector is refused elsewhere, with a clearer message.
            continue
        if await find_model_pricing(db, resolved.instance, resolved.model) is None:
            missing.append(selector)
    return missing
