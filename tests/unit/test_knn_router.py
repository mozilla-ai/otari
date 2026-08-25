"""Unit tests for the kNN router's decision logic.

These isolate scoring, ordering, and trace stickiness from the database and the
embedding provider by patching the backend's async I/O helpers (``_embed``,
``_load_records``, ``_candidate_prices``). What is exercised is the part that
decides *which* candidate wins: cost-biased kNN voting, the decline paths (cold
pool, sparse neighborhood, tools, embedding failure), the confidence floor,
trace-sticky reuse, and the default-target fallthrough ordering.

A decline is an empty ``ordered_models``, which the compiler turns into "serve the
policy's default target". The prompt text arrives already flattened, so the
message-shape helpers that produce it are tested in ``test_routing_signal.py``.

Each routing-memory record is one example: a prompt embedding plus a
``{model: quality}`` map, so the kNN votes over distinct prompts.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterator
from decimal import Decimal
from types import SimpleNamespace
from typing import Any, cast

import pytest

from gateway.api.routes._helpers import conversation_opening_text, first_user_text, latest_user_text
from gateway.core.config import GatewayConfig
from gateway.models.entities import RoutingMemory
from gateway.services import alias_service
from gateway.services.routing import knn
from gateway.services.routing.backends import RoutingContext
from gateway.services.routing.knn import KnnRoutingMemory

CHEAP = "openai/gpt-3.5-turbo"
STRONG = "openai/gpt-4o"

PRICES = {CHEAP: 1.0, STRONG: 10.0}


def _backend(**overrides: Any) -> KnnRoutingMemory:
    kwargs: dict[str, Any] = {"router_k": 2, "router_seed_count": 2}
    kwargs.update(overrides)
    return KnnRoutingMemory(GatewayConfig(**kwargs))


def _mem(qualities: dict[str, float], vec: tuple[float, ...] = (1.0, 0.0)) -> RoutingMemory:
    """One example: a prompt embedding plus each model's quality on it."""
    return RoutingMemory(user_id="u", embedding_model="m", embedding=list(vec), qualities=dict(qualities))


def _both_good() -> RoutingMemory:
    return _mem({CHEAP: 1.0, STRONG: 1.0})


def _cheap_fails() -> RoutingMemory:
    return _mem({CHEAP: 0.0, STRONG: 1.0})


def _ctx(
    *,
    candidates: tuple[str, ...] = (CHEAP, STRONG),
    default: str = STRONG,
    messages: list[dict[str, Any]] | None = None,
    user_id: str = "u",
    **kw: Any,
) -> RoutingContext:
    """A routing context, with the prompt signals derived from ``messages``.

    Built through the same helpers the endpoints use, so these tests exercise the
    real signal the router sees rather than a hand-written approximation.
    """
    convo = messages or [{"role": "user", "content": "hello"}]
    return RoutingContext(
        user_id=user_id,
        default_model=default,
        candidate_pool=list(candidates),
        task_signal=latest_user_text(convo),
        trace_signal=first_user_text(convo),
        trace_anchor=conversation_opening_text(convo),
        **kw,
    )


def _wire(
    backend: KnnRoutingMemory,
    records: list[RoutingMemory],
    *,
    prices: dict[str, float] | None = None,
    query: tuple[float, ...] = (1.0, 0.0),
    total: int | None = None,
) -> None:
    async def _embed(text: str) -> list[float]:
        return list(query)

    async def _load(
        user_id: str, task_id: str | None, workspace_id: uuid.UUID | None = None
    ) -> list[RoutingMemory]:
        # `total` pads the record count without inventing neighbors, which is how
        # the seed gate and the sparse-neighborhood gate are tested separately.
        padding = [] if total is None else [_both_good()] * max(0, total - len(records))
        return [*records, *padding]

    async def _prices(pool: list[str], *, workspace_id: uuid.UUID | None = None) -> dict[str, float]:
        return prices or dict.fromkeys(pool, 1.0)

    backend._embed = _embed  # type: ignore[method-assign]
    backend._load_records = _load  # type: ignore[method-assign]
    backend._candidate_prices = _prices  # type: ignore[method-assign]


# -- decline paths ----------------------------------------------------------


@pytest.mark.asyncio
async def test_single_candidate_pool_declines() -> None:
    backend = _backend()
    _wire(backend, [])
    decision = await backend.rank(_ctx(candidates=(STRONG,)))
    assert decision.ordered_models == []
    assert "nothing to route among" in decision.rationale


@pytest.mark.asyncio
async def test_tools_are_capability_gated_to_the_default() -> None:
    backend = _backend()
    _wire(backend, [_both_good(), _both_good()])
    decision = await backend.rank(_ctx(has_tools=True))
    assert decision.ordered_models == []
    assert "tools" in decision.rationale


@pytest.mark.asyncio
async def test_cold_pool_below_seed_count_declines() -> None:
    backend = _backend(router_seed_count=10)
    _wire(backend, [_both_good(), _both_good()])
    decision = await backend.rank(_ctx())
    assert decision.ordered_models == []
    assert "cold pool" in decision.rationale


@pytest.mark.asyncio
async def test_cold_pool_names_the_task_partition() -> None:
    # The partition is in the message because "cold" is per pool: an operator who
    # taught the default pool needs to know this task's own partition is empty.
    backend = _backend(router_seed_count=10)
    _wire(backend, [])
    decision = await backend.rank(_ctx(task_id="summarize"))
    assert "task 'summarize'" in decision.rationale


@pytest.mark.asyncio
async def test_sparse_neighborhood_declines() -> None:
    # The seed gate is satisfied but there are fewer than k comparable records.
    backend = _backend(router_k=5, router_seed_count=1)
    _wire(backend, [_both_good()])
    decision = await backend.rank(_ctx())
    assert decision.ordered_models == []
    assert "sparse" in decision.rationale


@pytest.mark.asyncio
async def test_embedding_failure_declines() -> None:
    backend = _backend()
    _wire(backend, [_both_good(), _both_good()])

    async def _boom(text: str) -> list[float]:
        raise RuntimeError("embedding down")

    backend._embed = _boom  # type: ignore[method-assign]
    decision = await backend.rank(_ctx())
    assert decision.ordered_models == []
    assert "embedding error" in decision.rationale


@pytest.mark.asyncio
async def test_unpriced_candidate_declines_rather_than_failing() -> None:
    # Pricing is validated when a policy is written, so this is the "pricing was
    # deleted under a running gateway" case. It must not fail the request.
    from gateway.services.routing.knn import RouterPricingError

    backend = _backend()
    _wire(backend, [_both_good(), _both_good()])

    async def _prices(pool: list[str], *, workspace_id: uuid.UUID | None = None) -> dict[str, float]:
        raise RouterPricingError(f"Router candidate '{CHEAP}' has no configured pricing.")

    backend._candidate_prices = _prices  # type: ignore[method-assign]
    decision = await backend.rank(_ctx())
    assert decision.ordered_models == []
    assert "no configured pricing" in decision.rationale


# -- the core routing decision ---------------------------------------------


@pytest.mark.asyncio
async def test_easy_region_routes_to_cheap() -> None:
    # Neighbors say cheap is as good as strong here; cost bias picks cheap.
    backend = _backend(router_alpha=0.3)
    _wire(backend, [_both_good(), _both_good()], prices=PRICES)
    decision = await backend.rank(_ctx())
    assert decision.ordered_models[0] == CHEAP
    # The policy default is always the cascade's last resort.
    assert decision.ordered_models[-1] == STRONG


@pytest.mark.asyncio
async def test_hard_region_routes_to_strong() -> None:
    # Neighbors say cheap fails here (quality 0); strong wins despite cost.
    backend = _backend(router_alpha=0.3)
    _wire(backend, [_cheap_fails(), _cheap_fails()], prices=PRICES)
    decision = await backend.rank(_ctx())
    assert decision.ordered_models[0] == STRONG


@pytest.mark.asyncio
async def test_a_record_keyed_in_another_spelling_still_scores_its_candidate() -> None:
    """Candidates and stored scores match on model identity, not on spelling.

    `openai/gpt-3.5-turbo` and `openai:gpt-3.5-turbo` name one model, and a record
    can hold either: written before keys were canonicalized, or taught through a
    second learned policy that spells the pool the other way. A miss is the silent
    failure, the cheap candidate scores nothing and the strong model wins unopposed
    at confidence 1.0, on a pool that reports warm.
    """
    backend = _backend(router_alpha=0.3)
    colon = {CHEAP.replace("/", ":"): 1.0, STRONG.replace("/", ":"): 1.0}
    _wire(backend, [_mem(colon), _mem(colon)], prices=PRICES)

    decision = await backend.rank(_ctx())

    assert decision.ordered_models[0] == CHEAP
    assert decision.confidence == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_one_record_holding_both_spellings_of_a_model_keeps_the_first() -> None:
    # `/rank` refuses this, so it can only reach the store from before it did. The
    # two scores cannot both apply, so the rule is simply deterministic: the same
    # score wins on every request rather than by dict ordering luck.
    backend = _backend(router_alpha=0.3)
    both = {CHEAP: 1.0, CHEAP.replace("/", ":"): 0.0, STRONG: 1.0}
    _wire(backend, [_mem(both), _mem(both)], prices=PRICES)

    decision = await backend.rank(_ctx())

    assert decision.ordered_models[0] == CHEAP


@pytest.mark.asyncio
async def test_higher_alpha_pushes_toward_cheap() -> None:
    # A large enough cost dial overrides a modest quality gap, proving the dial
    # actually moves the operating point.
    records = [_mem({CHEAP: 0.6, STRONG: 1.0}), _mem({CHEAP: 0.6, STRONG: 1.0})]
    low = _backend(router_alpha=0.0)
    high = _backend(router_alpha=5.0)
    _wire(low, records, prices=PRICES)
    _wire(high, list(records), prices=PRICES)
    assert (await low.rank(_ctx())).ordered_models[0] == STRONG
    assert (await high.rank(_ctx())).ordered_models[0] == CHEAP


# Neighborhoods where cost bias picks cheap, but the neighbor prompts back it to
# differing degrees: 3-of-4 prefer cheap (dense) vs 1-of-4 (thin).
_DENSE = [
    _mem({CHEAP: 0.9, STRONG: 0.5}),
    _mem({CHEAP: 0.9, STRONG: 0.5}),
    _mem({CHEAP: 0.9, STRONG: 0.5}),
    _mem({CHEAP: 0.5, STRONG: 0.9}),
]
_THIN = [
    _mem({CHEAP: 0.9, STRONG: 0.5}),
    _mem({CHEAP: 0.7, STRONG: 0.75}),
    _mem({CHEAP: 0.7, STRONG: 0.75}),
    _mem({CHEAP: 0.7, STRONG: 0.75}),
]


@pytest.mark.asyncio
async def test_confidence_is_local_support_for_the_pick() -> None:
    # Confidence is the share of the k neighbor prompts whose own best-scoring
    # model is the chosen one, so a densely-supported cheap pick reads high and a
    # thinly-supported one reads low even though both pick cheap on cost.
    dense = _backend(router_alpha=0.3, router_k=4)
    _wire(dense, list(_DENSE), prices=PRICES)
    decision = await dense.rank(_ctx())
    assert decision.ordered_models[0] == CHEAP
    assert decision.confidence == pytest.approx(0.75)

    thin = _backend(router_alpha=0.3, router_k=4)
    _wire(thin, list(_THIN), prices=PRICES)
    decision = await thin.rank(_ctx())
    assert decision.ordered_models[0] == CHEAP
    assert decision.confidence == pytest.approx(0.25)


@pytest.mark.asyncio
async def test_confidence_ignores_non_candidate_favorites() -> None:
    # Neighbors prefer a model that is not in the policy's pool. Confidence should
    # reflect support among the candidates (here every neighbor's best candidate is
    # the chosen one), not collapse because the global favorite is unavailable,
    # which would let a confidence floor veto a correct pick.
    other = "openai/gpt-5.4"
    backend = _backend(router_alpha=0.3, router_k=3, router_confidence_floor=0.5)
    records = [_mem({CHEAP: 0.8, STRONG: 0.6, other: 1.0})] * 3  # `other` best, but not a candidate
    _wire(backend, records, prices=PRICES)
    decision = await backend.rank(_ctx())  # pool is (CHEAP, STRONG)
    assert decision.ordered_models[0] == CHEAP
    assert decision.confidence == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_confidence_floor_vetoes_thinly_supported_pick_only() -> None:
    # A moderate floor keeps a well-supported cheap pick (0.75 >= 0.5) but vetoes a
    # thinly-supported one (0.25 < 0.5) back to the default target. This is the
    # behavior the docs promise: the floor gates on real local support, so it does
    # not silently disable routing on every cost-saving downgrade.
    kept = _backend(router_alpha=0.3, router_k=4, router_confidence_floor=0.5)
    _wire(kept, list(_DENSE), prices=PRICES)
    assert (await kept.rank(_ctx())).ordered_models[0] == CHEAP

    vetoed = _backend(router_alpha=0.3, router_k=4, router_confidence_floor=0.5)
    _wire(vetoed, list(_THIN), prices=PRICES)
    decision = await vetoed.rank(_ctx())
    assert decision.ordered_models[0] == STRONG
    assert "below floor" in decision.rationale
    # Vetoed, not declined: the ranking still orders the rest of the pool, so a
    # failover after the default target follows the router's preference.
    assert decision.ordered_models == [STRONG, CHEAP]


# -- trace stickiness -------------------------------------------------------


@pytest.mark.asyncio
async def test_trace_sticky_reuses_first_decision() -> None:
    backend = _backend(router_alpha=0.3, router_granularity="trace_sticky")
    _wire(backend, [_both_good(), _both_good()], prices=PRICES)

    convo: list[dict[str, Any]] = [{"role": "user", "content": "start the trace"}]
    first = await backend.rank(_ctx(messages=convo))
    assert first.ordered_models[0] == CHEAP

    # A continuation of the same trace (same opening turn) must reuse the original
    # pick without re-routing, even if neighbors now disagree.
    _wire(backend, [_cheap_fails(), _cheap_fails()], prices=PRICES)
    cont = [*convo, {"role": "assistant", "content": "..."}, {"role": "user", "content": "next step"}]
    second = await backend.rank(_ctx(messages=cont, is_trace_continuation=True))
    assert second.ordered_models[0] == CHEAP
    assert "trace-sticky" in second.rationale


@pytest.mark.asyncio
async def test_step_granularity_reroutes_each_call() -> None:
    backend = _backend(router_alpha=0.3, router_granularity="step")
    convo: list[dict[str, Any]] = [{"role": "user", "content": "q"}]
    _wire(backend, [_both_good(), _both_good()], prices=PRICES)
    assert (await backend.rank(_ctx(messages=convo))).ordered_models[0] == CHEAP
    # New neighbor signal flips the decision because step mode does not stick.
    _wire(backend, [_cheap_fails(), _cheap_fails()], prices=PRICES)
    cont = [*convo, {"role": "assistant", "content": "a"}, {"role": "user", "content": "q2"}]
    assert (await backend.rank(_ctx(messages=cont, is_trace_continuation=True))).ordered_models[0] == STRONG


def _capture_embed_signal(backend: KnnRoutingMemory, captured: dict[str, str]) -> None:
    _wire(backend, [_both_good(), _both_good()], prices=PRICES)

    async def _embed(text: str) -> list[float]:
        captured["signal"] = text
        return [1.0, 0.0]

    backend._embed = _embed  # type: ignore[method-assign]


@pytest.mark.asyncio
async def test_embedded_signal_is_latest_turn_in_step_mode_and_opener_in_sticky() -> None:
    # Which signal the router embeds depends on granularity, not on whether the
    # request is a continuation: step routes on the current turn; trace_sticky
    # anchors on the conversation's opener so a cache miss reproduces turn one.
    convo: list[dict[str, Any]] = [
        {"role": "user", "content": "OPENER"},
        {"role": "assistant", "content": "a"},
        {"role": "user", "content": "LATEST"},
    ]

    step = _backend(router_granularity="step")
    step_captured: dict[str, str] = {}
    _capture_embed_signal(step, step_captured)
    await step.rank(_ctx(messages=convo, is_trace_continuation=True))
    assert step_captured["signal"] == "LATEST"

    sticky = _backend(router_granularity="trace_sticky")
    sticky_captured: dict[str, str] = {}
    _capture_embed_signal(sticky, sticky_captured)
    await sticky.rank(_ctx(messages=convo, is_trace_continuation=True))
    assert sticky_captured["signal"] == "OPENER"


@pytest.mark.asyncio
async def test_conversation_id_makes_stickiness_robust_to_content() -> None:
    # With an explicit conversation id, a continuation reuses the first decision
    # even though its message content is completely different (a real conversation
    # id, not a content hash, is the trace identity).
    backend = _backend(router_alpha=0.3, router_granularity="trace_sticky")
    _wire(backend, [_both_good(), _both_good()], prices=PRICES)
    first = await backend.rank(_ctx(messages=[{"role": "user", "content": "A"}], trace_key="conv-1"))
    assert first.ordered_models[0] == CHEAP

    _wire(backend, [_cheap_fails(), _cheap_fails()], prices=PRICES)
    different: list[dict[str, Any]] = [{"role": "user", "content": "ENTIRELY DIFFERENT"}]
    cont = await backend.rank(_ctx(messages=different, trace_key="conv-1", is_trace_continuation=True))
    assert cont.ordered_models[0] == CHEAP
    assert "trace-sticky" in cont.rationale


@pytest.mark.asyncio
async def test_conversation_id_is_namespaced_per_user() -> None:
    # The same conversation id from two users must not share a decision.
    backend = _backend(router_alpha=0.3, router_granularity="trace_sticky")

    _wire(backend, [_both_good(), _both_good()], prices=PRICES)
    first = await backend.rank(_ctx(user_id="user-a", trace_key="conv-1"))
    assert first.ordered_models[0] == CHEAP

    # User B shares the conversation id but has no decision under its own
    # namespace, so it routes fresh on its own neighbors (here cheap fails).
    _wire(backend, [_cheap_fails(), _cheap_fails()], prices=PRICES)
    decision = await backend.rank(_ctx(user_id="user-b", trace_key="conv-1", is_trace_continuation=True))
    assert decision.ordered_models[0] == STRONG
    assert "trace-sticky" not in decision.rationale


@pytest.mark.asyncio
async def test_conversation_id_is_namespaced_per_workspace() -> None:
    """The same user, the same conversation id, two workspaces: no shared decision.

    The trace cache is consulted before any record loads, so a key that omitted
    the workspace would replay workspace A's pick in workspace B without ever
    reading workspace B's examples. That is the cross-workspace leak the rest of
    the scoping closes at the query, and this closes in process memory.
    """
    workspace_a = uuid.UUID("00000000-0000-4000-8000-00000000000a")
    workspace_b = uuid.UUID("00000000-0000-4000-8000-00000000000b")
    backend = _backend(router_alpha=0.3, router_granularity="trace_sticky")

    _wire(backend, [_both_good(), _both_good()], prices=PRICES)
    first = await backend.rank(_ctx(user_id="ada", workspace_id=workspace_a, trace_key="conv-1"))
    assert first.ordered_models[0] == CHEAP

    # Workspace B has no decision under its own namespace, so it routes fresh on
    # its own neighbors (here cheap fails) rather than reusing A's.
    _wire(backend, [_cheap_fails(), _cheap_fails()], prices=PRICES)
    decision = await backend.rank(
        _ctx(user_id="ada", workspace_id=workspace_b, trace_key="conv-1", is_trace_continuation=True)
    )
    assert decision.ordered_models[0] == STRONG
    assert "trace-sticky" not in decision.rationale

    # And workspace A still has its own, so the namespacing partitions rather
    # than simply defeating stickiness.
    _wire(backend, [_cheap_fails(), _cheap_fails()], prices=PRICES)
    reused = await backend.rank(
        _ctx(user_id="ada", workspace_id=workspace_a, trace_key="conv-1", is_trace_continuation=True)
    )
    assert reused.ordered_models[0] == CHEAP
    assert "trace-sticky" in reused.rationale


@pytest.mark.asyncio
async def test_distinct_system_preamble_separates_traces_without_a_conversation_id() -> None:
    # Without a conversation id, the opener anchor includes the system turn, so two
    # conversations that share a first user message but differ in system preamble
    # are kept apart (first-user-text alone would have collided them).
    backend = _backend(router_alpha=0.3, router_granularity="trace_sticky")
    _wire(backend, [_both_good(), _both_good()], prices=PRICES)
    convo_a: list[dict[str, Any]] = [{"role": "system", "content": "agent A"}, {"role": "user", "content": "go"}]
    assert (await backend.rank(_ctx(messages=convo_a))).ordered_models[0] == CHEAP

    # Different system preamble, same first user turn, marked as a continuation:
    # it must NOT reuse conversation A's decision (no shared trace key).
    _wire(backend, [_cheap_fails(), _cheap_fails()], prices=PRICES)
    convo_b: list[dict[str, Any]] = [
        {"role": "system", "content": "agent B"},
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "..."},
        {"role": "user", "content": "next"},
    ]
    decision = await backend.rank(_ctx(messages=convo_b, is_trace_continuation=True))
    assert decision.ordered_models[0] == STRONG
    assert "trace-sticky" not in decision.rationale


@pytest.mark.asyncio
async def test_sticky_decision_is_dropped_when_it_leaves_the_pool() -> None:
    # A remembered pick that the policy no longer offers (edited candidates, or a
    # caller whose allow-list narrowed) must be re-decided rather than returned.
    backend = _backend(router_alpha=0.3, router_granularity="trace_sticky")
    _wire(backend, [_both_good(), _both_good()], prices=PRICES)
    assert (await backend.rank(_ctx(trace_key="conv-1"))).ordered_models[0] == CHEAP

    narrowed = await backend.rank(
        _ctx(candidates=(STRONG, "openai/gpt-4o-mini"), trace_key="conv-1", is_trace_continuation=True)
    )
    assert narrowed.ordered_models[0] != CHEAP
    assert "trace-sticky" not in narrowed.rationale


# -- store bounds -----------------------------------------------------------


@pytest.mark.asyncio
async def test_the_record_read_is_bounded_and_takes_the_newest() -> None:
    """One decision may not load an unbounded number of rows.

    Eviction is enforced lazily on write and only over a user's whole set, so
    nothing else stops a partition from growing past the cap; without a limit here
    a single request would load and cosine-score every row it found.
    """
    backend = _backend(router_max_records_per_user=25)
    assert backend._read_limit == 25
    # Eviction off is not a license for an unbounded read.
    assert _backend(router_max_records_per_user=0)._read_limit == 5000


# -- ordering ---------------------------------------------------------------


def test_ordered_with_fallthrough_appends_the_default_last() -> None:
    backend = _backend()
    # The chosen model leads; the default (missing from the ranked list) is
    # appended as the tail safety net.
    assert backend._ordered_with_fallthrough([CHEAP], STRONG, [CHEAP, STRONG]) == [CHEAP, STRONG]
    # Chosen != default: the default is demoted to the cascade's final position.
    assert backend._ordered_with_fallthrough([CHEAP, STRONG], STRONG, [CHEAP, STRONG]) == [CHEAP, STRONG]
    # Chosen == default: it must NOT be demoted; it stays first.
    assert backend._ordered_with_fallthrough([STRONG, CHEAP], STRONG, [CHEAP, STRONG]) == [STRONG, CHEAP]


# ---------------------------------------------------------------------------
# Workspace-scoped aliases (otari-ai#1643)
#
# A candidate selector can name an alias, and this PR is what makes an alias
# point somewhere different per workspace. Every reader of one inside a decision
# therefore has to resolve it in the *request's* workspace: the pool filter, the
# score keys and the price all key on the resolved target, so a reader that used
# the default workspace instead would disagree with the others and the candidate
# would drop out of scoring with no score and no error.
# ---------------------------------------------------------------------------

_WORKSPACE = uuid.UUID("00000000-0000-4000-8000-0000000000ab")
_DEFAULT_WORKSPACE = uuid.UUID("00000000-0000-4000-8000-00000000d3fa")
_ALIAS = "fast-alias"


@pytest.fixture
def _aliased_workspaces() -> Iterator[None]:
    """``fast-alias`` points at the cheap model in one workspace, the strong one by default."""
    alias_service.reset_alias_cache()
    alias_service._cache[_WORKSPACE] = {_ALIAS: CHEAP}
    alias_service._cache[_DEFAULT_WORKSPACE] = {_ALIAS: STRONG}
    alias_service._default_workspace = _DEFAULT_WORKSPACE
    alias_service._cached_at = time.monotonic()
    yield
    alias_service.reset_alias_cache()


def test_scoring_canonicalizes_an_alias_in_the_requests_workspace(_aliased_workspaces: None) -> None:
    """The score-key half. Resolving in the default workspace names the wrong model."""
    backend = _backend()

    assert backend._canonical(_ALIAS, "u", {}, workspace_id=_WORKSPACE) == "openai:gpt-3.5-turbo"
    # The same name, the deployment-wide reading: a different model entirely.
    assert backend._canonical(_ALIAS, "u", {}, workspace_id=None) == "openai:gpt-4o"


@pytest.mark.asyncio
async def test_pricing_follows_the_same_workspaces_alias_as_scoring(
    _aliased_workspaces: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The price half, which has to name the same model the score key does.

    Pricing the default workspace's target while scoring this workspace's would
    rank the candidate on a price the request never pays. Separate from *whose*
    rates apply, which stays the deployment list either way.
    """
    asked: list[tuple[str, str]] = []

    async def _pricing(db: Any, instance: str, model: str) -> Any:
        asked.append((instance, model))
        return SimpleNamespace(input_price_per_million=Decimal("1"))

    monkeypatch.setattr(knn, "find_model_pricing", _pricing)
    backend = _backend()

    await backend._input_price(cast(Any, object()), _ALIAS, workspace_id=_WORKSPACE)
    await backend._input_price(cast(Any, object()), _ALIAS, workspace_id=None)

    assert asked == [("openai", "gpt-3.5-turbo"), ("openai", "gpt-4o")]


@pytest.mark.asyncio
async def test_an_aliased_candidate_matches_the_scores_taught_in_its_workspace(
    _aliased_workspaces: None,
) -> None:
    """End to end: the decline this bug produced, and the routing it should produce.

    ``/rank`` canonicalizes a stored score key in the workspace it was taught in,
    so these records are keyed on the cheap model alone, which is what teaching
    ``fast-alias`` in this workspace produces. Resolving the candidate in the
    default workspace instead looks up ``openai:gpt-4o``, which no record scores,
    so every candidate is skipped and the router declines while the pool reports
    warm. That silent decline is the whole failure mode.
    """
    backend = _backend(router_alpha=0.0, router_confidence_floor=0.0)
    records = [_mem({"openai:gpt-3.5-turbo": 1.0}), _mem({"openai:gpt-3.5-turbo": 1.0})]
    _wire(backend, records, prices={_ALIAS: 1.0, STRONG: 10.0})

    decision = await backend.rank(
        _ctx(candidates=(_ALIAS, STRONG), default=STRONG, workspace_id=_WORKSPACE)
    )

    assert decision.ordered_models[0] == _ALIAS
    assert "no neighbor scored any candidate" not in decision.rationale
