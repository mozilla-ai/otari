"""Unit tests for the kNN routing-memory algorithm.

These isolate the scoring/ordering/trace logic from the database and the
embedding provider by patching the backend's async I/O helpers (``_embed``,
``_load_records``, ``_candidate_prices``). What is exercised here is the part
that decides *which* model wins: cost-biased kNN voting, cold-start and
sparse-neighborhood pass-through, the confidence floor, trace-sticky reuse, the
tools capability gate, and requested-model fallthrough ordering.

Each routing-memory record is one example: a prompt embedding plus a
``{model: quality}`` map, so the kNN votes over distinct prompts.
"""

from __future__ import annotations

from typing import Any

import pytest

from gateway.core.config import GatewayConfig
from gateway.models.entities import RoutingMemory
from gateway.services.knn_router import KnnRoutingMemory
from gateway.services.router_backend import RoutingContext

CHEAP = "openai/gpt-3.5-turbo"
STRONG = "openai/gpt-4o"

PRICES = {CHEAP: 1.0, STRONG: 10.0}


def _backend(**overrides: Any) -> KnnRoutingMemory:
    kwargs: dict[str, Any] = {"router_backend": "knn", "router_k": 2, "router_seed_count": 2}
    kwargs.update(overrides)
    return KnnRoutingMemory(GatewayConfig(**kwargs))


def _mem(qualities: dict[str, float], vec: tuple[float, ...] = (1.0, 0.0)) -> RoutingMemory:
    """One example: a prompt embedding plus each model's quality on it."""
    return RoutingMemory(tenant_id="t", embedding_model="m", embedding=list(vec), qualities=dict(qualities))


def _both_good() -> RoutingMemory:
    return _mem({CHEAP: 1.0, STRONG: 1.0})


def _cheap_fails() -> RoutingMemory:
    return _mem({CHEAP: 0.0, STRONG: 1.0})


def _ctx(
    *,
    candidates: tuple[str, ...] = (CHEAP, STRONG),
    requested: str = STRONG,
    messages: list[dict[str, Any]] | None = None,
    **kw: Any,
) -> RoutingContext:
    return RoutingContext(
        tenant_id="t",
        messages=messages or [{"role": "user", "content": "hello"}],
        requested_model=requested,
        candidate_pool=list(candidates),
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

    async def _load(tenant_id: str, task_id: str | None) -> tuple[list[RoutingMemory], int]:
        return records, total if total is not None else len(records)

    async def _prices(pool: list[str]) -> dict[str, float]:
        return prices or dict.fromkeys(pool, 1.0)

    backend._embed = _embed  # type: ignore[method-assign]
    backend._load_records = _load  # type: ignore[method-assign]
    backend._candidate_prices = _prices  # type: ignore[method-assign]


# -- pass-through guards ----------------------------------------------------


@pytest.mark.asyncio
async def test_single_candidate_pool_passes_through() -> None:
    backend = _backend()
    _wire(backend, [])
    decision = await backend.route(_ctx(candidates=(STRONG,)))
    assert decision.ordered_models == [STRONG]
    assert "single-candidate" in decision.rationale


@pytest.mark.asyncio
async def test_tools_are_capability_gated_to_requested_model() -> None:
    backend = _backend()
    _wire(backend, [_both_good(), _both_good()])
    decision = await backend.route(_ctx(has_tools=True))
    assert decision.ordered_models == [STRONG]
    assert "tools" in decision.rationale


@pytest.mark.asyncio
async def test_cold_tenant_below_seed_count_passes_through() -> None:
    backend = _backend(router_seed_count=10)
    _wire(backend, [_both_good(), _both_good()], total=2)
    decision = await backend.route(_ctx())
    assert decision.ordered_models == [STRONG]
    assert "cold tenant" in decision.rationale


@pytest.mark.asyncio
async def test_sparse_neighborhood_passes_through() -> None:
    # seed gate satisfied (total high) but fewer than k usable neighbor records.
    backend = _backend(router_k=5)
    _wire(backend, [_both_good()], total=50)
    decision = await backend.route(_ctx())
    assert decision.ordered_models == [STRONG]
    assert "sparse" in decision.rationale


@pytest.mark.asyncio
async def test_embedding_failure_passes_through() -> None:
    backend = _backend()
    _wire(backend, [_both_good(), _both_good()])

    async def _boom(text: str) -> list[float]:
        raise RuntimeError("embedding down")

    backend._embed = _boom  # type: ignore[method-assign]
    decision = await backend.route(_ctx())
    assert decision.ordered_models == [STRONG]
    assert "embedding error" in decision.rationale


# -- the core routing decision ---------------------------------------------


@pytest.mark.asyncio
async def test_easy_region_routes_to_cheap() -> None:
    # Neighbors say cheap is as good as strong here; cost bias picks cheap.
    backend = _backend(router_alpha=0.3)
    _wire(backend, [_both_good(), _both_good()], prices=PRICES)
    decision = await backend.route(_ctx())
    assert decision.ordered_models[0] == CHEAP
    # requested (strong) is always the final fallthrough.
    assert decision.ordered_models[-1] == STRONG


@pytest.mark.asyncio
async def test_hard_region_routes_to_strong() -> None:
    # Neighbors say cheap fails here (quality 0); strong wins despite cost.
    backend = _backend(router_alpha=0.3)
    _wire(backend, [_cheap_fails(), _cheap_fails()], prices=PRICES)
    decision = await backend.route(_ctx())
    assert decision.ordered_models[0] == STRONG


@pytest.mark.asyncio
async def test_higher_alpha_pushes_toward_cheap() -> None:
    # A large enough cost dial overrides a modest quality gap, proving the dial
    # actually moves the operating point.
    records = [_mem({CHEAP: 0.6, STRONG: 1.0}), _mem({CHEAP: 0.6, STRONG: 1.0})]
    low = _backend(router_alpha=0.0)
    high = _backend(router_alpha=5.0)
    _wire(low, records, prices=PRICES)
    _wire(high, list(records), prices=PRICES)
    assert (await low.route(_ctx())).ordered_models[0] == STRONG
    assert (await high.route(_ctx())).ordered_models[0] == CHEAP


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
    decision = await dense.route(_ctx())
    assert decision.ordered_models[0] == CHEAP
    assert decision.confidence == pytest.approx(0.75)

    thin = _backend(router_alpha=0.3, router_k=4)
    _wire(thin, list(_THIN), prices=PRICES)
    decision = await thin.route(_ctx())
    assert decision.ordered_models[0] == CHEAP
    assert decision.confidence == pytest.approx(0.25)


@pytest.mark.asyncio
async def test_confidence_ignores_non_candidate_favorites() -> None:
    # Neighbors prefer a model that is not in the candidate pool. Confidence
    # should reflect support among CANDIDATES (here every neighbor's best
    # candidate is the chosen one), not collapse because the global favorite is
    # unavailable, which would let a confidence floor veto a correct pick.
    OTHER = "openai/gpt-5.4"
    backend = _backend(router_alpha=0.3, router_k=3, router_confidence_floor=0.5)
    recs = [_mem({CHEAP: 0.8, STRONG: 0.6, OTHER: 1.0})] * 3  # OTHER best, but not a candidate
    _wire(backend, recs, prices=PRICES)
    decision = await backend.route(_ctx())  # pool is (CHEAP, STRONG)
    assert decision.ordered_models[0] == CHEAP
    assert decision.confidence == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_confidence_floor_vetoes_thinly_supported_pick_only() -> None:
    # A moderate floor keeps a well-supported cheap pick (0.75 >= 0.5) but vetoes
    # a thinly-supported one (0.25 < 0.5) back to the requested model. This is the
    # behavior the docs promise: the floor gates on real local support, so it does
    # not silently disable routing on every cost-saving downgrade.
    kept = _backend(router_alpha=0.3, router_k=4, router_confidence_floor=0.5)
    _wire(kept, list(_DENSE), prices=PRICES)
    assert (await kept.route(_ctx())).ordered_models[0] == CHEAP

    vetoed = _backend(router_alpha=0.3, router_k=4, router_confidence_floor=0.5)
    _wire(vetoed, list(_THIN), prices=PRICES)
    decision = await vetoed.route(_ctx())
    assert decision.ordered_models[0] == STRONG
    assert "below floor" in decision.rationale


# -- trace stickiness -------------------------------------------------------


@pytest.mark.asyncio
async def test_trace_sticky_reuses_first_decision() -> None:
    backend = _backend(router_alpha=0.3, router_granularity="trace_sticky")
    _wire(backend, [_both_good(), _both_good()], prices=PRICES)

    convo: list[dict[str, Any]] = [{"role": "user", "content": "start the trace"}]
    first = await backend.route(_ctx(messages=convo))
    assert first.ordered_models[0] == CHEAP

    # A continuation of the same trace (same first user turn) must reuse the
    # original pick without re-routing, even if neighbors now disagree.
    _wire(backend, [_cheap_fails(), _cheap_fails()], prices=PRICES)
    cont = convo + [{"role": "assistant", "content": "..."}, {"role": "user", "content": "next step"}]
    second = await backend.route(_ctx(messages=cont, is_trace_continuation=True))
    assert second.ordered_models[0] == CHEAP
    assert "trace-sticky" in second.rationale


@pytest.mark.asyncio
async def test_step_granularity_reroutes_each_call() -> None:
    backend = _backend(router_alpha=0.3, router_granularity="step")
    convo: list[dict[str, Any]] = [{"role": "user", "content": "q"}]
    _wire(backend, [_both_good(), _both_good()], prices=PRICES)
    assert (await backend.route(_ctx(messages=convo))).ordered_models[0] == CHEAP
    # New neighbor signal flips the decision because step mode does not stick.
    _wire(backend, [_cheap_fails(), _cheap_fails()], prices=PRICES)
    cont = convo + [{"role": "assistant", "content": "a"}, {"role": "user", "content": "q2"}]
    assert (await backend.route(_ctx(messages=cont, is_trace_continuation=True))).ordered_models[0] == STRONG


def _capture_embed_signal(backend: KnnRoutingMemory, captured: dict[str, str]) -> None:
    _wire(backend, [_both_good(), _both_good()], prices=PRICES)

    async def _embed(text: str) -> list[float]:
        captured["signal"] = text
        return [1.0, 0.0]

    backend._embed = _embed  # type: ignore[method-assign]


@pytest.mark.asyncio
async def test_embedding_signal_is_latest_turn_in_step_mode_and_opener_in_sticky() -> None:
    # The signal the router embeds depends on granularity, not on whether the
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
    await step.route(_ctx(messages=convo, is_trace_continuation=True))
    assert step_captured["signal"] == "LATEST"

    sticky = _backend(router_granularity="trace_sticky")
    sticky_captured: dict[str, str] = {}
    _capture_embed_signal(sticky, sticky_captured)
    await sticky.route(_ctx(messages=convo, is_trace_continuation=True))
    assert sticky_captured["signal"] == "OPENER"


@pytest.mark.asyncio
async def test_conversation_id_makes_stickiness_robust_to_content() -> None:
    # With an explicit conversation id, a continuation reuses the first decision
    # even though its message content is completely different (a real conversation
    # id, not a content hash, is the trace identity).
    backend = _backend(router_alpha=0.3, router_granularity="trace_sticky")
    _wire(backend, [_both_good(), _both_good()], prices=PRICES)
    first = await backend.route(_ctx(messages=[{"role": "user", "content": "A"}], trace_key="conv-1"))
    assert first.ordered_models[0] == CHEAP

    _wire(backend, [_cheap_fails(), _cheap_fails()], prices=PRICES)
    different: list[dict[str, Any]] = [{"role": "user", "content": "ENTIRELY DIFFERENT"}]
    cont = await backend.route(_ctx(messages=different, trace_key="conv-1", is_trace_continuation=True))
    assert cont.ordered_models[0] == CHEAP
    assert "trace-sticky" in cont.rationale


@pytest.mark.asyncio
async def test_conversation_id_is_namespaced_per_tenant() -> None:
    # The same conversation id from two tenants must not share a decision.
    backend = _backend(router_alpha=0.3, router_granularity="trace_sticky")

    def _ctx_for(tenant: str, *, cont: bool) -> RoutingContext:
        return RoutingContext(
            tenant_id=tenant,
            messages=[{"role": "user", "content": "x"}],
            requested_model=STRONG,
            candidate_pool=[CHEAP, STRONG],
            trace_key="conv-1",
            is_trace_continuation=cont,
        )

    _wire(backend, [_both_good(), _both_good()], prices=PRICES)
    assert (await backend.route(_ctx_for("tenant-a", cont=False))).ordered_models[0] == CHEAP

    # Tenant B shares the conversation id but has no decision under its own
    # namespace, so it routes fresh on its own neighbors (here cheap fails).
    _wire(backend, [_cheap_fails(), _cheap_fails()], prices=PRICES)
    decision = await backend.route(_ctx_for("tenant-b", cont=True))
    assert decision.ordered_models[0] == STRONG
    assert "trace-sticky" not in decision.rationale


@pytest.mark.asyncio
async def test_distinct_system_preamble_separates_traces_without_a_conversation_id() -> None:
    # Without a conversation id, the opener anchor includes the system turn, so two
    # conversations that share a first user message but differ in system preamble
    # are kept apart (first-user-text alone would have collided them).
    backend = _backend(router_alpha=0.3, router_granularity="trace_sticky")
    _wire(backend, [_both_good(), _both_good()], prices=PRICES)
    convo_a: list[dict[str, Any]] = [{"role": "system", "content": "agent A"}, {"role": "user", "content": "go"}]
    assert (await backend.route(_ctx(messages=convo_a))).ordered_models[0] == CHEAP

    # Different system preamble, same first user turn, marked as a continuation:
    # it must NOT reuse conversation A's decision (no shared trace key).
    _wire(backend, [_cheap_fails(), _cheap_fails()], prices=PRICES)
    convo_b: list[dict[str, Any]] = [
        {"role": "system", "content": "agent B"},
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "..."},
        {"role": "user", "content": "next"},
    ]
    decision = await backend.route(_ctx(messages=convo_b, is_trace_continuation=True))
    assert decision.ordered_models[0] == STRONG
    assert "trace-sticky" not in decision.rationale


# -- ordering ---------------------------------------------------------------


def test_order_with_fallthrough_appends_requested_last() -> None:
    backend = _backend()
    # Chosen model leads; requested (missing from the ranked list) is appended
    # as the tail safety net.
    assert backend._order_with_fallthrough([CHEAP], STRONG, [CHEAP, STRONG]) == [CHEAP, STRONG]
    # Chosen != requested: requested is demoted to the cascade's final position.
    assert backend._order_with_fallthrough([CHEAP, STRONG], STRONG, [CHEAP, STRONG]) == [CHEAP, STRONG]
    # Chosen == requested: it must NOT be demoted; it stays first.
    assert backend._order_with_fallthrough([STRONG, CHEAP], STRONG, [CHEAP, STRONG]) == [STRONG, CHEAP]
