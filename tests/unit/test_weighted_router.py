"""Unit tests for the weighted router: the schema rules for a traffic split, the
draw itself, and the synchronous view ``explain`` renders from it.

Three things are worth pinning down here, because each is a claim the feature
makes that a plausible implementation would break:

* A split is *normalized*, not percentages, and a candidate omitted from
  ``weights`` is drained rather than defaulted, so a drained provider stays in the
  plan as a failover target.
* The draw is without replacement over the whole pool, so the ordering a failure
  walks is itself weighted, and a zero-weight candidate can never win the head
  (including at the floating-point boundary).
* Shares are normalized over the candidates the *caller* may use, so a policy
  whose heavy candidate is filtered out reports the split that is really running.
"""

from __future__ import annotations

import asyncio
import random
from collections import Counter

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from gateway.core.config import GatewayConfig
from gateway.models.routing import PolicySpec, RoutingConfig
from gateway.services.routing.backends import (
    WEIGHTED_BACKEND,
    RoutingContext,
    backend_is_weighted,
    backend_pool_is_teachable,
    backend_requires_pricing,
    get_router_backend,
)
from gateway.services.routing.compiler import compile_policy
from gateway.services.routing.decide import RoutingSignal, decide_ordering, explain_router_ordering
from gateway.services.routing.weighted import (
    WeightedRouterBackend,
    declared_shares,
    describe_split,
    explain_ordering,
    weighted_ordering,
)


@pytest.fixture
def config() -> GatewayConfig:
    return GatewayConfig(
        master_key="test-master-key",
        model_discovery=False,
        providers={
            "openai": {"api_key": "sk-openai"},
            "anthropic": {"api_key": "sk-anthropic"},
            "mistral": {"api_key": "sk-mistral"},
        },
    )


def _spec(
    weights: dict[str, float],
    *,
    candidates: list[str] | None = None,
    default: str = "openai:gpt-5",
    on_failure: list[str] | None = None,
) -> PolicySpec:
    return PolicySpec.model_validate(
        {
            "select": [
                {
                    "router": WEIGHTED_BACKEND,
                    "candidates": candidates or ["openai:gpt-5", "anthropic:claude-sonnet-4-5"],
                    "weights": weights,
                },
                {"default": default},
            ],
            "on_failure": on_failure or [],
        }
    )


def _rank(pool: list[str], weights: dict[str, float], seed: int = 0) -> list[str]:
    backend = WeightedRouterBackend(random.Random(seed))
    ctx = RoutingContext(
        user_id="alice", default_model=pool[-1], candidate_pool=pool, weights=weights
    )
    return asyncio.run(backend.rank(ctx)).ordered_models


# -- schema ----------------------------------------------------------------


def test_weights_are_normalized_not_percentages() -> None:
    # 7:3 and 70:30 are the same split. Requiring a sum of 100 would forbid the
    # first spelling for no benefit.
    pool = ["a:b", "c:d"]
    assert declared_shares({"a:b": 7, "c:d": 3}, pool) == declared_shares({"a:b": 70, "c:d": 30}, pool)


def test_a_weighted_entry_requires_weights() -> None:
    # Omitting a *candidate* already means zero share, so an omitted map cannot also
    # mean "even split" without contradicting that. An even split is written out.
    with pytest.raises(ValidationError, match="needs `weights`"):
        PolicySpec.model_validate(
            {
                "select": [
                    {"router": WEIGHTED_BACKEND, "candidates": ["openai:gpt-5", "openai:gpt-5-mini"]},
                    {"default": "openai:gpt-5"},
                ]
            }
        )


def test_weights_are_refused_on_any_other_entry() -> None:
    # A weight map on a knn entry or a static target reads as a traffic split and
    # would silently do nothing.
    with pytest.raises(ValidationError, match="only applies to a `router: weighted` entry"):
        PolicySpec.model_validate(
            {
                "select": [
                    {
                        "router": "knn",
                        "candidates": ["openai:gpt-5", "openai:gpt-5-mini"],
                        "weights": {"openai:gpt-5": 1},
                    },
                    {"default": "openai:gpt-5"},
                ]
            }
        )
    with pytest.raises(ValidationError, match="only applies to a `router: weighted` entry"):
        PolicySpec.model_validate(
            {"select": [{"default": "openai:gpt-5", "weights": {"openai:gpt-5": 1}}]}
        )


@pytest.mark.parametrize("weight", [-1, float("inf"), float("nan")])
def test_a_weight_must_be_finite_and_non_negative(weight: float) -> None:
    with pytest.raises(ValidationError, match="finite, non-negative"):
        _spec({"openai:gpt-5": weight, "anthropic:claude-sonnet-4-5": 1})


def test_an_all_zero_split_is_refused() -> None:
    # It would select nothing and the policy would always serve its default, which
    # is a routing policy that does not route.
    with pytest.raises(ValidationError, match="every weight is 0"):
        _spec({"openai:gpt-5": 0, "anthropic:claude-sonnet-4-5": 0})


def test_a_weight_key_must_name_a_candidate() -> None:
    # A typo'd key would renormalize the recognized weights to fill the gap, so the
    # split that runs would be nothing like the one written.
    with pytest.raises(ValidationError, match="do not name a candidate"):
        _spec({"openai:gpt-4": 5, "anthropic:claude-sonnet-4-5": 5})


def test_the_default_target_may_be_weighted() -> None:
    # The default is part of the pool, so weighting it is how it takes a share of
    # the traffic rather than only serving as the last resort.
    spec = _spec(
        {"openai:gpt-5-mini": 1, "openai:gpt-5": 1},
        candidates=["openai:gpt-5-mini", "anthropic:claude-sonnet-4-5"],
        default="openai:gpt-5",
    )
    assert spec.router_weights["openai:gpt-5"] == 1
    assert "openai:gpt-5" in spec.router_candidates


def test_a_weighted_policy_is_dynamic_and_names_its_backend() -> None:
    spec = _spec({"openai:gpt-5": 70, "anthropic:claude-sonnet-4-5": 30})
    assert spec.is_dynamic
    assert spec.router_backend == WEIGHTED_BACKEND


# -- the draw --------------------------------------------------------------


def test_the_head_follows_the_weights() -> None:
    pool = ["openai:gpt-5", "anthropic:claude-sonnet-4-5"]
    weights = {"openai:gpt-5": 70.0, "anthropic:claude-sonnet-4-5": 30.0}
    rng = random.Random(7)
    heads = Counter(weighted_ordering(pool, weights, rng)[0] for _ in range(4000))
    # Wide bounds: this asserts the split is real, not that the RNG is a given one.
    assert 0.65 < heads["openai:gpt-5"] / 4000 < 0.75


def test_the_whole_pool_is_ordered_so_a_failure_stays_balanced() -> None:
    # The draw continues without replacement, so a failure lands on another
    # weighted provider rather than jumping to on_failure.
    pool = ["openai:gpt-5", "anthropic:claude-sonnet-4-5", "mistral:mistral-large-latest"]
    ordered = _rank(
        pool,
        {"openai:gpt-5": 5, "anthropic:claude-sonnet-4-5": 3, "mistral:mistral-large-latest": 2},
    )
    assert sorted(ordered) == sorted(pool)


def test_a_zero_weight_candidate_is_drained_not_removed() -> None:
    # `{a: 1, b: 0}` is how a provider is drained without being deleted: it takes no
    # traffic but stays in the plan as a failover target, at the tail.
    pool = ["openai:gpt-5", "openai:gpt-5-mini"]
    weights: dict[str, float] = {"openai:gpt-5": 1, "openai:gpt-5-mini": 0}
    for seed in range(25):
        assert _rank(pool, weights, seed=seed) == ["openai:gpt-5", "openai:gpt-5-mini"]
    assert declared_shares(weights, pool)["openai:gpt-5-mini"] == 0.0


def test_a_zero_weight_candidate_never_wins_the_boundary_draw() -> None:
    # A draw that lands exactly on the cumulative total (or past it through
    # floating-point drift) must fall back to a candidate that has a share.
    class _Ceiling(random.Random):
        def random(self) -> float:
            return 1.0

    ordered = weighted_ordering(
        ["openai:gpt-5", "openai:gpt-5-mini"],
        {"openai:gpt-5": 1, "openai:gpt-5-mini": 0},
        _Ceiling(),
    )
    assert ordered[0] == "openai:gpt-5"


def test_a_fully_drained_pool_serves_its_one_survivor() -> None:
    # Only reachable when every weighted candidate was filtered out for this caller
    # and the drained one is all that is left. Serving it beats declining.
    assert declared_shares({"a:b": 0}, ["a:b"]) == {"a:b": 100.0}


def test_a_drained_pool_reports_the_head_that_actually_serves() -> None:
    # The draw has nothing to sample here, so `weighted_ordering` keeps declared
    # order and the first candidate serves every request. The reported split has to
    # say that: an even one would describe a balanced policy that is not balancing,
    # and these numbers are what `explain`, the CLI and `router_weights` print.
    pool = ["openai:gpt-5", "openai:gpt-5-mini"]
    weights: dict[str, float] = {"openai:gpt-5": 0, "openai:gpt-5-mini": 0}
    assert declared_shares(weights, pool) == {"openai:gpt-5": 100.0, "openai:gpt-5-mini": 0.0}
    for seed in range(10):
        assert weighted_ordering(pool, weights, random.Random(seed)) == pool


def test_a_drained_pool_agrees_between_the_split_and_the_ordering() -> None:
    # The property the case above is an instance of: whatever `declared_shares`
    # calls the majority share is the candidate `weighted_ordering` leads with, so
    # no surface can report a split the runtime does not serve.
    pool = ["a:b", "c:d", "e:f"]
    weights: dict[str, float] = {"a:b": 0, "c:d": 0, "e:f": 0}
    shares = declared_shares(weights, pool)
    heaviest = max(shares, key=lambda selector: shares[selector])
    for seed in range(10):
        assert weighted_ordering(pool, weights, random.Random(seed))[0] == heaviest
    assert explain_ordering(pool, weights)[0] == heaviest


def test_the_zero_weight_tail_keeps_declared_order() -> None:
    ordered = weighted_ordering(["a:b", "c:d", "e:f"], {"c:d": 0, "e:f": 0, "a:b": 1}, random.Random(1))
    assert ordered == ["a:b", "c:d", "e:f"]


def test_the_backend_declines_when_no_candidate_is_usable() -> None:
    backend = WeightedRouterBackend(random.Random(0))
    decision = asyncio.run(
        backend.rank(RoutingContext(user_id="alice", default_model="openai:gpt-5", candidate_pool=[]))
    )
    assert decision.ordered_models == []


def test_the_backend_declines_a_policy_with_no_weights() -> None:
    # Unreachable through a validated policy; this is the older-document case, and
    # declining serves the policy's default target.
    backend = WeightedRouterBackend(random.Random(0))
    decision = asyncio.run(
        backend.rank(
            RoutingContext(
                user_id="alice", default_model="openai:gpt-5", candidate_pool=["openai:gpt-5", "a:b"]
            )
        )
    )
    assert decision.ordered_models == []
    assert "no weights" in decision.rationale


def test_a_weighted_decision_is_not_logged_per_request() -> None:
    # One INFO line per request is the whole log at load-balancer volume, and the
    # usage row already records what served.
    backend = WeightedRouterBackend(random.Random(0))
    decision = asyncio.run(
        backend.rank(
            RoutingContext(
                user_id="alice",
                default_model="openai:gpt-5",
                candidate_pool=["openai:gpt-5", "anthropic:claude-sonnet-4-5"],
                weights={"openai:gpt-5": 1, "anthropic:claude-sonnet-4-5": 1},
            )
        )
    )
    assert decision.log_decision is False


def test_a_weighted_policy_is_not_a_consumer_of_routing_memory() -> None:
    # A weighted policy names a router, but it reads no examples, so the routing
    # memory surfaces must not claim it. `/v1/routing/status` would report it under
    # a warmth it never uses, and `rank` would let its pool decide which score keys
    # a user may teach, refusing the examples a learned policy is being prepared with.
    from gateway.api.routes.routing_memory import ScoredExample, _learned_policies, _validated_scores

    weighted_config = GatewayConfig(
        master_key="test-master-key",
        model_discovery=False,
        providers={"openai": {"api_key": "sk-openai"}, "anthropic": {"api_key": "sk-anthropic"}},
        routing=RoutingConfig(
            policies={"balanced": _spec({"openai:gpt-5": 70, "anthropic:claude-sonnet-4-5": 30})}
        ),
    )
    assert not backend_pool_is_teachable(WEIGHTED_BACKEND)
    assert not backend_pool_is_teachable(" Weighted ")
    assert _learned_policies(weighted_config, "alice") == []
    assert _validated_scores(
        weighted_config, "alice", [ScoredExample(prompt="hi", scores={"openai:gpt-5-mini": 1.0})]
    ) == {"openai:gpt-5-mini": "openai:gpt-5-mini"}


def test_a_noop_placeholder_pool_is_still_teachable() -> None:
    # `noop` exists to hold a policy's shape while its pool is being taught, so its
    # candidates are the very ones an operator is seeding. Excluding it alongside
    # `weighted` would drop the typo guard and hide the pool's warmth for exactly
    # the workflow the backend was added for.
    from gateway.api.routes.routing_memory import ScoredExample, _learned_policies, _validated_scores

    placeholder = PolicySpec.model_validate(
        {
            "select": [
                {"router": "noop", "candidates": ["openai:gpt-5", "openai:gpt-5-mini"]},
                {"default": "openai:gpt-5"},
            ]
        }
    )
    cfg = GatewayConfig(
        master_key="test-master-key",
        model_discovery=False,
        providers={"openai": {"api_key": "sk-openai"}},
        routing=RoutingConfig(policies={"warming": placeholder}),
    )
    assert backend_pool_is_teachable("noop")
    assert backend_pool_is_teachable("some-future-backend")
    assert [policy.name for policy in _learned_policies(cfg, "alice")] == ["warming"]
    assert _validated_scores(
        cfg, "alice", [ScoredExample(prompt="hi", scores={"openai:gpt-5-mini": 1.0})]
    ) == {"openai:gpt-5-mini": "openai:gpt-5-mini"}
    with pytest.raises(HTTPException, match="do not name a model"):
        _validated_scores(cfg, "alice", [ScoredExample(prompt="hi", scores={"openai:gpt-4o": 1.0})])


def test_the_backend_is_registered_and_needs_no_pricing(config: GatewayConfig) -> None:
    assert isinstance(get_router_backend(config, WEIGHTED_BACKEND), WeightedRouterBackend)
    assert isinstance(get_router_backend(config, " Weighted "), WeightedRouterBackend)
    assert backend_is_weighted(" WEIGHTED ")
    # The weighted router reads no prices, so demanding them would refuse a working
    # policy. Only the learned router scores cost.
    assert not backend_requires_pricing(WEIGHTED_BACKEND)
    assert backend_requires_pricing("knn")


# -- the request path ------------------------------------------------------


def test_the_ordering_becomes_the_plan_ahead_of_on_failure(config: GatewayConfig) -> None:
    spec = _spec(
        {"openai:gpt-5": 1, "anthropic:claude-sonnet-4-5": 0},
        on_failure=["mistral:mistral-large-latest"],
    )
    ordering = asyncio.run(
        decide_ordering(
            config,
            spec,
            policy_name="balanced",
            user_id="alice",
            allowlist=None,
            signal=RoutingSignal(task_signal="hi", trace_signal="hi", trace_anchor="hi"),
        )
    )
    assert ordering is not None
    plan = compile_policy(config, "balanced", spec, user_id="alice", router_ordering=ordering)
    assert [f"{attempt.instance}:{attempt.model}" for attempt in plan.attempts] == [
        "openai:gpt-5",
        "anthropic:claude-sonnet-4-5",
        "mistral:mistral-large-latest",
    ]
    assert plan.selection_reason == f"router:{WEIGHTED_BACKEND}"
    assert plan.attempts[-1].selection_reason == "on_failure"


def test_the_caller_can_opt_out_and_get_the_default(config: GatewayConfig) -> None:
    # `Otari-Router: off` means the same thing on every backend: serve the policy's
    # default target. On a weighted policy that pins the caller to one provider.
    spec = _spec({"openai:gpt-5": 1, "anthropic:claude-sonnet-4-5": 99})
    ordering = asyncio.run(
        decide_ordering(
            config,
            spec,
            policy_name="balanced",
            user_id="alice",
            allowlist=None,
            signal=RoutingSignal(opted_out=True),
        )
    )
    assert ordering is not None
    assert ordering.selectors == []
    plan = compile_policy(config, "balanced", spec, user_id="alice", router_ordering=ordering)
    assert f"{plan.head.instance}:{plan.head.model}" == "openai:gpt-5"
    assert plan.selection_reason == "default"


def test_a_filtered_pool_only_ranks_what_the_caller_may_use(config: GatewayConfig) -> None:
    spec = _spec({"openai:gpt-5": 99, "anthropic:claude-sonnet-4-5": 1})
    ordering = asyncio.run(
        decide_ordering(
            config,
            spec,
            policy_name="balanced",
            user_id="alice",
            allowlist=["anthropic:*"],
            signal=RoutingSignal(task_signal="hi", trace_signal="hi", trace_anchor="hi"),
        )
    )
    assert ordering is not None
    assert ordering.selectors == ["anthropic:claude-sonnet-4-5"]


# -- explain ---------------------------------------------------------------


def test_explain_shows_the_split_rather_than_the_decline_path(config: GatewayConfig) -> None:
    # A weighted decision needs no request state, so the synchronous surfaces show
    # the real ordering by share instead of the default-target decline path.
    spec = _spec({"openai:gpt-5": 70, "anthropic:claude-sonnet-4-5": 30})
    ordering, shares = explain_router_ordering(config, spec)
    assert ordering is not None
    assert ordering.selectors == ["openai:gpt-5", "anthropic:claude-sonnet-4-5"]
    assert [(item.selector, round(item.share_pct)) for item in shares] == [
        ("openai:gpt-5", 70),
        ("anthropic:claude-sonnet-4-5", 30),
    ]
    assert shares[0].canonical == "openai:gpt-5"


def test_explain_is_deterministic(config: GatewayConfig) -> None:
    # Sampling the draw would make the same command print a different plan per run.
    spec = _spec({"openai:gpt-5": 50, "anthropic:claude-sonnet-4-5": 50})
    first, _ = explain_router_ordering(config, spec)
    second, _ = explain_router_ordering(config, spec)
    assert first is not None and second is not None
    assert first.selectors == second.selectors


def test_explain_renormalizes_over_what_the_caller_may_use(config: GatewayConfig) -> None:
    # With the 70% candidate forbidden, the 30% one really does serve every request.
    spec = _spec({"openai:gpt-5": 70, "anthropic:claude-sonnet-4-5": 30})
    _, shares = explain_router_ordering(config, spec, allowlist=["anthropic:*"])
    assert [(item.selector, item.share_pct) for item in shares] == [
        ("anthropic:claude-sonnet-4-5", 100.0)
    ]


def test_explain_keeps_a_filtered_candidate_in_the_dropped_list(config: GatewayConfig) -> None:
    # The dropped list is most of the point of explain: it is how an operator finds
    # that a balanced policy compiles to one provider for a given key. Pre-filtering
    # the ordering would hide it.
    spec = _spec({"openai:gpt-5": 70, "anthropic:claude-sonnet-4-5": 30})
    ordering, _ = explain_router_ordering(config, spec, allowlist=["anthropic:*"])
    plan = compile_policy(config, "balanced", spec, allowlist=["anthropic:*"], router_ordering=ordering)
    assert [(item.selector, item.reason) for item in plan.dropped] == [("openai:gpt-5", "not_allowed")]


def test_explain_names_every_candidate_when_the_whole_split_is_filtered_out(
    config: GatewayConfig,
) -> None:
    # The case an operator most needs explain for. With the split gone and only the
    # failure chain usable, the plan is the chain, and the split has to appear in
    # `dropped` rather than vanish: reporting the chain alone reads as a policy that
    # never had a split.
    spec = _spec(
        {"openai:gpt-5": 70, "anthropic:claude-sonnet-4-5": 30},
        on_failure=["mistral:mistral-large-latest"],
    )
    allowlist = ["mistral:*"]
    ordering, shares = explain_router_ordering(config, spec, allowlist=allowlist)
    assert shares == []
    assert ordering is not None
    plan = compile_policy(config, "balanced", spec, allowlist=allowlist, router_ordering=ordering)
    assert [f"{attempt.instance}:{attempt.model}" for attempt in plan.attempts] == [
        "mistral:mistral-large-latest"
    ]
    assert [(item.selector, item.reason) for item in plan.dropped] == [
        ("openai:gpt-5", "not_allowed"),
        ("anthropic:claude-sonnet-4-5", "not_allowed"),
    ]


def test_explain_declines_for_a_router_that_needs_a_request(config: GatewayConfig) -> None:
    knn = PolicySpec.model_validate(
        {
            "select": [
                {"router": "knn", "candidates": ["openai:gpt-5", "openai:gpt-5-mini"]},
                {"default": "openai:gpt-5"},
            ]
        }
    )
    assert explain_router_ordering(config, knn) == (None, [])


def test_describe_split_and_explain_ordering_agree() -> None:
    pool = ["a:b", "c:d"]
    weights = {"a:b": 1, "c:d": 3}
    assert explain_ordering(pool, weights) == ["c:d", "a:b"]
    assert describe_split(pool, weights) == "c:d 75%, a:b 25%"
