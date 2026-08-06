"""Unit tests for the routing-policy compiler's empty-plan behavior (issue #463).

A policy that compiles to nothing is the one outcome that has to explain itself,
because the caller sees a failure for a model name that exists. The two ways it
happens are different faults with different fixes, so they get different statuses
and different wording:

- access rules filtered every candidate out: the caller is being denied something
  that does exist, which is a 403 they can act on by asking for access;
- nothing resolves to a configured provider: the gateway's provider config no
  longer matches its policies, which is a 502. The caller did nothing wrong.

Sending 403 with access-rule wording for both told an operator whose provider
instance had been deleted to go audit their allow-lists.
"""

import logging
from collections.abc import Callable, Iterator

import pytest

from gateway.core.config import GatewayConfig
from gateway.log_config import logger as gateway_logger
from gateway.models.routing import PolicySpec
from gateway.services.routing import (
    BudgetState,
    NoEligibleCandidatesError,
    RouterOrdering,
    compile_policy,
    selection_consults_router,
)


@pytest.fixture
def router_warnings(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Callable[[], list[str]]]:
    """Call the returned function to read the router warnings logged so far.

    Two pieces of plumbing: the ``gateway`` logger does not propagate
    (``log_config`` sets ``propagate=False``), so caplog's handler is attached to
    it directly rather than to root; and the once-per-process warned set is reset
    so an earlier test cannot suppress this one's warning.
    """
    monkeypatch.setattr("gateway.services.routing.backends._warned_missing", set())
    gateway_logger.addHandler(caplog.handler)
    caplog.set_level(logging.WARNING, logger="gateway")
    try:
        yield lambda: [record.getMessage() for record in caplog.records if "router" in record.getMessage()]
    finally:
        gateway_logger.removeHandler(caplog.handler)


@pytest.fixture
def config() -> GatewayConfig:
    return GatewayConfig(
        master_key="test-master-key",
        model_discovery=False,
        providers={"openai": {"api_key": "sk-openai"}},
    )


def _spec(default: str, *on_failure: str) -> PolicySpec:
    return PolicySpec.model_validate({"select": [{"default": default}], "on_failure": list(on_failure)})


def test_a_resolvable_candidate_compiles(config: GatewayConfig) -> None:
    plan = compile_policy(config, "fast", _spec("openai:gpt-5-mini"))
    assert [attempt.model for attempt in plan.attempts] == ["gpt-5-mini"]


def test_every_candidate_unresolvable_is_a_502_about_provider_config(config: GatewayConfig) -> None:
    """The shape of an operator deleting a provider instance out from under a
    running gateway, which the resolver cannot distinguish from a typo.
    """
    with pytest.raises(NoEligibleCandidatesError) as exc_info:
        compile_policy(config, "stale", _spec("nosuchprovider:some-model", "alsogone:other-model"))

    err = exc_info.value
    assert err.status_code == 502
    assert "resolve to a configured provider" in err.caller_detail
    # The wrong config must not be named: this is not an access problem.
    assert "filtered" not in err.caller_detail
    # Targets stay off the caller-facing string; a policy exists partly to hide them.
    assert "nosuchprovider" not in err.caller_detail
    assert "nosuchprovider" in err.operator_detail


def test_every_candidate_filtered_is_a_403_about_access(config: GatewayConfig) -> None:
    with pytest.raises(NoEligibleCandidatesError) as exc_info:
        compile_policy(config, "denied", _spec("openai:gpt-5-mini"), allowlist=["openai:something-else"])

    err = exc_info.value
    assert err.status_code == 403
    assert "filtered out before dispatch" in err.caller_detail
    assert "gpt-5-mini" not in err.caller_detail


def test_a_mixed_drop_reports_the_actionable_one(config: GatewayConfig) -> None:
    """One candidate the caller may not use plus one that no longer exists is
    still an access decision from the caller's side: there is a real model in the
    policy, and access is the reason they did not get it. Reporting 502 here would
    tell them to wait for an operator when asking for access is the fix.
    """
    with pytest.raises(NoEligibleCandidatesError) as exc_info:
        compile_policy(config, "mixed", _spec("openai:gpt-5-mini", "nosuchprovider:x"), allowlist=["openai:other"])

    assert exc_info.value.status_code == 403
    assert {item.reason for item in exc_info.value.dropped} == {"not_allowed", "unresolvable"}


def _router_spec(*candidates: str) -> PolicySpec:
    pool = list(candidates) or ["openai:gpt-5-nano", "openai:gpt-5-mini"]
    return PolicySpec.model_validate(
        {
            "select": [{"router": "knn", "candidates": pool}, {"default": "openai:gpt-5-mini"}],
            "on_failure": [],
        }
    )


def test_no_ordering_serves_the_default_and_says_nothing(
    config: GatewayConfig, router_warnings: Callable[[], list[str]]
) -> None:
    """Compiling without an ordering is silent, whichever reason produced it.

    A decline (cold pool, low confidence, caller opted out) is normal operation, and
    ``explain`` and the CLI compile with no request at all by design. Only the
    caller can tell those apart from a backend this build does not have, so the
    warning lives there (see ``test_routing_decide``); warning here made ``explain``
    report a misconfiguration that did not exist.
    """
    for ordering in (None, RouterOrdering([], rationale="cold pool")):
        plan = compile_policy(config, "routed", _router_spec(), router_ordering=ordering)
        assert [attempt.model for attempt in plan.attempts] == ["gpt-5-mini"]
        assert plan.selection_reason == "default"
        assert plan.router_ordering is None
    assert router_warnings() == []


def test_a_router_ranking_becomes_the_whole_plan(config: GatewayConfig) -> None:
    """The ranking is the plan, not just its head.

    The walker can try candidates in order, so keeping the rest means a routed
    request that fails over lands on the router's second choice rather than
    jumping to the operator's failure chain. Every attempt carries the router as
    its selection reason, which is what the usage rows are attributed to.
    """
    spec = PolicySpec.model_validate(
        {
            "select": [
                {"router": "knn", "candidates": ["openai:gpt-5-nano", "openai:gpt-5-mini"]},
                {"default": "openai:gpt-5-mini"},
            ],
            "on_failure": ["openai:gpt-5"],
        }
    )
    ordering = RouterOrdering(["openai:gpt-5-nano", "openai:gpt-5-mini"], confidence=0.8, rationale="knn")

    plan = compile_policy(config, "routed", spec, router_ordering=ordering)

    assert [attempt.model for attempt in plan.attempts] == ["gpt-5-nano", "gpt-5-mini", "gpt-5"]
    assert plan.selection_reason == "router:knn"
    assert [attempt.selection_reason for attempt in plan.attempts] == [
        "router:knn",
        "router:knn",
        "on_failure",
    ]
    # Kept on the plan so the activity log can say why this model was chosen.
    assert plan.router_ordering is not None
    assert plan.router_ordering.confidence == 0.8


def test_a_routed_candidate_the_caller_may_not_use_is_still_dropped(config: GatewayConfig) -> None:
    """The allow-list outranks the router.

    The pool is filtered before ranking (see ``services/routing/decide``), so this
    is the belt-and-braces case: a stale ordering, or a backend that returned
    something outside the pool, must not become an access-control bypass.
    """
    ordering = RouterOrdering(["openai:gpt-5-nano", "openai:gpt-5-mini"], rationale="knn")

    plan = compile_policy(
        config,
        "routed",
        _router_spec(),
        allowlist=["openai:gpt-5-mini"],
        router_ordering=ordering,
    )

    assert [attempt.model for attempt in plan.attempts] == ["gpt-5-mini"]
    assert [dropped.reason for dropped in plan.dropped] == ["not_allowed"]


def test_a_condition_ahead_of_the_router_means_the_router_is_not_consulted() -> None:
    """Order decides whether the router runs at all.

    A ``when`` entry ahead of the router wins outright and the ranking is thrown
    away, so asking a backend for one costs a paid embedding call plus a scan of the
    caller's examples for nothing, and logs a decision the request did not use. The
    pipeline gates on this before it ranks.
    """
    spec = PolicySpec.model_validate(
        {
            "select": [
                {"when": {"budget_used_pct": {"gte": 80}}, "target": "openai:gpt-5-nano"},
                {"router": "knn", "candidates": ["openai:gpt-5-nano", "openai:gpt-5-mini"]},
                {"default": "openai:gpt-5-mini"},
            ]
        }
    )

    assert not selection_consults_router(spec, budget=BudgetState(used_pct=95.0))
    # Under the threshold the condition does not match, so the router is next in line.
    assert selection_consults_router(spec, budget=BudgetState(used_pct=10.0))


def test_a_policy_with_no_router_never_consults_one() -> None:
    assert not selection_consults_router(_spec("openai:gpt-5-mini"))


def test_a_bare_router_policy_always_consults_it() -> None:
    assert selection_consults_router(_router_spec())
