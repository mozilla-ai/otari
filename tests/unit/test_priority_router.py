"""The ``priority`` router backend: candidates in declared order, the first with room serving.

Room is decided by the walker under the ``per: model`` rate limits, so what is
pinned here is the order the plan is compiled in and how it is shown.
"""

import asyncio

import pytest

from gateway.core.config import GatewayConfig
from gateway.models.routing import PolicySpec
from gateway.services.routing.backends import (
    PRIORITY_BACKEND,
    backend_pool_is_teachable,
    backend_requires_pricing,
    known_backends,
)
from gateway.services.routing.compiler import compile_policy
from gateway.services.routing.decide import RoutingSignal, decide_ordering, explain_router_ordering


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


def _spec() -> PolicySpec:
    return PolicySpec.model_validate(
        {
            "select": [
                {"router": PRIORITY_BACKEND, "candidates": ["openai:gpt-5", "anthropic:claude-sonnet-4-5"]},
                {"default": "anthropic:claude-sonnet-4-5"},
            ],
            "on_failure": ["mistral:mistral-small"],
        }
    )


def _canonical(config: GatewayConfig, spec: PolicySpec, ordering: object) -> list[str]:
    plan = compile_policy(config, "spill", spec, user_id="alice", router_ordering=ordering)  # type: ignore[arg-type]
    return [f"{attempt.instance}:{attempt.model}" for attempt in plan.attempts]


def test_the_plan_keeps_the_declared_order_ahead_of_on_failure(config: GatewayConfig) -> None:
    spec = _spec()
    ordering = asyncio.run(
        decide_ordering(
            config,
            spec,
            policy_name="spill",
            user_id="alice",
            allowlist=None,
            signal=RoutingSignal(task_signal="hi", trace_signal="hi", trace_anchor="hi"),
        )
    )

    assert _canonical(config, spec, ordering) == [
        "openai:gpt-5",
        "anthropic:claude-sonnet-4-5",
        "mistral:mistral-small",
    ]


def test_a_candidate_the_caller_may_not_use_is_left_out(config: GatewayConfig) -> None:
    spec = _spec()
    ordering = asyncio.run(
        decide_ordering(
            config,
            spec,
            policy_name="spill",
            user_id="alice",
            allowlist=["anthropic:claude-sonnet-4-5", "mistral:mistral-small"],
            signal=RoutingSignal(task_signal="hi", trace_signal="hi", trace_anchor="hi"),
        )
    )

    assert ordering is not None
    assert ordering.selectors == ["anthropic:claude-sonnet-4-5"]


def test_explain_shows_the_declared_order(config: GatewayConfig) -> None:
    spec = _spec()
    ordering, shares = explain_router_ordering(config, spec)

    assert shares == []
    assert _canonical(config, spec, ordering)[:2] == ["openai:gpt-5", "anthropic:claude-sonnet-4-5"]


def test_priority_is_a_known_backend_that_needs_no_pricing_or_examples() -> None:
    assert PRIORITY_BACKEND in known_backends()
    assert not backend_requires_pricing(PRIORITY_BACKEND)
    assert not backend_pool_is_teachable(PRIORITY_BACKEND)
