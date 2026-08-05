"""Unit tests for the router step: schema rules for a router entry, and the
decision layer that turns a policy plus a request into an ordering.

The decision layer is where a router can go wrong in ways the compiler cannot see:
handing a backend candidates the caller may not use, asking a backend that does not
exist, or asking at all when the caller said not to.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator

import pytest
from pydantic import ValidationError

from gateway.core.config import GatewayConfig
from gateway.log_config import logger as gateway_logger
from gateway.models.routing import PolicySpec
from gateway.services.routing.backends import (
    NoOpRouterBackend,
    RouterBackend,
    RoutingContext,
    RoutingDecision,
    clear_router_backend_cache,
    get_router_backend,
)
from gateway.services.routing.decide import RoutingSignal, decide_ordering


@pytest.fixture
def router_warnings(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Callable[[], list[str]]]:
    """Call the returned function to read the router warnings logged so far.

    Two pieces of plumbing: the ``gateway`` logger does not propagate
    (``log_config`` sets ``propagate=False``), so caplog's handler is attached to it
    directly rather than to root; and the once-per-process warned set is reset so an
    earlier test cannot suppress this one's warning.
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
        providers={"openai": {"api_key": "sk-openai"}, "anthropic": {"api_key": "sk-anthropic"}},
    )


def _spec(*candidates: str, default: str = "openai:gpt-5", backend: str = "knn") -> PolicySpec:
    return PolicySpec.model_validate(
        {
            "select": [
                {"router": backend, "candidates": list(candidates) or ["openai:gpt-5-mini", "openai:gpt-5"]},
                {"default": default},
            ]
        }
    )


def _signal(**kw: object) -> RoutingSignal:
    return RoutingSignal(task_signal="hello", trace_signal="hello", trace_anchor="hello", **kw)  # type: ignore[arg-type]


# -- schema ----------------------------------------------------------------


def test_a_router_entry_requires_candidates() -> None:
    # Without a pool there is nothing to route among, and an operator who wrote it
    # believes they configured routing.
    with pytest.raises(ValidationError, match="needs `candidates`"):
        PolicySpec.model_validate({"select": [{"router": "knn"}, {"default": "openai:gpt-5"}]})


def test_a_router_entry_needs_at_least_two_candidates() -> None:
    with pytest.raises(ValidationError, match="at least 2 `candidates`"):
        PolicySpec.model_validate(
            {"select": [{"router": "knn", "candidates": ["openai:gpt-5"]}, {"default": "openai:gpt-5"}]}
        )


def test_candidates_are_rejected_on_a_static_entry() -> None:
    with pytest.raises(ValidationError, match="only applies to a `router` entry"):
        PolicySpec.model_validate(
            {"select": [{"default": "openai:gpt-5", "candidates": ["openai:gpt-5-mini"]}]}
        )


def test_duplicate_candidates_are_refused() -> None:
    with pytest.raises(ValidationError, match="listed twice"):
        PolicySpec.model_validate(
            {
                "select": [
                    {"router": "knn", "candidates": ["openai:gpt-5", "openai:gpt-5"]},
                    {"default": "openai:gpt-5"},
                ]
            }
        )


def test_the_default_target_joins_the_pool_as_its_last_resort() -> None:
    # An operator who lists only the cheap models still gets the strong one as the
    # cascade's tail, because that is what serves when the router declines.
    spec = _spec("openai:gpt-5-nano", "openai:gpt-5-mini", default="openai:gpt-5")
    assert spec.router_candidates == ["openai:gpt-5-nano", "openai:gpt-5-mini", "openai:gpt-5"]
    # ...and is not duplicated when it was listed.
    listed = _spec("openai:gpt-5-mini", "openai:gpt-5", default="openai:gpt-5")
    assert listed.router_candidates == ["openai:gpt-5-mini", "openai:gpt-5"]


def test_the_candidate_cap_counts_the_whole_routed_pool() -> None:
    # The walker cascades through the ranking, so the pool counts against the cap
    # rather than one head candidate.
    with pytest.raises(ValidationError, match="at most 5 candidates"):
        PolicySpec.model_validate(
            {
                "select": [
                    {
                        "router": "knn",
                        "candidates": [
                            "openai:a",
                            "openai:b",
                            "openai:c",
                            "openai:d",
                            "openai:e",
                        ],
                    },
                    {"default": "openai:f"},
                ]
            }
        )


def test_a_router_policy_is_dynamic_and_its_candidates_are_validated_selectors() -> None:
    spec = _spec("openai:gpt-5-mini", "anthropic:claude-haiku-4-5")
    assert spec.is_dynamic is True
    assert spec.router_backend == "knn"
    # static_selectors feeds startup and write-time checks (resolvable, not an
    # alias or another policy), so the candidates have to appear in it.
    assert "anthropic:claude-haiku-4-5" in spec.static_selectors()


# -- backend resolution ----------------------------------------------------


def test_noop_backend_declines() -> None:
    backend = get_router_backend(GatewayConfig(), "noop")
    assert isinstance(backend, NoOpRouterBackend)
    assert isinstance(backend, RouterBackend)


@pytest.mark.asyncio
async def test_noop_decline_is_empty() -> None:
    decision = await NoOpRouterBackend().rank(
        RoutingContext(user_id="u", default_model="openai:gpt-5", candidate_pool=["openai:gpt-5-mini"])
    )
    assert decision.ordered_models == []


def test_an_unknown_backend_resolves_to_none_rather_than_raising() -> None:
    # A build without the named backend must serve the policy's default target, not
    # fail every request through it.
    clear_router_backend_cache()
    assert get_router_backend(GatewayConfig(), "cheapest") is None


@pytest.mark.asyncio
async def test_an_unknown_backend_warns_once_per_policy(
    config: GatewayConfig, router_warnings: Callable[[], list[str]]
) -> None:
    """The one "no ordering" case that is a misconfiguration gets one log line.

    Once, because the condition is static config and the policy compiles on every
    request through it. Per policy, because the policy name is what makes the
    warning actionable, and suppressing per backend name alone would hide a second
    broken policy behind the first.
    """
    unknown = _spec(backend="cheapest")
    for _ in range(3):
        ordering = await decide_ordering(
            config, unknown, policy_name="first", user_id="u", allowlist=None, signal=_signal()
        )
    assert ordering is None
    warnings = router_warnings()
    assert len(warnings) == 1
    assert "first" in warnings[0]
    assert "cheapest" in warnings[0]
    # It names what will serve instead, which is the question an operator has next.
    assert "openai:gpt-5" in warnings[0]

    await decide_ordering(
        config, unknown, policy_name="second", user_id="u", allowlist=None, signal=_signal()
    )
    assert len(router_warnings()) == 2


@pytest.mark.asyncio
async def test_a_decline_is_not_warned_about(
    config: GatewayConfig, recorder: _Recorder, router_warnings: Callable[[], list[str]]
) -> None:
    # A cold pool or an opted-out caller is normal operation, and warning on it
    # would log a line for every request a cold router serves, which is all of them
    # until someone teaches it.
    await decide_ordering(
        config, _spec(), policy_name="smart", user_id="u", allowlist=None, signal=_signal(opted_out=True)
    )
    assert router_warnings() == []


def test_the_knn_backend_instance_is_reused_for_one_config() -> None:
    # It holds the trace-sticky decision cache, so a fresh instance per request
    # would silently break stickiness.
    clear_router_backend_cache()
    config = GatewayConfig()
    assert get_router_backend(config, "knn") is get_router_backend(config, "knn")
    clear_router_backend_cache()
    assert get_router_backend(config, "knn") is not None


# -- the decision layer ----------------------------------------------------


class _Recorder:
    """A backend that records what it was asked and ranks the pool as given."""

    def __init__(self) -> None:
        self.seen: RoutingContext | None = None

    async def rank(self, ctx: RoutingContext) -> RoutingDecision:
        self.seen = ctx
        return RoutingDecision(ordered_models=list(ctx.candidate_pool), confidence=1.0, rationale="recorded")


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> _Recorder:
    backend = _Recorder()
    monkeypatch.setattr(
        "gateway.services.routing.decide.get_router_backend", lambda config, name: backend
    )
    return backend


@pytest.mark.asyncio
async def test_a_policy_without_a_router_is_not_asked(config: GatewayConfig) -> None:
    static = PolicySpec.model_validate({"select": [{"default": "openai:gpt-5"}]})
    assert await decide_ordering(
        config, static, policy_name="fast", user_id="u", allowlist=None, signal=_signal()
    ) is None


@pytest.mark.asyncio
async def test_a_surface_with_no_request_is_not_asked(config: GatewayConfig, recorder: _Recorder) -> None:
    # `explain` and the model catalog have no prompt to route on.
    assert await decide_ordering(
        config, _spec(), policy_name="smart", user_id="u", allowlist=None, signal=None
    ) is None
    assert recorder.seen is None


@pytest.mark.asyncio
async def test_the_caller_opt_out_declines_without_asking(config: GatewayConfig, recorder: _Recorder) -> None:
    ordering = await decide_ordering(
        config, _spec(), policy_name="smart", user_id="u", allowlist=None, signal=_signal(opted_out=True)
    )
    assert ordering is not None
    assert ordering.selectors == []
    assert "Otari-Router: off" in ordering.rationale
    # Not asked at all: an opt-out must not spend an embedding call.
    assert recorder.seen is None


@pytest.mark.asyncio
async def test_the_pool_is_filtered_before_the_backend_sees_it(
    config: GatewayConfig, recorder: _Recorder
) -> None:
    # A router that picked a forbidden model would have its choice dropped by the
    # compiler and serve something else, which reads as the router misbehaving.
    ordering = await decide_ordering(
        config,
        _spec("openai:gpt-5-mini", "anthropic:claude-haiku-4-5"),
        policy_name="smart",
        user_id="u",
        allowlist=["openai:*"],
        signal=_signal(),
    )

    assert recorder.seen is not None
    assert recorder.seen.candidate_pool == ["openai:gpt-5-mini", "openai:gpt-5"]
    assert ordering is not None
    assert "anthropic:claude-haiku-4-5" not in ordering.selectors


@pytest.mark.asyncio
async def test_an_unresolvable_candidate_is_left_out(config: GatewayConfig, recorder: _Recorder) -> None:
    await decide_ordering(
        config,
        _spec("openai:gpt-5-mini", "not-a-provider:whatever"),
        policy_name="smart",
        user_id="u",
        allowlist=None,
        signal=_signal(),
    )
    assert recorder.seen is not None
    assert "not-a-provider:whatever" not in recorder.seen.candidate_pool


@pytest.mark.asyncio
async def test_nothing_usable_declines_rather_than_asking(config: GatewayConfig, recorder: _Recorder) -> None:
    ordering = await decide_ordering(
        config,
        _spec("openai:gpt-5-mini", "openai:gpt-5", default="openai:gpt-5"),
        policy_name="smart",
        user_id="u",
        allowlist=["anthropic:*"],
        signal=_signal(),
    )
    assert ordering is not None
    assert ordering.selectors == []
    assert recorder.seen is None


@pytest.mark.asyncio
async def test_the_request_headers_reach_the_backend(config: GatewayConfig, recorder: _Recorder) -> None:
    await decide_ordering(
        config,
        _spec(),
        policy_name="smart",
        user_id="user-7",
        allowlist=None,
        signal=_signal(conversation_id="conv-1", task_id="summarize", has_tools=True, is_continuation=True),
    )
    seen = recorder.seen
    assert seen is not None
    assert (seen.user_id, seen.trace_key, seen.task_id) == ("user-7", "conv-1", "summarize")
    assert (seen.has_tools, seen.is_trace_continuation) == (True, True)
    assert seen.default_model == "openai:gpt-5"
