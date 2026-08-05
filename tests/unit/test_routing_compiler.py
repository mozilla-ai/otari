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
from gateway.services.routing import NoEligibleCandidatesError, compile_policy


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
    monkeypatch.setattr("gateway.services.routing.compiler._warned_routers", set())
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


def _router_spec() -> PolicySpec:
    return PolicySpec.model_validate(
        {"select": [{"router": "knn"}, {"default": "openai:gpt-5-mini"}], "on_failure": []}
    )


def test_an_unavailable_router_warns_once_per_policy(
    config: GatewayConfig, router_warnings: Callable[[], list[str]]
) -> None:
    """Router backends are not wired yet, and the policy compiles on every request
    through it, so an unconditional warning is one log line per request forever.
    The condition is static config: the first line says everything the thousandth
    would, and the rest only bury real warnings.
    """
    spec = _router_spec()
    for _ in range(3):
        plan = compile_policy(config, "routed", spec)

    # Falling through to the default is the safe reading: a router is an
    # optimization and must never be why a request cannot be served.
    assert [attempt.model for attempt in plan.attempts] == ["gpt-5-mini"]
    warnings = router_warnings()
    assert len(warnings) == 1
    assert "routed" in warnings[0]


def test_a_second_policy_naming_the_same_router_warns_on_its_own(
    config: GatewayConfig, router_warnings: Callable[[], list[str]]
) -> None:
    """Keyed per (policy, router), so the suppression cannot hide a second
    misconfigured policy behind the first.
    """
    spec = _router_spec()
    compile_policy(config, "first", spec)
    compile_policy(config, "second", spec)

    assert len(router_warnings()) == 2
