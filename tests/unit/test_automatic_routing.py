"""Automatic provider selection: missing signals, isolation, load and recovery."""

from dataclasses import replace
from decimal import Decimal
from typing import Any

import pytest
from any_llm import LLMProvider
from pydantic import ValidationError
from starlette.datastructures import Headers

from gateway.core.config import GatewayConfig
from gateway.core.settings.routing import AutomaticRouting
from gateway.services.routing import AutomaticRouter, CompiledPlan, preferred_region
from gateway.types.attempt import Attempt


@pytest.fixture
def now() -> list[float]:
    return [100.0]


@pytest.fixture
def router(now: list[float]) -> AutomaticRouter:
    return AutomaticRouter(clock=lambda: now[0])


def _plan() -> CompiledPlan:
    return CompiledPlan(
        policy_name="acme/model",
        attempts=[
            Attempt(1, "us", LLMProvider.OPENAI, "model", {"api_key": "us-secret"}),
            Attempt(2, "eu", LLMProvider.OPENAI, "model", {"api_key": "eu-secret"}),
        ],
    )


def _rank(router: AutomaticRouter, **changes: Any) -> CompiledPlan:
    arguments: dict[str, Any] = {
        "settings": AutomaticRouting.model_validate({"offerings": {"us": {"region": "us"}, "eu": {"region": "eu"}}}),
        "costs": {"us:model": Decimal("0.01"), "eu:model": Decimal("0.01")},
        "scope": "workspace:user:key:chat",
        "anchor": "private conversation",
        "region": None,
        "mode": "chat:completion",
    }
    arguments.update(changes)
    return router.rank(arguments.pop("plan", _plan()), **arguments)


@pytest.mark.parametrize("region,expected", [("eu", "eu"), ("us", "us"), (None, "us")])
def test_prefers_known_local_provider(router: AutomaticRouter, region: str | None, expected: str) -> None:
    assert _rank(router, region=region).head.instance == expected


def test_unknown_prices_are_never_free(router: AutomaticRouter) -> None:
    assert _rank(router, costs={"us:model": None, "eu:model": Decimal(1)}).head.instance == "eu"
    assert _rank(router, costs={"us:model": None, "eu:model": None}).head.instance == "us"
    assert _rank(router, costs={"us:model": Decimal(0), "eu:model": Decimal(1)}).head.instance == "us"


def test_concurrent_load_avoids_busy_provider_and_releases_on_cancel(router: AutomaticRouter) -> None:
    first = _rank(router)
    assert first.automatic is not None
    observation = first.automatic.start(first.head)
    assert _rank(router).head.instance == "eu"
    observation.finish()
    observation.finish()
    assert _rank(router).head.instance == "us"
    assert all(health.active == 0 for health in router.health.values())


def test_capacity_normalizes_load(router: AutomaticRouter) -> None:
    first = _rank(router)
    assert first.automatic is not None
    first.automatic.start(first.head)
    settings = AutomaticRouting.model_validate({"offerings": {"us": {"capacity": 100}}})
    costs = {"us:model": Decimal("0.01"), "eu:model": Decimal("0.012")}
    assert _rank(router, costs=costs).head.instance == "eu"
    assert _rank(router, costs=costs, settings=settings).head.instance == "us"


def test_latency_expires_and_does_not_mix_streaming(router: AutomaticRouter, now: list[float]) -> None:
    first = _rank(router, anchor="")
    assert first.automatic is not None
    slow = first.automatic.start(first.attempts[0])
    fast = first.automatic.start(first.attempts[1])
    now[0] += 1
    fast.finish(success=True)
    now[0] += 9
    slow.finish(success=True)
    assert _rank(router).head.instance == "eu"
    assert _rank(router, mode="chat:stream").head.instance == "us"
    now[0] += 301
    assert _rank(router).head.instance == "us"


def test_outage_overrides_locality_then_recovers(router: AutomaticRouter, now: list[float]) -> None:
    first = _rank(router, region="us")
    assert first.automatic is not None
    first.automatic.start(first.head).finish(unhealthy=True)
    assert _rank(router, region="us").head.instance == "eu"
    now[0] += 16
    assert _rank(router, region="us").head.instance == "us"


def test_session_is_scoped_expires_and_follows_serving_provider(router: AutomaticRouter, now: list[float]) -> None:
    first = _rank(router)
    assert first.automatic is not None
    first.automatic.start(first.attempts[1]).finish(success=True)
    assert _rank(router).head.instance == "eu"
    assert _rank(router, scope="other-workspace:user:key:chat").head.instance == "us"
    assert _rank(router, anchor="different conversation").head.instance == "us"
    assert "private conversation" not in repr(router.affinities)
    assert "secret" not in repr(router.health)
    now[0] += 3601
    assert _rank(router).head.instance == "us"


def test_session_stays_despite_large_cost_change(router: AutomaticRouter) -> None:
    first = _rank(router)
    assert first.automatic is not None
    first.automatic.start(first.attempts[1]).finish(success=True)
    assert _rank(router, costs={"us:model": Decimal(1), "eu:model": Decimal(2)}).head.instance == "eu"


def test_rotated_credential_does_not_inherit_affinity(router: AutomaticRouter) -> None:
    first = _rank(router)
    assert first.automatic is not None
    first.automatic.start(first.attempts[1]).finish(success=True)
    plan = _plan()
    plan = replace(plan, attempts=[plan.attempts[0], replace(plan.attempts[1], kwargs={"api_key": "rotated"})])
    assert _rank(router, plan=plan).head.instance == "us"


def test_rank_before_capping_and_renumber_attempts(router: AutomaticRouter) -> None:
    plan = replace(_plan(), attempts=[Attempt(i, f"p{i}", LLMProvider.OPENAI, "model") for i in range(1, 9)])
    ranked = _rank(router, plan=plan, costs={f"p{i}:model": Decimal(9 - i) for i in range(1, 9)})
    assert ranked.head.instance == "p8"
    assert [a.position for a in ranked.attempts] == [1, 2, 3, 4, 5]
    assert [a.selection_reason for a in ranked.attempts] == ["catalog", *(["on_failure"] * 4)]


@pytest.mark.parametrize("country,region", [("US", "us"), ("PT", "eu"), ("DE", "eu"), ("GB", None), ("XX", None)])
def test_country_requires_explicitly_trusted_header(country: str, region: str | None) -> None:
    headers = Headers({"Trusted-Country": country})
    assert preferred_region(AutomaticRouting(), headers) is None
    assert preferred_region(AutomaticRouting(country_header="Trusted-Country"), headers) == region


def test_explicit_region_wins_and_unknown_location_uses_configured_default() -> None:
    settings = AutomaticRouting(country_header="Trusted-Country", default_region="eu")
    assert preferred_region(settings, Headers({"Otari-Region": "us", "Trusted-Country": "PT"})) == "us"
    assert preferred_region(settings, Headers({})) == "eu"


@pytest.mark.parametrize(
    "config",
    [
        {"default_region": "anywhere"},
        {"offerings": {"a": {"capacity": 0}}},
        {"country_header": "bad header"},
        {"typo": True},
        {"session_ttl_seconds": 0},
        {"session_ttl_seconds": 86401},
    ],
)
def test_invalid_automatic_settings_fail_at_load(config: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        GatewayConfig.model_validate({"automatic_routing": config})


def test_unpriced_provider_does_not_block_a_priced_provider_in_cooldown(router: AutomaticRouter) -> None:
    first = _rank(router)
    assert first.automatic is not None
    first.automatic.start(first.head).finish(unhealthy=True)
    assert _rank(router, costs={"us:model": Decimal(1), "eu:model": None}).head.instance == "us"


def test_other_regions_prices_do_not_drown_out_local_load(router: AutomaticRouter) -> None:
    plan = _plan()
    plan = replace(plan, attempts=[*plan.attempts, Attempt(3, "us2", LLMProvider.OPENAI, "model")])
    settings = AutomaticRouting.model_validate(
        {
            "offerings": {
                "us": {"region": "us"},
                "eu": {"region": "eu"},
                "us2": {"region": "us"},
            }
        }
    )
    costs = {"us:model": Decimal("0.01"), "eu:model": Decimal("0.000001"), "us2:model": Decimal("0.0101")}
    first = _rank(router, plan=plan, settings=settings, costs=costs, region="us")
    assert first.head.instance == "us"
    assert first.automatic is not None
    first.automatic.start(first.head)
    assert _rank(router, plan=plan, settings=settings, costs=costs, region="us").head.instance == "us2"


def test_session_requires_an_explicit_id(router: AutomaticRouter) -> None:
    plan = _rank(router, anchor="")
    assert plan.automatic is not None
    plan.automatic.start(plan.attempts[1]).finish(success=True)
    assert not router.affinities


def test_success_refreshes_session_idle_expiry(router: AutomaticRouter, now: list[float]) -> None:
    plan = _rank(router)
    assert plan.automatic is not None
    plan.automatic.start(plan.attempts[1]).finish(success=True)
    now[0] += 3500
    following = _rank(router)
    assert following.head.instance == "eu"
    assert following.automatic is not None
    following.automatic.start(following.head).finish(success=True)
    now[0] += 3500
    assert _rank(router).head.instance == "eu"
    now[0] += 101
    assert _rank(router).head.instance == "us"


def test_region_change_drops_previous_binding(router: AutomaticRouter) -> None:
    plan = _rank(router, region="eu")
    assert plan.automatic is not None
    plan.automatic.start(plan.head).finish(success=True)
    assert _rank(router, region="us").head.instance == "us"
    assert not router.affinities


def test_failed_binding_does_not_return_after_cooldown(router: AutomaticRouter, now: list[float]) -> None:
    plan = _rank(router)
    assert plan.automatic is not None
    plan.automatic.start(plan.attempts[1]).finish(success=True)
    plan.automatic.start(plan.attempts[1]).finish(unhealthy=True)
    now[0] += 16
    assert _rank(router).head.instance == "us"
    assert not router.affinities


def test_cancellation_preserves_binding_but_does_not_refresh_it(router: AutomaticRouter, now: list[float]) -> None:
    plan = _rank(router)
    assert plan.automatic is not None
    plan.automatic.start(plan.attempts[1]).finish(success=True)
    now[0] += 3500
    plan.automatic.start(plan.attempts[1]).finish()
    assert _rank(router).head.instance == "eu"
    now[0] += 101
    assert _rank(router).head.instance == "us"


def test_old_failure_does_not_erase_new_binding(router: AutomaticRouter) -> None:
    plan = _rank(router)
    assert plan.automatic is not None
    old = plan.automatic.start(plan.attempts[0])
    plan.automatic.start(plan.attempts[1]).finish(success=True)
    old.finish(unhealthy=True)
    assert _rank(router).head.instance == "eu"


def test_session_does_not_keep_unpriced_or_removed_provider(router: AutomaticRouter) -> None:
    plan = _rank(router)
    assert plan.automatic is not None
    plan.automatic.start(plan.attempts[1]).finish(success=True)
    assert _rank(router, costs={"us:model": Decimal(1), "eu:model": None}).head.instance == "us"
    assert not router.affinities
    plan.automatic.start(plan.attempts[1]).finish(success=True)
    assert _rank(router, plan=replace(_plan(), attempts=[_plan().head])).head.instance == "us"
    assert not router.affinities


def test_dispatch_credential_rebinding_has_its_own_health_and_no_false_affinity(router: AutomaticRouter) -> None:
    first = _rank(router)
    assert first.automatic is not None
    served = replace(first.head, instance="hosted", kwargs={"api_key": "hosted-secret"}, deployment_paid=True)
    observation = first.automatic.start(served)
    observation.finish(success=True)
    assert all(health.active == 0 for health in router.health.values())
    assert _rank(router, costs={"us:model": Decimal(2), "eu:model": Decimal(1)}).head.instance == "eu"
    first.automatic.forget(served)
    assert not router.affinities
    assert "hosted-secret" not in repr(router.health)
