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
    """Provide a mutable monotonic clock for expiry and cooldown assertions."""
    return [100.0]


@pytest.fixture
def router(now: list[float]) -> AutomaticRouter:
    """Build an isolated router whose clock advances only when the test changes it."""
    return AutomaticRouter(clock=lambda: now[0])


def _plan() -> CompiledPlan:
    """Build two same-model offerings with distinct US and EU credentials."""
    return CompiledPlan(
        policy_name="acme/model",
        attempts=[
            Attempt(1, "us", LLMProvider.OPENAI, "model", {"api_key": "us-secret"}),
            Attempt(2, "eu", LLMProvider.OPENAI, "model", {"api_key": "eu-secret"}),
        ],
    )


def _rank(router: AutomaticRouter, **changes: Any) -> CompiledPlan:
    """Rank the two-offering fixture with explicit signal and session overrides."""
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
    """Region preference selects a matching offering and absent preference preserves tied catalog order."""
    assert _rank(router, region=region).head.instance == expected


def test_unknown_prices_are_never_free(router: AutomaticRouter) -> None:
    """Missing prices rank after known prices, while an explicit zero remains a valid free rate."""
    assert _rank(router, costs={"us:model": None, "eu:model": Decimal(1)}).head.instance == "eu"
    assert _rank(router, costs={"us:model": None, "eu:model": None}).head.instance == "us"
    assert _rank(router, costs={"us:model": Decimal(0), "eu:model": Decimal(1)}).head.instance == "us"


def test_concurrent_load_avoids_busy_provider_and_releases_on_cancel(router: AutomaticRouter) -> None:
    """Active calls affect ranking and repeated cleanup releases their load only once."""
    first = _rank(router)
    assert first.automatic is not None
    observation = first.automatic.start(first.head)
    assert _rank(router).head.instance == "eu"
    observation.finish()
    observation.finish()
    assert _rank(router).head.instance == "us"
    assert all(health.active == 0 for health in router.health.values())


def test_capacity_normalizes_load(router: AutomaticRouter) -> None:
    """The configured capacity scales the load penalty when comparing otherwise eligible offerings."""
    first = _rank(router)
    assert first.automatic is not None
    first.automatic.start(first.head)
    settings = AutomaticRouting.model_validate({"offerings": {"us": {"capacity": 100}}})
    costs = {"us:model": Decimal("0.01"), "eu:model": Decimal("0.012")}
    assert _rank(router, costs=costs).head.instance == "eu"
    assert _rank(router, costs=costs, settings=settings).head.instance == "us"


def test_latency_expires_and_does_not_mix_streaming(router: AutomaticRouter, now: list[float]) -> None:
    """Latency affects its own response mode only and loses influence after its observation expires."""
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
    """Cooldown temporarily outweighs regional preference and expires without a successful probe."""
    first = _rank(router, region="us")
    assert first.automatic is not None
    first.automatic.start(first.head).finish(unhealthy=True)
    assert _rank(router, region="us").head.instance == "eu"
    now[0] += 16
    assert _rank(router, region="us").head.instance == "us"


def test_session_is_scoped_expires_and_follows_serving_provider(router: AutomaticRouter, now: list[float]) -> None:
    """Affinity follows the serving credential, stays isolated and expires without storing raw identifiers."""
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
    """An eligible session binding takes precedence over a cheaper alternative."""
    first = _rank(router)
    assert first.automatic is not None
    first.automatic.start(first.attempts[1]).finish(success=True)
    assert _rank(router, costs={"us:model": Decimal(1), "eu:model": Decimal(2)}).head.instance == "eu"


def test_rotated_credential_does_not_inherit_affinity(router: AutomaticRouter) -> None:
    """Changing a provider key invalidates affinity even when the offering name is unchanged."""
    first = _rank(router)
    assert first.automatic is not None
    first.automatic.start(first.attempts[1]).finish(success=True)
    plan = _plan()
    plan = replace(plan, attempts=[plan.attempts[0], replace(plan.attempts[1], kwargs={"api_key": "rotated"})])
    assert _rank(router, plan=plan).head.instance == "us"


def test_rank_before_capping_and_renumber_attempts(router: AutomaticRouter) -> None:
    """Ranking considers the whole plan before retaining and renumbering the best five attempts."""
    plan = replace(_plan(), attempts=[Attempt(i, f"p{i}", LLMProvider.OPENAI, "model") for i in range(1, 9)])
    ranked = _rank(router, plan=plan, costs={f"p{i}:model": Decimal(9 - i) for i in range(1, 9)})
    assert ranked.head.instance == "p8"
    assert [a.position for a in ranked.attempts] == [1, 2, 3, 4, 5]
    assert [a.selection_reason for a in ranked.attempts] == ["catalog", *(["on_failure"] * 4)]


@pytest.mark.parametrize("country,region", [("US", "us"), ("PT", "eu"), ("DE", "eu"), ("GB", None), ("XX", None)])
def test_country_requires_explicitly_trusted_header(country: str, region: str | None) -> None:
    """Country hints influence region only when the deployment names their trusted header."""
    headers = Headers({"Trusted-Country": country})
    assert preferred_region(AutomaticRouting(), headers) is None
    assert preferred_region(AutomaticRouting(country_header="Trusted-Country"), headers) == region


def test_explicit_region_wins_and_unknown_location_uses_configured_default() -> None:
    """Explicit region overrides country inference, and absent location uses the deployment default."""
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
    """Invalid regions, capacities, headers, lifetimes and unknown settings fail config validation."""
    with pytest.raises(ValidationError):
        GatewayConfig.model_validate({"automatic_routing": config})


def test_unpriced_provider_does_not_block_a_priced_provider_in_cooldown(router: AutomaticRouter) -> None:
    """Known pricing takes precedence over an unpriced alternative even during cooldown."""
    first = _rank(router)
    assert first.automatic is not None
    first.automatic.start(first.head).finish(unhealthy=True)
    assert _rank(router, costs={"us:model": Decimal(1), "eu:model": None}).head.instance == "us"


def test_other_regions_prices_do_not_drown_out_local_load(router: AutomaticRouter) -> None:
    """Cost normalization within a region preserves meaningful load differences between local offerings."""
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
    """Successful calls without a conversation ID never create session affinity."""
    plan = _rank(router, anchor="")
    assert plan.automatic is not None
    plan.automatic.start(plan.attempts[1]).finish(success=True)
    assert not router.affinities


def test_success_refreshes_session_idle_expiry(router: AutomaticRouter, now: list[float]) -> None:
    """Successful reuse extends the idle lifetime, which expires after the last success."""
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
    """Moving regions removes the old binding instead of retaining a conflicting preference."""
    plan = _rank(router, region="eu")
    assert plan.automatic is not None
    plan.automatic.start(plan.head).finish(success=True)
    assert _rank(router, region="us").head.instance == "us"
    assert not router.affinities


def test_failed_binding_does_not_return_after_cooldown(router: AutomaticRouter, now: list[float]) -> None:
    """A provider failure removes affinity permanently until a later success establishes it again."""
    plan = _rank(router)
    assert plan.automatic is not None
    plan.automatic.start(plan.attempts[1]).finish(success=True)
    plan.automatic.start(plan.attempts[1]).finish(unhealthy=True)
    now[0] += 16
    assert _rank(router).head.instance == "us"
    assert not router.affinities


def test_cancellation_preserves_binding_but_does_not_refresh_it(router: AutomaticRouter, now: list[float]) -> None:
    """Cancellation leaves existing affinity intact without extending its idle lifetime."""
    plan = _rank(router)
    assert plan.automatic is not None
    plan.automatic.start(plan.attempts[1]).finish(success=True)
    now[0] += 3500
    plan.automatic.start(plan.attempts[1]).finish()
    assert _rank(router).head.instance == "eu"
    now[0] += 101
    assert _rank(router).head.instance == "us"


def test_old_failure_does_not_erase_new_binding(router: AutomaticRouter) -> None:
    """A late failure from an earlier attempt cannot erase a newer successful binding."""
    plan = _rank(router)
    assert plan.automatic is not None
    old = plan.automatic.start(plan.attempts[0])
    plan.automatic.start(plan.attempts[1]).finish(success=True)
    old.finish(unhealthy=True)
    assert _rank(router).head.instance == "eu"


def test_session_does_not_keep_unpriced_or_removed_provider(router: AutomaticRouter) -> None:
    """Affinity is discarded when its offering disappears or loses pricing beside a priced alternative."""
    plan = _rank(router)
    assert plan.automatic is not None
    plan.automatic.start(plan.attempts[1]).finish(success=True)
    assert _rank(router, costs={"us:model": Decimal(1), "eu:model": None}).head.instance == "us"
    assert not router.affinities
    plan.automatic.start(plan.attempts[1]).finish(success=True)
    assert _rank(router, plan=replace(_plan(), attempts=[_plan().head])).head.instance == "us"
    assert not router.affinities


def test_dispatch_credential_rebinding_has_its_own_health_and_no_false_affinity(router: AutomaticRouter) -> None:
    """Dispatch-time credentials have separate private health and cannot bind the original offering."""
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


def test_bedrock_rebuilt_client_config_preserves_session_and_health(router: AutomaticRouter) -> None:
    """Recreated Bedrock transport objects retain the same credential affinity and health history."""
    from gateway.services.provider_kwargs import get_provider_kwargs

    config = GatewayConfig(
        providers={
            "eu": {
                "provider": "bedrock",
                "client_args": {
                    "aws_access_key_id": "account",
                    "aws_secret_access_key": "secret",
                    "region_name": "eu-west-1",
                },
            }
        }
    )

    def plan() -> CompiledPlan:
        """Rebuild Bedrock provider kwargs with a fresh SDK transport configuration."""
        original = _plan()
        bedrock = replace(
            original.attempts[1],
            provider=LLMProvider.BEDROCK,
            kwargs=get_provider_kwargs(config, LLMProvider.BEDROCK, "eu"),
        )
        return replace(original, attempts=[original.head, bedrock])

    first = _rank(router, plan=plan())
    assert first.automatic is not None
    first.automatic.start(first.attempts[1]).finish(success=True)
    following = _rank(router, plan=plan())
    assert following.head.instance == "eu"
    assert following.automatic is not None
    following.automatic.start(following.head).finish(unhealthy=True)
    assert len(router.health) == 1
    assert _rank(router, plan=plan()).head.instance == "us"


def test_vertex_rebuilt_credentials_preserve_affinity_but_key_rotation_does_not(router: AutomaticRouter) -> None:
    """Equivalent Vertex credentials retain affinity, while location or signing-key changes invalidate it."""
    from unittest.mock import Mock

    from google.oauth2.service_account import Credentials

    def plan(key_id: str = "key-one", location: str = "europe-west1") -> CompiledPlan:
        """Rebuild Vertex credentials for the requested signer and location."""
        credentials = Credentials(  # type: ignore[no-untyped-call]
            signer=Mock(key_id=key_id),
            service_account_email="service@example.test",
            token_uri="https://oauth2.googleapis.com/token",
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        original = _plan()
        vertex = replace(
            original.attempts[1],
            provider=LLMProvider.VERTEXAI,
            kwargs={"client_args": {"credentials": credentials, "project": "project-one", "location": location}},
        )
        return replace(original, attempts=[original.head, vertex])

    first = _rank(router, plan=plan())
    assert first.automatic is not None
    first.automatic.start(first.attempts[1]).finish(success=True)
    assert _rank(router, plan=plan()).head.instance == "eu"
    assert _rank(router, plan=plan(location="us-central1")).head.instance == "us"
    first.automatic.start(first.attempts[1]).finish(success=True)
    assert _rank(router, plan=plan(key_id="rotated-key")).head.instance == "us"


def test_endpoint_and_nested_auth_changes_do_not_share_a_candidate(router: AutomaticRouter) -> None:
    """Endpoint and nested authorization changes produce distinct credential fingerprints."""
    original = _plan().head
    original = replace(
        original,
        kwargs={"api_base": "https://one.example.test", "client_args": {"default_headers": {"Authorization": "first"}}},
    )
    assert router.candidate_key(original) != router.candidate_key(
        replace(original, kwargs={**original.kwargs, "api_base": "https://two.example.test"})
    )
    assert router.candidate_key(original) != router.candidate_key(
        replace(original, kwargs={**original.kwargs, "client_args": {"default_headers": {"Authorization": "second"}}})
    )


def test_unknown_credential_objects_cannot_inherit_affinity(router: AutomaticRouter) -> None:
    """Unsupported credential objects receive non-reusable identities rather than unsafe shared affinity."""
    candidate = replace(_plan().head, kwargs={"client_args": {"credentials": object()}})
    assert router.candidate_key(candidate) != router.candidate_key(candidate)


def test_own_key_leads_a_cheaper_closer_offering(router: AutomaticRouter) -> None:
    """An organization's priced, healthy own key outranks cost and region; the rest stay ranked after it."""
    costs = {"us:model": Decimal("0.01"), "eu:model": Decimal("0.50")}
    plan = _rank(router, costs=costs, region="us", own=frozenset({"eu:model"}))
    assert [attempt.instance for attempt in plan.attempts] == ["eu", "us"]
    assert _rank(router, costs=costs, region="us").head.instance == "us"


def test_own_key_yields_when_cooling_down_or_unpriced(router: AutomaticRouter) -> None:
    """The deployment's offering serves while the own key is in cooldown or has no known price."""
    first = _rank(router, own=frozenset({"us:model"}))
    assert first.automatic is not None
    assert first.head.instance == "us"
    first.automatic.start(first.head).finish(unhealthy=True)
    assert _rank(router, own=frozenset({"us:model"})).head.instance == "eu"
    unpriced = {"us:model": None, "eu:model": Decimal("0.01")}
    assert _rank(AutomaticRouter(), costs=unpriced, own=frozenset({"us:model"})).head.instance == "eu"


def test_session_on_a_deployment_offering_yields_to_a_healthy_own_key(router: AutomaticRouter) -> None:
    """A binding to an offering the deployment pays for does not hold the conversation off the own key."""
    first = _rank(router)
    assert first.automatic is not None
    first.automatic.start(first.attempts[1]).finish(success=True)
    assert _rank(router).head.instance == "eu"
    assert _rank(router, own=frozenset({"us:model"})).head.instance == "us"


def test_session_on_one_own_key_holds_over_another(router: AutomaticRouter) -> None:
    """Between two own keys, the binding keeps the conversation where its prompt cache is."""
    own = frozenset({"us:model", "eu:model"})
    first = _rank(router, own=own)
    assert first.automatic is not None
    first.automatic.start(first.attempts[1]).finish(success=True)
    assert _rank(router, own=own, costs={"us:model": Decimal(1), "eu:model": Decimal(2)}).head.instance == "eu"
