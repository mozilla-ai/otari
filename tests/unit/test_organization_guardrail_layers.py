"""How an organization's mandated guardrails fold in with the caller's and a policy's.

The composition half of otari#654. `merge_guardrail_layers` is the one place
three layers meet, so the rules that matter are asserted directly on it rather
than through a request: an organization may add a check or tighten one and can
never weaken what is already there, the operator's policy is the outermost layer
and owns the endpoint where both name a profile, and a credential never travels
to an endpoint other than the one it was stored for.

The zero-rows case (`test_no_layer_asked_for_anything...`) is the requirement
#655 puts on every one of the four planes: a deployment that configures nothing
behaves as it did.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Literal, cast

from any_llm import LLMProvider

from gateway.api.routes._pipeline import RequestContext, merge_guardrail_layers
from gateway.core.config import GatewayConfig
from gateway.models.guardrails import GuardrailConfig
from gateway.services.routing import CompiledPlan
from gateway.services.tenancy.organization_guardrail_service import ResolvedOrganizationGuardrail
from gateway.types.attempt import Attempt


def _guardrail(
    profile: str,
    *,
    mode: Literal["block", "monitor"] = "block",
    on_unavailable: Literal["block", "monitor"] = "block",
    url: str | None = None,
) -> GuardrailConfig:
    return GuardrailConfig(profile=profile, mode=mode, on_unavailable=on_unavailable, url=url)


def _organization(guardrail: GuardrailConfig, *, credential: str | None = None) -> ResolvedOrganizationGuardrail:
    return ResolvedOrganizationGuardrail(config=guardrail, credential=credential)


def _ctx(*policy_guardrails: GuardrailConfig) -> RequestContext:
    plan: Any = None
    if policy_guardrails:
        plan = CompiledPlan(
            policy_name="p",
            attempts=[
                Attempt(
                    position=1,
                    instance="openai",
                    provider=LLMProvider.OPENAI,
                    model="m",
                    kwargs={"api_key": "sk-test"},
                )
            ],
            guardrails=list(policy_guardrails),
        )
    return RequestContext(
        config=GatewayConfig(),
        db=None,
        log_writer=cast(Any, None),
        hybrid_mode=False,
        route=None,
        user_token=None,
        api_key_id="key-1",
        user_id="user-1",
        rate_limit_info=None,
        reservation=None,
        started_at=time.monotonic(),
        workspace_id=uuid.uuid4(),
        plan=plan,
    )


def test_no_layer_asked_for_anything_leaves_the_request_exactly_as_it_was() -> None:
    """The zero-rows requirement: no organization entries, no policy, no change."""
    caller = [_guardrail("pii", mode="monitor")]

    unrouted = merge_guardrail_layers(_ctx(), caller, [])
    assert unrouted.configs is caller
    assert unrouted.credentials == {} and unrouted.mandated == frozenset()

    empty = merge_guardrail_layers(_ctx(), None, [])
    assert empty.configs is None
    assert empty.credentials == {} and empty.mandated == frozenset()


def test_an_organization_guardrail_runs_when_the_caller_asked_for_nothing() -> None:
    merged = merge_guardrail_layers(_ctx(), None, [_organization(_guardrail("prompt-injection"))])

    assert merged.configs is not None
    assert [g.profile for g in merged.configs] == ["prompt-injection"]
    assert merged.configs[0].mode == "block"
    assert merged.credentials == {}


def test_a_caller_cannot_weaken_what_the_organization_mandated() -> None:
    merged = merge_guardrail_layers(
        _ctx(),
        [_guardrail("prompt-injection", mode="monitor", on_unavailable="monitor")],
        [_organization(_guardrail("prompt-injection", mode="block", on_unavailable="block"))],
    )

    assert merged.configs is not None and len(merged.configs) == 1
    assert merged.configs[0].mode == "block"
    assert merged.configs[0].on_unavailable == "block"


def test_a_caller_may_tighten_what_the_organization_mandated() -> None:
    merged = merge_guardrail_layers(
        _ctx(),
        [_guardrail("prompt-injection", mode="block", on_unavailable="block")],
        [_organization(_guardrail("prompt-injection", mode="monitor", on_unavailable="monitor"))],
    )

    assert merged.configs is not None
    assert merged.configs[0].mode == "block"
    assert merged.configs[0].on_unavailable == "block"


def test_a_caller_may_add_their_own_guardrails_alongside_an_organization_mandate() -> None:
    merged = merge_guardrail_layers(
        _ctx(),
        [_guardrail("pii", mode="monitor")],
        [_organization(_guardrail("prompt-injection"))],
    )

    assert merged.configs is not None
    assert sorted(g.profile for g in merged.configs) == ["pii", "prompt-injection"]


def test_the_organization_owns_the_endpoint_for_a_profile_the_caller_also_named() -> None:
    """A caller cannot point a mandated check at a service of their choosing."""
    merged = merge_guardrail_layers(
        _ctx(),
        [_guardrail("prompt-injection", url="https://caller.example/guardrails")],
        [_organization(_guardrail("prompt-injection", url="https://org.example/guardrails"), credential="s3cret")],
    )

    assert merged.configs is not None
    assert merged.configs[0].url == "https://org.example/guardrails"
    assert merged.credentials == {"prompt-injection": "s3cret"}


def test_the_policy_layer_is_outermost_and_takes_the_endpoint_from_the_organization() -> None:
    """Where both mandate one profile, the operator's entry wins, so the credential is dropped.

    The credential was stored for the endpoint the organization named. Once the
    policy's URL has replaced it, carrying the secret along would be sending it
    somewhere it was never meant for.
    """
    merged = merge_guardrail_layers(
        _ctx(_guardrail("prompt-injection", url="https://operator.example/guardrails")),
        None,
        [_organization(_guardrail("prompt-injection", url="https://org.example/guardrails"), credential="s3cret")],
    )

    assert merged.configs is not None and len(merged.configs) == 1
    assert merged.configs[0].url == "https://operator.example/guardrails"
    assert merged.credentials == {}


def test_a_credential_survives_a_policy_that_mandates_a_different_profile() -> None:
    merged = merge_guardrail_layers(
        _ctx(_guardrail("pii")),
        None,
        [_organization(_guardrail("prompt-injection"), credential="s3cret")],
    )

    assert merged.configs is not None
    assert sorted(g.profile for g in merged.configs) == ["pii", "prompt-injection"]
    assert merged.credentials == {"prompt-injection": "s3cret"}


def test_the_strictest_of_all_three_layers_wins() -> None:
    merged = merge_guardrail_layers(
        _ctx(_guardrail("prompt-injection", mode="monitor", on_unavailable="monitor")),
        [_guardrail("prompt-injection", mode="monitor", on_unavailable="block")],
        [_organization(_guardrail("prompt-injection", mode="block", on_unavailable="monitor"))],
    )

    assert merged.configs is not None and len(merged.configs) == 1
    assert merged.configs[0].mode == "block", "the organization's block survives two monitor layers"
    assert merged.configs[0].on_unavailable == "block", "and so does the caller's own"
