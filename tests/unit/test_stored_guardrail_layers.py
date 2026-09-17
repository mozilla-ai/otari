"""Where a deployment's own stored definitions sit among the other guardrail layers.

`merge_guardrail_layers` is the one place the layers meet, so the rules are
asserted on it directly rather than through a request. What matters here is that
a stored definition is a mandate like the other two, that a caller who names the
same profile is checked once and cannot weaken it, and that the operator's
routing policy is still the outermost layer and still owns the endpoint.

The zero-rows case is the requirement every plane carries: a deployment that
stores nothing behaves exactly as it did.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Literal, cast

import pytest
from any_llm import LLMProvider

from gateway.api.routes._pipeline import (
    RequestContext,
    _resolve_workspace_guardrails,
    merge_guardrail_layers,
)
from gateway.core.config import GatewayConfig
from gateway.models.guardrails import GuardrailConfig
from gateway.services.routing import CompiledPlan
from gateway.services.tenancy.organization_guardrail_service import ResolvedOrganizationGuardrail
from gateway.types.attempt import Attempt

_ANY_WORKSPACE = uuid.uuid4()


def _guardrail(
    profile: str,
    *,
    mode: Literal["block", "monitor"] = "block",
    on_unavailable: Literal["block", "monitor"] = "block",
    url: str | None = None,
) -> GuardrailConfig:
    return GuardrailConfig(profile=profile, mode=mode, on_unavailable=on_unavailable, url=url)


def _ctx(
    *policy_guardrails: GuardrailConfig,
    hybrid_mode: bool = False,
    workspace_id: uuid.UUID | None = _ANY_WORKSPACE,
) -> RequestContext:
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
        hybrid_mode=hybrid_mode,
        route=None,
        user_token=None,
        api_key_id="key-1",
        user_id="user-1",
        rate_limit_info=None,
        reservation=None,
        started_at=time.monotonic(),
        workspace_id=workspace_id,
        plan=plan,
    )


def test_a_deployment_that_stores_nothing_leaves_the_request_exactly_as_it_was() -> None:
    caller = [_guardrail("pii", mode="monitor")]

    merged = merge_guardrail_layers(_ctx(), caller, [], [])

    assert merged.configs is caller
    assert merged.mandated == frozenset()


def test_a_stored_definition_runs_without_the_caller_asking() -> None:
    """The whole point: an enabled definition checks a request that named nothing."""
    merged = merge_guardrail_layers(_ctx(), None, [], [_guardrail("prompt-injection")])

    assert merged.configs is not None
    assert [g.profile for g in merged.configs] == ["prompt-injection"]
    assert merged.mandated == frozenset({"prompt-injection"})


def test_a_caller_naming_the_same_profile_is_checked_once() -> None:
    """Union by profile, so the vendor is called once rather than twice."""
    caller = [_guardrail("prompt-injection", mode="monitor")]

    merged = merge_guardrail_layers(_ctx(), caller, [], [_guardrail("prompt-injection")])

    assert merged.configs is not None
    assert len(merged.configs) == 1


def test_a_caller_cannot_weaken_a_stored_definition() -> None:
    caller = [_guardrail("prompt-injection", mode="monitor", on_unavailable="monitor")]

    merged = merge_guardrail_layers(_ctx(), caller, [], [_guardrail("prompt-injection")])

    assert merged.configs is not None
    assert merged.configs[0].mode == "block"
    assert merged.configs[0].on_unavailable == "block"


def test_a_caller_may_still_tighten_one() -> None:
    """Stricter wins whichever layer asked for it, so an observing definition can be enforced."""
    caller = [_guardrail("prompt-injection", mode="block")]
    stored = [_guardrail("prompt-injection", mode="monitor", on_unavailable="monitor")]

    merged = merge_guardrail_layers(_ctx(), caller, [], stored)

    assert merged.configs is not None
    assert merged.configs[0].mode == "block"


def test_a_caller_cannot_point_a_stored_definition_somewhere_else() -> None:
    """The stored entry owns the endpoint, and it names none, so the check runs in process."""
    caller = [_guardrail("prompt-injection", url="https://mine.example")]

    merged = merge_guardrail_layers(_ctx(), caller, [], [_guardrail("prompt-injection")])

    assert merged.configs is not None
    assert merged.configs[0].url is None
    assert "prompt-injection" in merged.mandated


def test_a_stored_definition_beats_an_organization_entry_of_the_same_name() -> None:
    """The deployment operator owns the gateway, so the local build wins the endpoint.

    The organization's credential goes with it, for the reason a policy takeover
    drops one: it was stored for the endpoint that entry named.
    """
    organization = [
        ResolvedOrganizationGuardrail(
            config=_guardrail("prompt-injection", url="https://org.example"), credential="bearer"
        )
    ]

    merged = merge_guardrail_layers(_ctx(), None, organization, [_guardrail("prompt-injection")])

    assert merged.configs is not None
    assert merged.configs[0].url is None
    assert merged.credentials == {}


def test_a_routing_policy_is_still_the_outermost_layer() -> None:
    """The operator wrote both, and the policy names an endpoint on purpose."""
    policy = _guardrail("prompt-injection", url="https://policy.example")

    merged = merge_guardrail_layers(_ctx(policy), None, [], [_guardrail("prompt-injection")])

    assert merged.configs is not None
    assert merged.configs[0].url == "https://policy.example"


def test_an_organization_entry_of_another_name_is_untouched() -> None:
    """Layers compose; they do not replace each other."""
    organization = [ResolvedOrganizationGuardrail(config=_guardrail("pii"), credential="bearer")]

    merged = merge_guardrail_layers(_ctx(), None, organization, [_guardrail("prompt-injection")])

    assert merged.configs is not None
    assert sorted(g.profile for g in merged.configs) == ["pii", "prompt-injection"]
    assert merged.credentials == {"pii": "bearer"}


class _Adapter:
    """Only the one method ``_resolve_workspace_guardrails`` uses to refuse."""

    def error(self, status: int, detail: str, _kind: object) -> Exception:
        return AssertionError(f"{status}: {detail}")


@pytest.mark.asyncio
async def test_hybrid_mode_enforces_no_stored_definition() -> None:
    """The store is not mounted there and the loader builds nothing, so there is nothing to read."""
    assert await _resolve_workspace_guardrails(cast(Any, _Adapter()), _ctx(hybrid_mode=True)) == []


@pytest.mark.asyncio
async def test_a_request_with_no_workspace_is_refused_rather_than_served_unchecked() -> None:
    """Fails closed for the reason the organization resolve beside it does.

    Unreachable today, since the workspace always resolves. Pinned because what
    it guards is an enforcement decision: the day it stops holding is the day a
    request that should have been checked would be served.
    """
    with pytest.raises(AssertionError, match="500"):
        await _resolve_workspace_guardrails(cast(Any, _Adapter()), _ctx(workspace_id=None))
