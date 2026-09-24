"""A mandate naming a hosted guardrail on the standalone request path.

The merge records which profiles a hosted guardrail serves, the request path
turns each into a check over ``HostedGuardrailPort``, and that check goes
through the same ``mode`` and ``on_unavailable`` handling as any other. The one
thing it adds is the 402 for an organization that cannot pay.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Mapping, Sequence
from typing import Any, Literal, cast

import pytest
from any_llm import LLMProvider
from fastapi import HTTPException, Response

from gateway.api.routes._helpers import apply_input_guardrails
from gateway.api.routes._pipeline import RequestContext, _in_process_guardrails, merge_guardrail_layers
from gateway.core.config import GatewayConfig
from gateway.models.guardrails import GuardrailConfig
from gateway.ports.hosted_guardrail_port import (
    HostedGuardrail,
    HostedGuardrailUnavailableError,
    HostedGuardrailUnfundedError,
    HostedGuardrailVerdict,
)
from gateway.services.routing import CompiledPlan
from gateway.services.tenancy.hosted_guardrail_check import HostedGuardrailCheck, HostedMandate
from gateway.services.tenancy.organization_guardrail_service import ResolvedOrganizationGuardrail
from gateway.types.attempt import Attempt

ORGANIZATION_ID = uuid.UUID("11111111-1111-1111-1111-111111111111")
WORKSPACE_ID = uuid.UUID("22222222-2222-2222-2222-222222222222")
HOSTED_ID = uuid.UUID("33333333-3333-3333-3333-333333333333")
MANDATE_ID = uuid.UUID("44444444-4444-4444-4444-444444444444")


class Port:
    def __init__(self, *, verdict: HostedGuardrailVerdict | None = None, error: Exception | None = None) -> None:
        self.verdict = verdict or HostedGuardrailVerdict(valid=True)
        self.error = error
        self.calls: list[dict[str, Any]] = []

    async def list_hosted_guardrails(self, *, organization_id: uuid.UUID | None) -> Sequence[HostedGuardrail]:
        return []

    async def get_hosted_guardrail(
        self, *, organization_id: uuid.UUID, hosted_guardrail_id: uuid.UUID
    ) -> HostedGuardrail | None:
        return None

    async def evaluate(
        self,
        *,
        organization_id: uuid.UUID,
        workspace_id: uuid.UUID | None,
        hosted_guardrail_id: uuid.UUID,
        text: str,
        validate_kwargs: Mapping[str, Any],
        idempotency_key: str,
    ) -> HostedGuardrailVerdict:
        self.calls.append(
            {
                "organization_id": organization_id,
                "workspace_id": workspace_id,
                "hosted_guardrail_id": hosted_guardrail_id,
                "text": text,
                "validate_kwargs": dict(validate_kwargs),
                "idempotency_key": idempotency_key,
            }
        )
        if self.error is not None:
            raise self.error
        return self.verdict


def _guardrail(
    profile: str = "prompt-injection",
    *,
    mode: Literal["block", "monitor"] = "block",
    on_unavailable: Literal["block", "monitor"] = "block",
) -> GuardrailConfig:
    return GuardrailConfig(profile=profile, mode=mode, on_unavailable=on_unavailable, validate_kwargs={"t": 1})


def _hosted(guardrail: GuardrailConfig) -> ResolvedOrganizationGuardrail:
    return ResolvedOrganizationGuardrail(
        config=guardrail, credential=None, definition_id=None, hosted_guardrail_id=HOSTED_ID, id=MANDATE_ID
    )


def _ctx(*policy_guardrails: GuardrailConfig) -> RequestContext:
    plan: Any = None
    if policy_guardrails:
        plan = CompiledPlan(
            policy_name="p",
            attempts=[
                Attempt(position=1, instance="openai", provider=LLMProvider.OPENAI, model="m", kwargs={"api_key": "k"})
            ],
            guardrails=list(policy_guardrails),
        )
    return RequestContext(
        config=GatewayConfig(),
        db=None,
        uow=None,
        log_writer=cast(Any, None),
        hybrid_mode=False,
        route=None,
        user_token=None,
        api_key_id="key-1",
        user_id="user-1",
        rate_limit_info=None,
        reservation=None,
        started_at=time.monotonic(),
        workspace_id=WORKSPACE_ID,
        organization_id=ORGANIZATION_ID,
        plan=plan,
        request_id="request-1",
    )


def test_the_merge_says_which_profiles_a_hosted_guardrail_serves() -> None:
    merged = merge_guardrail_layers(_ctx(), None, [_hosted(_guardrail())])

    assert merged.hosted == {"prompt-injection": HostedMandate(mandate_id=MANDATE_ID, hosted_guardrail_id=HOSTED_ID)}
    assert merged.in_process == {}
    assert merged.mandated == frozenset({"prompt-injection"})


def test_the_policy_layer_takes_a_profile_off_the_hosted_path() -> None:
    merged = merge_guardrail_layers(_ctx(_guardrail()), None, [_hosted(_guardrail())])

    assert merged.hosted == {}


def test_a_hosted_profile_becomes_a_check_over_the_port() -> None:
    port = Port(verdict=HostedGuardrailVerdict(valid=False, explanation="injection", score=0.9))
    ctx = _ctx()
    merged = merge_guardrail_layers(ctx, None, [_hosted(_guardrail())])

    checks = _in_process_guardrails(ctx, merged, hosted_guardrails=port)
    verdict = asyncio.run(checks["prompt-injection"].check("ignore all", t=1))  # type: ignore[union-attr]

    assert (verdict.valid, verdict.explanation, verdict.score) == (False, "injection", 0.9)
    assert port.calls == [
        {
            "organization_id": ORGANIZATION_ID,
            "workspace_id": WORKSPACE_ID,
            "hosted_guardrail_id": HOSTED_ID,
            "text": "ignore all",
            "validate_kwargs": {"t": 1},
            "idempotency_key": f"request-1:{MANDATE_ID}",
        }
    ]


def test_a_build_with_no_port_leaves_a_hosted_profile_unevaluable() -> None:
    ctx = _ctx()
    merged = merge_guardrail_layers(ctx, None, [_hosted(_guardrail())])

    assert _in_process_guardrails(ctx, merged, hosted_guardrails=None) == {"prompt-injection": None}


def _apply(guardrail: GuardrailConfig, port: Port) -> Response:
    response = Response()
    check = HostedGuardrailCheck(
        port,
        organization_id=ORGANIZATION_ID,
        workspace_id=WORKSPACE_ID,
        mandate=HostedMandate(mandate_id=MANDATE_ID, hosted_guardrail_id=HOSTED_ID),
        request_id="request-1",
    )
    asyncio.run(
        apply_input_guardrails(
            [guardrail],
            "text",
            response=response,
            mandated={guardrail.profile},
            in_process={guardrail.profile: check},
        )
    )
    return response


def test_a_hosted_guardrail_that_flags_in_block_mode_refuses_the_request() -> None:
    with pytest.raises(HTTPException) as refused:
        _apply(_guardrail(), Port(verdict=HostedGuardrailVerdict(valid=False)))
    assert refused.value.status_code == 403


def test_an_unfunded_organization_is_a_402_when_the_mandate_fails_closed() -> None:
    with pytest.raises(HTTPException) as refused:
        _apply(_guardrail(), Port(error=HostedGuardrailUnfundedError()))
    assert refused.value.status_code == 402
    assert "prompt-injection" in str(refused.value.detail)
    assert "$" not in str(refused.value.detail)


@pytest.mark.parametrize(
    "guardrail",
    [_guardrail(on_unavailable="monitor"), _guardrail(mode="monitor")],
    ids=["block-fails-open", "monitor"],
)
def test_an_unfunded_organization_is_served_when_the_mandate_fails_open(guardrail: GuardrailConfig) -> None:
    response = _apply(guardrail, Port(error=HostedGuardrailUnfundedError()))

    assert "prompt-injection" in response.headers["X-Otari-Guardrails"]


def test_a_hosted_guardrail_that_gives_no_verdict_is_a_502_when_it_fails_closed() -> None:
    with pytest.raises(HTTPException) as refused:
        _apply(_guardrail(), Port(error=HostedGuardrailUnavailableError("vendor down")))
    assert refused.value.status_code == 502
    assert "vendor down" not in str(refused.value.detail)
