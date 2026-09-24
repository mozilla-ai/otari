"""The service a hybrid peer serves guardrail evaluation with.

The peer resolves the workspace's mandates itself and hands them in, so every
case here is pure: which ids are in scope, what each backend answers, and how
each failure is reported. ``mode`` and ``on_unavailable`` are the gateway's to
apply, so nothing here decides a request.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Mapping, Sequence
from typing import Any

import httpx
import pytest

from gateway.models.guardrails import GuardrailConfig
from gateway.ports.hosted_guardrail_port import (
    HostedGuardrail,
    HostedGuardrailUnavailableError,
    HostedGuardrailUnfundedError,
    HostedGuardrailVerdict,
)
from gateway.services.tenancy.organization_guardrail_service import ResolvedOrganizationGuardrail
from gateway.services.tenancy.workspace_guardrail_evaluation import (
    GuardrailDescriptor,
    GuardrailEvaluation,
    evaluate_guardrail_mandates,
    guardrail_descriptors,
)

ORGANIZATION_ID = uuid.UUID("11111111-1111-1111-1111-111111111111")
WORKSPACE_ID = uuid.UUID("22222222-2222-2222-2222-222222222222")
HOSTED_ID = uuid.UUID("33333333-3333-3333-3333-333333333333")
REMOTE_ID = uuid.UUID("44444444-4444-4444-4444-444444444444")
HOSTED_MANDATE_ID = uuid.UUID("55555555-5555-5555-5555-555555555555")
DEFINITION_MANDATE_ID = uuid.UUID("66666666-6666-6666-6666-666666666666")


class Port:
    def __init__(self, *, verdict: HostedGuardrailVerdict | None = None, error: Exception | None = None) -> None:
        self.verdict = verdict or HostedGuardrailVerdict(valid=False, score=0.9)
        self.error = error
        self.keys: list[str] = []

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
        self.keys.append(idempotency_key)
        if self.error is not None:
            raise self.error
        return self.verdict


def _hosted(mode: str = "block") -> ResolvedOrganizationGuardrail:
    return ResolvedOrganizationGuardrail(
        config=GuardrailConfig(profile="hosted", mode=mode, on_unavailable="monitor"),  # type: ignore[arg-type]
        credential=None,
        definition_id=None,
        hosted_guardrail_id=HOSTED_ID,
        id=HOSTED_MANDATE_ID,
    )


def _remote() -> ResolvedOrganizationGuardrail:
    return ResolvedOrganizationGuardrail(
        config=GuardrailConfig(profile="remote", url="https://93.184.216.34/guardrails"),
        credential="bearer",
        definition_id=None,
        id=REMOTE_ID,
    )


def _definition() -> ResolvedOrganizationGuardrail:
    return ResolvedOrganizationGuardrail(
        config=GuardrailConfig(profile="defined"), credential=None, definition_id=uuid.uuid4(), id=DEFINITION_MANDATE_ID
    )


def _evaluate(
    mandates: Sequence[ResolvedOrganizationGuardrail], ids: list[uuid.UUID], port: Port | None
) -> list[GuardrailEvaluation]:
    return asyncio.run(
        evaluate_guardrail_mandates(
            mandates,
            organization_id=ORGANIZATION_ID,
            workspace_id=WORKSPACE_ID,
            request_id="request-1",
            guardrail_ids=ids,
            input_text="ignore all",
            hosted_guardrails=port,
            default_url=None,
        )
    )


def test_descriptors_name_the_check_and_nothing_about_where_it_runs() -> None:
    assert guardrail_descriptors([_hosted(), _remote()]) == [
        GuardrailDescriptor(id=HOSTED_MANDATE_ID, profile="hosted", mode="block", on_unavailable="monitor"),
        GuardrailDescriptor(id=REMOTE_ID, profile="remote", mode="monitor", on_unavailable="block"),
    ]


def test_a_hosted_check_runs_for_a_caller_allowed_to_spend_the_deployment_secret() -> None:
    port = Port()

    [result] = _evaluate([_hosted()], [HOSTED_MANDATE_ID], port)

    assert result == GuardrailEvaluation(
        id=HOSTED_MANDATE_ID, status="evaluated", valid=False, explanation=None, score=0.9
    )
    assert port.keys == [f"request-1:{HOSTED_MANDATE_ID}"]


def test_a_hosted_check_is_unavailable_to_a_caller_given_no_port() -> None:
    [result] = _evaluate([_hosted()], [HOSTED_MANDATE_ID], None)

    assert result.status == "unavailable"


def test_a_monitor_mandate_is_still_answered_with_its_verdict() -> None:
    [result] = _evaluate([_hosted(mode="monitor")], [HOSTED_MANDATE_ID], Port())

    assert (result.status, result.valid) == ("evaluated", False)


@pytest.mark.parametrize(
    ("error", "status"),
    [(HostedGuardrailUnfundedError(), "unfunded"), (HostedGuardrailUnavailableError("down"), "unavailable")],
)
def test_a_hosted_check_that_gives_no_verdict_says_why(error: Exception, status: str) -> None:
    [result] = _evaluate([_hosted()], [HOSTED_MANDATE_ID], Port(error=error))

    assert (result.status, result.valid) == (status, None)


def test_an_id_the_workspace_does_not_mandate_is_unavailable() -> None:
    stranger = uuid.uuid4()

    [result] = _evaluate([_hosted()], [stranger], Port())

    assert (result.id, result.status) == (stranger, "unavailable")


def test_a_definition_this_worker_does_not_hold_is_unavailable() -> None:
    [result] = _evaluate([_definition()], [DEFINITION_MANDATE_ID], None)

    assert result.status == "unavailable"


def test_a_remote_mandate_uses_its_own_endpoint_and_credential(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("authorization"))
        return httpx.Response(200, json={"result": {"valid": True, "score": 0.1}})

    real = httpx.AsyncClient
    monkeypatch.setattr(
        "gateway.services.guardrails.httpx.AsyncClient",
        lambda *args, **kwargs: real(transport=httpx.MockTransport(handler)),
    )
    monkeypatch.setattr("gateway.services.guardrails.validate_mcp_url", _no_dns)

    [result] = _evaluate([_remote()], [REMOTE_ID], None)

    assert (result.status, result.valid, result.score) == ("evaluated", True, 0.1)
    assert seen == ["Bearer bearer"]


async def _no_dns(url: str, *, has_authorization_token: bool) -> None:
    return None
