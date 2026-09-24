"""A hosted-guardrail adapter that a test binds in place of the core one."""

import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from fastapi.testclient import TestClient

from gateway.ports.hosted_guardrail_port import (
    HostedGuardrail,
    HostedGuardrailPort,
    HostedGuardrailUnavailableError,
    HostedGuardrailVerdict,
)

LAKERA = HostedGuardrail(
    id=uuid.UUID("33333333-3333-3333-3333-333333333333"),
    name="Prompt injection",
    guardrail_name="lakera_guard",
    description="Lakera Guard, run by the deployment",
    price_per_check=Decimal("0.0005"),
)


@dataclass(frozen=True)
class Evaluation:
    """One ``evaluate`` call, as the adapter received it."""

    organization_id: uuid.UUID
    workspace_id: uuid.UUID | None
    hosted_guardrail_id: uuid.UUID
    text: str
    validate_kwargs: Mapping[str, Any]
    idempotency_key: str


class HostedGuardrails:
    """Offers :data:`LAKERA` to one organization and answers checks with ``verdict`` or ``error``."""

    def __init__(
        self,
        offered_to: uuid.UUID | None,
        *,
        verdict: HostedGuardrailVerdict | None = None,
        error: HostedGuardrailUnavailableError | None = None,
    ) -> None:
        self.offered_to = offered_to
        self.verdict = verdict or HostedGuardrailVerdict(valid=True)
        self.error = error
        self.asked: list[uuid.UUID | None] = []
        self.evaluations: list[Evaluation] = []

    async def list_hosted_guardrails(self, *, organization_id: uuid.UUID | None) -> Sequence[HostedGuardrail]:
        self.asked.append(organization_id)
        return [LAKERA] if organization_id == self.offered_to else []

    async def get_hosted_guardrail(
        self, *, organization_id: uuid.UUID, hosted_guardrail_id: uuid.UUID
    ) -> HostedGuardrail | None:
        return LAKERA if organization_id == self.offered_to and hosted_guardrail_id == LAKERA.id else None

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
        self.evaluations.append(
            Evaluation(organization_id, workspace_id, hosted_guardrail_id, text, dict(validate_kwargs), idempotency_key)
        )
        if self.error is not None:
            raise self.error
        if await self.get_hosted_guardrail(organization_id=organization_id, hosted_guardrail_id=hosted_guardrail_id):
            return self.verdict
        raise HostedGuardrailUnavailableError("not offered")


def bind_hosted_guardrails(client: TestClient, hosted: HostedGuardrails) -> None:
    """Binds ``hosted`` on the app under test."""
    container: Any = client.app.state.container  # type: ignore[attr-defined]
    container.bind(HostedGuardrailPort, lambda session: hosted)
