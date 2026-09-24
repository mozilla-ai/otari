"""A mandate's hosted guardrail, as a check the request path runs like any in-process one.

Satisfies :class:`gateway.services.guardrails.InProcessGuardrail` by shape, so a
hosted check goes through the same ``mode`` and ``on_unavailable`` handling as
a definition this worker builds, and needs no failure branch of its own.
"""

import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from gateway.ports.hosted_guardrail_port import (
    HostedGuardrailPort,
    HostedGuardrailUnavailableError,
    HostedGuardrailUnfundedError,
    HostedGuardrailVerdict,
)
from gateway.services.guardrails import GuardrailsNotReachableError, GuardrailUnfundedError


@dataclass(frozen=True)
class HostedMandate:
    """Which mandate names a hosted guardrail, and which one it names."""

    mandate_id: uuid.UUID
    hosted_guardrail_id: uuid.UUID


class HostedGuardrailCheck:
    """One hosted mandate's check for one request.

    The idempotency key pairs the request with the mandate, so a request that
    checks twice (a retry, or a fallover to the next candidate) is charged once.
    """

    def __init__(
        self,
        port: HostedGuardrailPort,
        *,
        organization_id: uuid.UUID,
        workspace_id: uuid.UUID | None,
        mandate: HostedMandate,
        request_id: str,
    ) -> None:
        self._port = port
        self._organization_id = organization_id
        self._workspace_id = workspace_id
        self._mandate = mandate
        self._idempotency_key = f"{request_id}:{mandate.mandate_id}"

    async def check(self, prompt: str, **validate_kwargs: Any) -> HostedGuardrailVerdict:
        kwargs: Mapping[str, Any] = validate_kwargs
        try:
            return await self._port.evaluate(
                organization_id=self._organization_id,
                workspace_id=self._workspace_id,
                hosted_guardrail_id=self._mandate.hosted_guardrail_id,
                text=prompt,
                validate_kwargs=kwargs,
                idempotency_key=self._idempotency_key,
            )
        except HostedGuardrailUnfundedError as exc:
            msg = f"hosted guardrail {self._mandate.hosted_guardrail_id} refused: {exc}"
            raise GuardrailUnfundedError(msg) from exc
        except HostedGuardrailUnavailableError as exc:
            msg = f"hosted guardrail {self._mandate.hosted_guardrail_id} gave no verdict: {exc}"
            raise GuardrailsNotReachableError(msg) from exc
