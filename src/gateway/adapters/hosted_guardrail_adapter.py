"""Core adapter for ``HostedGuardrailPort``: this build hosts no guardrails.

An organization here runs the guardrails it defines itself. There is nothing
deployment-owned to offer, so the list is empty and a check naming a hosted
guardrail is unavailable. An overlay binds an adapter over its own store.
"""

import uuid
from collections.abc import Mapping, Sequence
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from gateway.ports.hosted_guardrail_port import (
    HostedGuardrail,
    HostedGuardrailUnavailableError,
    HostedGuardrailVerdict,
)


class NullHostedGuardrailAdapter:
    """Core adapter: nothing is hosted, so nothing can be picked or run."""

    def __init__(self, session: AsyncSession | None) -> None:
        # Accepted to match the container's per-request factory; the answer never depends on it.
        del session

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
        msg = "this build hosts no guardrails"
        raise HostedGuardrailUnavailableError(msg)
