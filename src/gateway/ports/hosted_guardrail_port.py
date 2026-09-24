"""Deployment-owned guardrails an organization may pick for its own mandates.

Core guardrails are organization-scoped: a definition and the mandate that runs
it both belong to one organization. A hosted guardrail is the other shape. The
deployment configures it once, with a vendor secret of its own, and every
organization it is offered to may name it in a mandate. Storing, running and
metering one is a hosted service's work, which is the hard boundary that earns
this port (``ARCHITECTURE.md``, rule 7).

The port never hands out a secret or a guardrail's arguments. It lists what may
be picked, and it runs a check. So a hosted guardrail's secret stays in the
process that stores it, whichever gateway asked.

The core adapter offers nothing, so the plain build behaves as if the port did
not exist.

Stability: this interface is not frozen while Otari is pre-1.0. Overlay authors
should pin a released tag and expect the shape to move.
"""

import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Protocol


@dataclass(frozen=True)
class HostedGuardrail:
    """A hosted guardrail as an organization picking it sees it.

    ``guardrail_name`` is the any-guardrail class that runs it, so a picker can
    describe its parameters from the built-in catalog. ``price_per_check`` is
    what one evaluated check costs the organization, ``None`` where the build
    does not charge for it.
    """

    id: uuid.UUID
    name: str
    guardrail_name: str
    description: str | None
    price_per_check: Decimal | None


@dataclass(frozen=True)
class HostedGuardrailVerdict:
    """What one hosted check answered, in the shape an in-process guardrail answers."""

    valid: bool
    explanation: str | None = None
    score: float | None = None


class HostedGuardrailUnavailableError(Exception):
    """Raised when a hosted check produced no verdict.

    The guardrail is not offered to the organization, is disabled or gone, or
    its vendor failed. The mandate's ``on_unavailable`` decides what the
    request does next. The message may be logged and is never shown to a caller.
    """


class HostedGuardrailUnfundedError(HostedGuardrailUnavailableError):
    """Raised when the organization cannot pay for the check, before the vendor is called.

    A kind of unavailable, so a caller that only knows "no verdict" still
    handles it, and a caller that tells the two apart can answer 402.
    """

    def __init__(self, message: str = "organization funds exhausted") -> None:
        super().__init__(message)


class HostedGuardrailPort(Protocol):
    """What a build answers to offer and run guardrails it hosts for every organization."""

    async def list_hosted_guardrails(self, *, organization_id: uuid.UUID | None) -> Sequence[HostedGuardrail]:
        """Return the hosted guardrails ``organization_id`` may pick.

        ``None`` asks for the deployment-wide list. Answered from the build's own
        store; it never calls a vendor.
        """
        ...

    async def get_hosted_guardrail(
        self, *, organization_id: uuid.UUID, hosted_guardrail_id: uuid.UUID
    ) -> HostedGuardrail | None:
        """Return one hosted guardrail if ``organization_id`` may pick it, else ``None``."""
        ...

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
        """Run one hosted check on ``text`` for ``organization_id``.

        Eligibility is checked again on every call, because nothing else scopes a
        hosted guardrail to an organization. ``workspace_id`` is attribution only.

        The build meters the check. It refuses an organization that cannot pay
        before it calls the vendor, charges only for a check that returned a
        verdict, and charges at most once per ``idempotency_key``, so a retried
        request is not charged twice.

        Raises:
            HostedGuardrailUnfundedError: If the organization cannot pay for the check.
            HostedGuardrailUnavailableError: If the check produced no verdict for any other reason.

        """
        ...
