"""Funding decisions for requests the deployment itself pays for.

The seam between the core and whichever build pays for a request that runs on
credentials the deployment owns. A request served from the caller's own
provider key never reaches this port: that caller is billed by their upstream
directly. Only deployment-paid traffic does, because only there does the
deployment owe the upstream bill and so has something to meter.

Billing is the overlay-only row of ``ARCHITECTURE.md``'s capability lines: the
core adapter is a Null Object, so a deployment can serve deployment-paid
traffic with no funding model at all, and nothing in the core needs to know
whether real billing exists anywhere.

Every method joins the caller's unit of work and commits nothing, so a hold,
the charge that consumes it, and whatever record the caller keeps of them
either all land or none do. That holds only where the caller's session is the
one the port was resolved against; see ``PortSessionDep`` in
``gateway.api.deps`` for which routes share it and which do not.

Each gate is a decision, not a reading: it either returns or raises
:class:`InsufficientFundsError`. Callers get no funding amounts to interpret,
so "this deployment does not bill" and "this organization is out of funds"
cannot be confused at a call site.
"""

import uuid
from decimal import Decimal
from typing import Protocol


class InsufficientFundsError(Exception):
    """Raised when an organization's funds cannot cover a deployment-paid request.

    The port owns this error so a caller can refuse a request without naming
    the funding model that produced it. An adapter may raise a subclass
    carrying its own wording; ``available`` is the amount the decision was made
    against, in USD, and is zero when the organization has no funding record at
    all.
    """

    def __init__(self, available: Decimal, message: str | None = None) -> None:
        super().__init__(message or f"Insufficient funds (${available:.2f}) to serve this request.")
        self.available = available


class BillingPort(Protocol):
    """What a build must answer to serve a request on deployment-paid credentials.

    Amounts are USD, and ``Decimal`` throughout for the reason the rest of the
    money path is (see ``src/gateway/AGENTS.md``, "Cost math").
    """

    async def apply_due_credit(self, *, organization_id: uuid.UUID) -> None:
        """Grant any credit the organization is owed by now.

        Idempotent: credit already granted for the current period is not
        granted twice, so a caller may invoke this as often as it likes.
        """
        ...

    async def require_funds_on_deposit(self, *, organization_id: uuid.UUID) -> None:
        """Refuse unless the organization has funds paid in.

        Counts what the organization has paid in and disregards holds, so it
        answers "may this organization be served at all", a question about the
        account rather than about what is currently in flight. It claims
        nothing: two callers can pass this gate on the same funds, and
        :meth:`hold` is what settles between them.

        Raises:
            InsufficientFundsError: If the organization has no funds paid in.

        """
        ...

    async def require_unheld_funds(self, *, organization_id: uuid.UUID) -> None:
        """Refuse unless the organization has funds no hold has claimed.

        The gate for a deployment-paid request with no upper-bound cost to
        hold. :meth:`hold` would wave such a request through for want of an
        amount, yet the deployment still owes the upstream bill, so an
        organization whose funds are entirely held must not be served on the
        strength of that.

        Raises:
            InsufficientFundsError: If every fund the organization has is held.

        """
        ...

    async def hold(self, *, organization_id: uuid.UUID, amount: Decimal) -> None:
        """Claim ``amount`` against the organization's unheld funds.

        Decided and claimed as one step, so overlapping requests cannot both be
        served on the same funds. Every hold is ended by :meth:`release_hold`,
        whether or not it is charged.

        Raises:
            InsufficientFundsError: If the unheld funds cannot cover ``amount``.

        """
        ...

    async def release_hold(self, *, organization_id: uuid.UUID, amount: Decimal) -> None:
        """Return a hold of ``amount`` to the organization's unheld funds.

        Not idempotent: whoever placed a hold decides once that it is over, and
        calls this exactly once for it.
        """
        ...

    async def charge(
        self,
        *,
        organization_id: uuid.UUID,
        amount: Decimal,
        description: str | None = None,
        actor_user_id: uuid.UUID | None = None,
        api_key_id: str | None = None,
    ) -> None:
        """Charge ``amount`` for a request already served.

        Records the charge even when the organization cannot cover it: the
        deployment has already paid for that request upstream, so dropping the
        charge would give the compute away. What is charged is bounded by the
        hold taken beforehand, and the gates above refuse the next request.

        ``description``, ``actor_user_id`` and ``api_key_id`` attribute the
        charge, so an adapter that keeps a ledger can trace it back to the
        request that incurred it.
        """
        ...
