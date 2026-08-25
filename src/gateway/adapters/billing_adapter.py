"""Null Object billing adapter, for a deployment that bills nobody.

Satisfies :class:`gateway.ports.billing_port.BillingPort` with the honest
answer for a deployment that owes nothing: an operator running Otari pays their
own upstream bill for their own users, so there is nothing to meter, hold, or
charge.

Every method does nothing and neither gate ever refuses, so the request path
runs unchanged with no funding model in it. Budgets still bound what a request
may spend; they are a separate capability, enforced in
``gateway.services.budget_service``, and are unaffected by this.
"""

import uuid
from decimal import Decimal

from sqlalchemy.ext.asyncio import AsyncSession


class NullBillingAdapter:
    """``BillingPort`` adapter for a deployment that does not bill.

    Holds no state, so the request's database session is unused.
    """

    def __init__(self, session: AsyncSession | None) -> None:
        # Accepted to match the container's per-request factory; unused, because
        # there is no funding record to read or write.
        del session

    async def apply_due_credit(self, *, organization_id: uuid.UUID) -> None:
        return None

    async def require_funds_on_deposit(self, *, organization_id: uuid.UUID) -> None:
        # Returns rather than raises: nothing is owed here, so no request is
        # ever refused for want of funds.
        return None

    async def require_unheld_funds(self, *, organization_id: uuid.UUID) -> None:
        return None

    async def hold(self, *, organization_id: uuid.UUID, amount: Decimal) -> None:
        return None

    async def release_hold(self, *, organization_id: uuid.UUID, amount: Decimal) -> None:
        return None

    async def charge(
        self,
        *,
        organization_id: uuid.UUID,
        amount: Decimal,
        description: str | None = None,
        actor_user_id: uuid.UUID | None = None,
        api_key_id: str | None = None,
    ) -> None:
        return None
