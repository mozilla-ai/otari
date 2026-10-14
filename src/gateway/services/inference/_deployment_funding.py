"""The funding rule for a completion served on a credential the deployment pays for.

Such a request is admitted only when its organization has funds paid in, and
its settled cost is charged to the organization afterwards. No hold is placed:
concurrent requests can pass on the same funds, the charge allows the overdraft
that leaves, and the next request is refused.
"""

import uuid
from dataclasses import dataclass
from decimal import Decimal

from gateway.log_config import logger
from gateway.ports.billing_port import BillingPort


def _actor_user_id(user_id: str | None) -> uuid.UUID | None:
    """The billed user as the dashboard identity it names, where it names one.

    A key a member minted for themselves carries their identity's uuid as its
    user id; any other user id is free-form and attributes to no one.
    """
    if user_id is None:
        return None
    try:
        return uuid.UUID(user_id)
    except ValueError:
        return None


@dataclass(frozen=True)
class DeploymentCharge:
    """Who an admitted deployment-paid request is charged to, and the port that charges them.

    ``billing`` writes on the session it was resolved against, so the caller
    resolves it on the session whose commit settles the request.
    """

    billing: BillingPort
    organization_id: uuid.UUID
    api_key_id: str | None
    actor_user_id: uuid.UUID | None
    request_id: str | None

    def amount_due(self, cost: Decimal | None, *, model: str, expects_cost: bool) -> Decimal | None:
        """What a settled ``cost`` owes, or ``None`` when nothing is charged.

        ``expects_cost`` says the settlement should have priced something (a
        success, or a failure the provider reported usage for); a ``None`` cost
        then means the deployment served the request free, which is logged.
        """
        if cost is None:
            if expects_cost:
                logger.warning(
                    "Deployment-paid request settled with no price for model %s; organization %s is not charged "
                    "(request_id=%s). Price the models this build serves.",
                    model,
                    self.organization_id,
                    self.request_id,
                )
            return None
        if cost <= 0:
            return None
        return cost

    async def charge(self, amount: Decimal, *, description: str) -> None:
        """Charge ``amount`` to the organization, attributed to the request's key and user."""
        await self.billing.charge(
            organization_id=self.organization_id,
            amount=amount,
            description=description,
            actor_user_id=self.actor_user_id,
            api_key_id=self.api_key_id,
        )


async def admit_deployment_paid(
    billing: BillingPort,
    *,
    organization_id: uuid.UUID,
    api_key_id: str | None,
    user_id: str | None,
    request_id: str | None,
) -> DeploymentCharge:
    """Admit a deployment-paid request and return what settlement charges it through.

    Grants any credit the organization is owed, then refuses one with nothing
    paid in.

    Raises:
        InsufficientFundsError: If the organization has no funds paid in.

    """
    await billing.apply_due_credit(organization_id=organization_id)
    await billing.require_funds_on_deposit(organization_id=organization_id)
    return DeploymentCharge(
        billing=billing,
        organization_id=organization_id,
        api_key_id=api_key_id,
        actor_user_id=_actor_user_id(user_id),
        request_id=request_id,
    )
