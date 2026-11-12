"""The funding rule for a completion served on a credential the deployment pays for.

Such a request is admitted only when its organization has funds paid in, and
its settled cost is charged to the organization afterwards. No hold is placed:
concurrent requests can pass on the same funds, the charge allows the overdraft
that leaves, and the next request is refused.
"""

import uuid
from collections.abc import Awaitable, Callable, Iterable, Mapping
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from gateway.core.database import DATABASE_ERRORS, describe_database_error
from gateway.core.settings.inference import (
    DEFAULT_CLOSED_MODEL_FAMILIES,
    DEFAULT_CLOSED_MODEL_PROVIDERS,
    DEFAULT_OPEN_MODEL_FAMILIES,
)
from gateway.log_config import logger
from gateway.ports.billing_port import BilledModel, BillingPort, ToolUse
from gateway.services.provider_kwargs import provider_key
from gateway.types.closed_models import ClosedModels

if TYPE_CHECKING:
    from any_llm import LLMProvider


def actor_user_id_for(user_id: str | None) -> uuid.UUID | None:
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


# Which models no open-weight plan covers. Everything not ruled closed is treated
# as open-weight: the closed labs are few and change slowly, where a list of open
# models would trail every release. Set from the deployment's settings at startup
# (``configure_closed_models``), since a model is billed deep in code that holds no config.
_closed = ClosedModels.of(
    providers=DEFAULT_CLOSED_MODEL_PROVIDERS,
    closed_families=DEFAULT_CLOSED_MODEL_FAMILIES,
    open_families=DEFAULT_OPEN_MODEL_FAMILIES,
)


def configure_closed_models(
    *, providers: Iterable[str], closed_families: Iterable[str], open_families: Iterable[str]
) -> None:
    """Set which models count as closed, from the deployment's settings."""
    global _closed
    _closed = ClosedModels.of(providers=providers, closed_families=closed_families, open_families=open_families)


def is_closed_model(provider: str, model: str) -> bool:
    """Whether ``model`` on ``provider`` (a wire name) is closed under the deployment's settings."""
    return _closed.is_closed(provider, model)


def billed_model(provider: "str | LLMProvider", model: str) -> BilledModel:
    """The model a deployment-paid call on ``provider`` is billed as, open-weight unless it is a closed one.

    Decided from a short list of closed providers and model families
    (:func:`is_closed_model`), so a new open model is covered without waiting
    on any catalog to list it.
    """
    wire_name = provider_key(provider)
    return BilledModel(provider=wire_name, model=model, open_weights=not is_closed_model(wire_name, model))


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
    billed: BilledModel | None
    """The model the request is served on, or ``None`` for a gateway tool's call."""
    concurrency_limit: int | None = None
    """How many of the organization's calls may be in flight at once, as the port answered; ``None`` for no limit."""

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

    async def charge(self, amount: Decimal, *, tools: Mapping[str, ToolUse], description: str) -> None:
        """Charge ``amount`` to the organization, attributed to the request's key and user.

        ``tools`` is the gateway-run tool calls ``amount`` pays for, as at :meth:`BillingPort.charge`.
        """
        await self.billing.charge(
            organization_id=self.organization_id,
            amount=amount,
            billed=self.billed,
            tools=tools,
            description=description,
            actor_user_id=self.actor_user_id,
            api_key_id=self.api_key_id,
        )

    async def settle(
        self,
        cost: Decimal | None,
        *,
        tools: Mapping[str, ToolUse],
        model: str,
        description: str,
        expects_cost: bool,
        savepoint: Callable[[], AbstractAsyncContextManager[Any]],
    ) -> None:
        """Charge what a settled ``cost`` owes, and the ``tools`` it pays for, inside ``savepoint``.

        Called before the settlement's commit, which is what lands the charge, so
        the budget and the organization's funds move together. Never raises: the
        response has already been served, so a failed charge is discarded with its
        savepoint and logged, and the settlement that follows still lands.
        """
        amount = self.amount_due(cost, model=model, expects_cost=expects_cost)
        if amount is None:
            return
        try:
            async with savepoint():
                await self.charge(amount, tools=tools, description=description)
        except Exception as exc:
            logger.error(
                "Charge of %s for a served deployment-paid request failed; it goes uncharged: "
                "organization=%s request_id=%s: %s",
                amount,
                self.organization_id,
                self.request_id,
                describe_port_error(exc),
            )


def deferred_charge(
    billing: BillingPort,
    *,
    organization_id: uuid.UUID,
    api_key_id: str | None,
    creator_user_id: str | None,
    provider: "str | LLMProvider",
    model: str,
) -> DeploymentCharge:
    """What a job settled after its request returned (a batch, a video) is charged through.

    Built from what the job recorded when it was admitted, never from whoever
    retrieves it: the creator is who a plan covers, ``model`` is the model
    requested, and ``provider`` is the implementation the catalog lists, not a
    hosted instance's name.
    """
    return DeploymentCharge(
        billing=billing,
        organization_id=organization_id,
        api_key_id=api_key_id,
        actor_user_id=actor_user_id_for(creator_user_id),
        request_id=None,
        billed=billed_model(provider, model),
    )


def organization_slot_key(organization_id: uuid.UUID) -> str:
    """Where an organization's deployment-paid calls in flight are counted."""
    return f"organization:{organization_id}:concurrent"


def describe_port_error(exc: Exception) -> str:
    """Name a port failure without the row data a database error's text can carry."""
    if isinstance(exc, DATABASE_ERRORS):
        return describe_database_error(exc)
    return type(exc).__name__


async def admit_deployment_paid(
    billing: BillingPort,
    *,
    deployment_paid: bool,
    organization_id: uuid.UUID | None,
    billed: BilledModel | None,
    api_key_id: str | None,
    user_id: str | None,
    request_id: str | None,
    tools: frozenset[str] = frozenset(),
) -> DeploymentCharge | None:
    """Admit a request on its credential, returning what settlement charges it through.

    Only a deployment-paid request with an organization is gated and charged; any
    other request is admitted with nothing to charge (``None``). A gated one is
    granted any credit its organization is owed, then refused with nothing paid in
    that can pay for ``billed``, the model it is served on (``None`` for a gateway
    tool's call, which names the tools it may run, by pricing key, in ``tools``).
    A model's call also learns its organization's concurrency limit.

    Raises:
        InsufficientFundsError: If the organization has no funds paid in.

    """
    if not deployment_paid or organization_id is None:
        return None
    actor_user_id = actor_user_id_for(user_id)
    await billing.apply_due_credit(organization_id=organization_id)
    await billing.require_funds_on_deposit(
        organization_id=organization_id, billed=billed, tools=tools, actor_user_id=actor_user_id
    )
    limit = (
        await billing.concurrency_limit(organization_id=organization_id, billed=billed, actor_user_id=actor_user_id)
        if billed
        else None
    )
    return DeploymentCharge(
        billing=billing,
        organization_id=organization_id,
        api_key_id=api_key_id,
        actor_user_id=actor_user_id,
        request_id=request_id,
        billed=billed,
        concurrency_limit=limit,
    )


@dataclass(frozen=True)
class WorkspaceFunding:
    """Whether a workspace's organization passes the gate :func:`admit_deployment_paid` applies.

    ``organization_for`` names a workspace's organization, ``None`` for one with
    none, which that gate admits.
    """

    billing: BillingPort
    organization_for: Callable[[uuid.UUID], Awaitable[uuid.UUID | None]]

    async def funded(self, workspace_id: uuid.UUID, user_id: str | None, tools: frozenset[str]) -> bool:
        """Whether a call of ``tools`` from the workspace, billed to ``user_id``, would be admitted.

        ``tools`` names gateway-run tools by pricing key, as the gate takes them.
        Asked without the gate's side effects. A port that fails to answer is
        logged and reads as funded: this answers what to advertise, and the
        request's own gate still decides.
        """
        organization_id = await self.organization_for(workspace_id)
        if organization_id is None:
            return True
        try:
            return await self.billing.has_funds_on_deposit(
                organization_id=organization_id, billed=None, tools=tools, actor_user_id=actor_user_id_for(user_id)
            )
        except Exception as exc:
            logger.error(
                "BillingPort failed to read an organization's funds: organization=%s: %s",
                organization_id,
                describe_port_error(exc),
            )
            return True
