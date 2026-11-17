"""Every port this build resolves from the container, and the session each one is resolved on.

A port names a capability in domain terms and the composition root binds an
adapter to it, so this module is the one place a deployment's rebindable seams
can be read as a set. A domain's own module re-exports the ports that are its.
"""

from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps._base import (
    ContainerDep,
    PortSessionDep,
    UnitOfWorkDep,
    get_config,
    get_container,
    get_unit_of_work,
)
from gateway.api.deps.auth import (
    verify_master_key,
)
from gateway.container import Container
from gateway.core.config import (
    GatewayConfig,
)
from gateway.core.database import get_db, release_session
from gateway.core.unit_of_work import UnitOfWork
from gateway.ports.agent_model_recommender_port import AgentModelRecommenderPort
from gateway.ports.api_key_format_port import ApiKeyFormatPort
from gateway.ports.billing_port import (
    BillingPort,
)
from gateway.ports.code_execution_port import CodeExecutionPort
from gateway.ports.entitlement_port import EntitlementPort
from gateway.ports.feedback_delivery_port import FeedbackDeliveryPort
from gateway.ports.growth_signal_port import GrowthSignalPort
from gateway.ports.identity_provider_port import IdentityProviderPort
from gateway.ports.model_provider_port import ModelProviderPort
from gateway.ports.signup_invite_port import SignupInvitePort
from gateway.ports.telemetry_storage_port import TelemetryStoragePort


def get_agent_model_recommender_port(container: ContainerDep) -> AgentModelRecommenderPort:
    """Resolve the subagent model recommender this build bound at startup.

    No session: the adapter asks a model, not this database, and the call is
    metered on the route's own session by the decisions scaffold.
    """
    return container.resolve(AgentModelRecommenderPort, None)


AgentModelRecommenderPortDep = Annotated[AgentModelRecommenderPort, Depends(get_agent_model_recommender_port)]


def get_api_key_format_port(db: PortSessionDep, container: ContainerDep) -> ApiKeyFormatPort:
    """Resolve the key-format adapter this build bound at startup."""
    return container.resolve(ApiKeyFormatPort, db)


ApiKeyFormatPortDep = Annotated[ApiKeyFormatPort, Depends(get_api_key_format_port)]


async def get_billing_port(db: PortSessionDep, container: ContainerDep) -> BillingPort:
    """Resolve the billing adapter this build bound at startup."""
    return container.resolve(BillingPort, db)


BillingPortDep = Annotated[BillingPort, Depends(get_billing_port)]


def get_request_session_billing_port(
    db: Annotated[AsyncSession, Depends(get_db)],
    container: ContainerDep,
) -> BillingPort:
    """Resolve the billing adapter on the request's ``get_db`` session."""
    return container.resolve(BillingPort, db)


RequestSessionBillingPortDep = Annotated[BillingPort, Depends(get_request_session_billing_port)]


def billing_port_on(request: Request, db: AsyncSession) -> BillingPort:
    """Resolve the billing adapter on ``db``, whose commit lands the charge it writes."""
    return get_container(request).resolve(BillingPort, db)


async def get_code_execution_port(
    config: Annotated[GatewayConfig, Depends(get_config)],
    container: ContainerDep,
) -> CodeExecutionPort | None:
    """Resolve the code-execution adapter this build bound at startup.

    ``None`` where the deployment has configured no sandbox, which is most of
    them: every completion request resolves this, and one that never asks for
    code execution must not be refused because there is nowhere to run it. The
    request that does ask is refused by name in ``prepare_gateway_tools``.

    No session either: the adapter reaches a sandbox, not this database, and
    which one it reaches is a deployment setting rather than a request fact.
    """
    if not config.sandbox_configured():
        return None
    return container.resolve(CodeExecutionPort, None)


CodeExecutionPortDep = Annotated[CodeExecutionPort | None, Depends(get_code_execution_port)]


def get_entitlement_port(db: PortSessionDep, container: ContainerDep) -> EntitlementPort:
    """Resolve the entitlement adapter this build bound at startup."""
    return container.resolve(EntitlementPort, db)


EntitlementPortDep = Annotated[EntitlementPort, Depends(get_entitlement_port)]


async def get_feedback_delivery_port(
    container: Annotated[Container, Depends(get_container)],
    db: Annotated[AsyncSession, Depends(get_db)],
    _authenticated: Annotated[str | None, Depends(verify_master_key)],
) -> FeedbackDeliveryPort:
    """Resolve where accepted feedback goes, with authentication's database work finished first.

    Delivery can wait seconds on the network, so the request's connection goes
    back to the pool here. The adapter gets no session: one that needs the
    database opens its own unit of work.
    """
    await release_session(db)
    return container.resolve(FeedbackDeliveryPort, None)


def get_growth_signal_port(
    db: Annotated[AsyncSession, Depends(get_db)],
    container: ContainerDep,
) -> GrowthSignalPort:
    """Resolve the growth-signal adapter this build bound at startup."""
    return container.resolve(GrowthSignalPort, db)


GrowthSignalPortDep = Annotated[GrowthSignalPort, Depends(get_growth_signal_port)]


def get_person_growth_signal_port(
    growth: Annotated[GrowthSignalPort, Depends(get_growth_signal_port)],
    master_key: Annotated[str | None, Depends(verify_master_key)],
) -> GrowthSignalPort | None:
    """The growth-signal adapter for a write a person made, or ``None`` when a header master key made it.

    A header master key is the deployment's credential, not a person's, even once
    the bootstrap operator it resolves to holds an address, so automation and
    smoke traffic on a tenant route report nothing. ``verify_master_key`` answers
    the raw key only for a header credential, and ``None`` for a dashboard session.
    """
    return None if master_key is not None else growth


PersonGrowthSignalPortDep = Annotated[GrowthSignalPort | None, Depends(get_person_growth_signal_port)]


# The adapter writes and does not commit, so it shares the session of the Unit of Work whose block commits its writes.
def get_identity_provider_port(
    db: Annotated[AsyncSession, Depends(get_db)],
    uow: UnitOfWorkDep,
    container: ContainerDep,
) -> IdentityProviderPort:
    """Resolve the identity adapter this build bound at startup, on the request's session and Unit of Work."""
    return container.resolve(IdentityProviderPort, db, uow=uow)


IdentityProviderPortDep = Annotated[IdentityProviderPort, Depends(get_identity_provider_port)]


async def get_model_provider_port(
    db: PortSessionDep,
    uow: UnitOfWorkDep,
    container: ContainerDep,
) -> ModelProviderPort:
    """Resolve the model-provider adapter this build bound at startup.

    On the request's session and Unit of Work: the core adapter reads the hosted-providers store through a block on it.
    """
    return container.resolve(ModelProviderPort, db, uow=uow)


ModelProviderPortDep = Annotated[ModelProviderPort, Depends(get_model_provider_port)]


def get_request_session_model_provider_port(
    db: Annotated[AsyncSession, Depends(get_db)],
    uow: Annotated[UnitOfWork, Depends(get_unit_of_work)],
    container: ContainerDep,
) -> ModelProviderPort:
    """Resolve the model-provider adapter on the request's ``get_db`` session and its Unit of Work."""
    return container.resolve(ModelProviderPort, db, uow=uow)


RequestSessionModelProviderPortDep = Annotated[ModelProviderPort, Depends(get_request_session_model_provider_port)]


def model_provider_port_on(request: Request, db: AsyncSession) -> ModelProviderPort:
    """Resolve the model-provider adapter on ``db``, for a path that resolves it only when it needs it."""
    return get_container(request).resolve(ModelProviderPort, db, uow=get_unit_of_work(db))


def get_signup_invite_port(
    db: Annotated[AsyncSession, Depends(get_db)],
    container: ContainerDep,
) -> SignupInvitePort:
    """Resolve the signup-invite adapter this build bound at startup."""
    return container.resolve(SignupInvitePort, db)


SignupInvitePortDep = Annotated[SignupInvitePort, Depends(get_signup_invite_port)]


def get_telemetry_storage_port(
    db: Annotated[AsyncSession, Depends(get_db)],
    container: ContainerDep,
) -> TelemetryStoragePort:
    """Resolve the telemetry-storage adapter this build bound at startup."""
    return container.resolve(TelemetryStoragePort, db)


TelemetryStoragePortDep = Annotated[TelemetryStoragePort, Depends(get_telemetry_storage_port)]
