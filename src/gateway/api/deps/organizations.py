"""How a request reaches the organizations domain."""

from typing import Annotated

from fastapi import Depends

from gateway.api.deps._base import UnitOfWorkDep
from gateway.repositories.api_keys import ApiKeyRepositories
from gateway.repositories.budgets import BudgetRepositories
from gateway.repositories.organizations import OrganizationRepositories
from gateway.repositories.traces import TracesRepositories
from gateway.services.api_keys import ApiKeyMembershipListener
from gateway.services.budgets import BudgetMembershipListener
from gateway.services.organizations import UserAccess
from gateway.services.tenancy import MembershipListener, MembershipListeners
from gateway.services.tools import McpGrantMembershipListener
from gateway.services.traces import TraceMembershipListener


def get_user_access(uow: UnitOfWorkDep) -> UserAccess:
    """Build the organizations domain's read of users' access lists on the request's Unit of Work."""
    return UserAccess(OrganizationRepositories.on(uow))


def get_membership_listener(uow: UnitOfWorkDep) -> MembershipListener:
    """Return the listener that keeps members' budget ceilings, own keys, session shares and MCP grants in step.

    It writes through the request's Unit of Work, so a service that changes membership must be built on the same one.
    """
    return MembershipListeners(
        BudgetMembershipListener(BudgetRepositories.on(uow)),
        ApiKeyMembershipListener(ApiKeyRepositories.on(uow)),
        TraceMembershipListener(TracesRepositories.on(uow)),
        McpGrantMembershipListener(uow),
    )


MembershipListenerDep = Annotated[MembershipListener, Depends(get_membership_listener)]
