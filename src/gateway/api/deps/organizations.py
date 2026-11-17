"""How a request reaches the organizations domain."""

from gateway.api.deps._base import UnitOfWorkDep
from gateway.repositories.organizations import OrganizationRepositories
from gateway.services.organizations import UserAccess


def get_user_access(uow: UnitOfWorkDep) -> UserAccess:
    """Build the organizations domain's read of users' access lists on the request's Unit of Work."""
    return UserAccess(OrganizationRepositories.on(uow))
