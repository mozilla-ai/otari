from dataclasses import dataclass
from typing import Self

from gateway.core.unit_of_work import UnitOfWork
from gateway.repositories.users_repository import UserAccessRepository


@dataclass(frozen=True)
class OrganizationRepositories:
    """The organizations domain's repositories, all on one Unit of Work.

    ``users`` is still in the flat ``users_repository`` module, which holds every query on the users table.
    """

    users: UserAccessRepository

    @classmethod
    def on(cls, uow: UnitOfWork) -> Self:
        """Build every repository on this Unit of Work."""
        return cls(users=UserAccessRepository(uow))
