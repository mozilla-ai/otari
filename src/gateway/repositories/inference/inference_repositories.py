"""The inference domain's repositories, as one service receives them."""

from dataclasses import dataclass
from typing import Self

from gateway.core.unit_of_work import UnitOfWork
from gateway.repositories.inference.idempotency_repository import IdempotencyRepository


@dataclass(frozen=True)
class InferenceRepositories:
    """The inference domain's repositories, all on one Unit of Work."""

    idempotency: IdempotencyRepository

    @classmethod
    def on(cls, uow: UnitOfWork) -> Self:
        """Build every repository on this Unit of Work."""
        return cls(idempotency=IdempotencyRepository(uow))
