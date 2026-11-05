"""The inference domain's repositories, as one service receives them."""

from dataclasses import dataclass
from typing import Self

from gateway.core.unit_of_work import UnitOfWork
from gateway.repositories.inference.country_database_repository import CountryDatabaseRepository
from gateway.repositories.inference.idempotency_repository import IdempotencyRepository
from gateway.repositories.inference.video_job_repository import VideoJobRepository


@dataclass(frozen=True)
class InferenceRepositories:
    """The inference domain's repositories, all on one Unit of Work."""

    idempotency: IdempotencyRepository
    country_database: CountryDatabaseRepository
    video_jobs: VideoJobRepository

    @classmethod
    def on(cls, uow: UnitOfWork) -> Self:
        """Build every repository on this Unit of Work."""
        return cls(
            idempotency=IdempotencyRepository(uow),
            country_database=CountryDatabaseRepository(uow),
            video_jobs=VideoJobRepository(uow),
        )
