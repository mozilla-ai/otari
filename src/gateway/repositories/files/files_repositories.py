"""The files domain's repositories, as one service receives them."""

from dataclasses import dataclass
from typing import Self

from gateway.core.unit_of_work import UnitOfWork
from gateway.repositories.files.file_provider_copy_repository import FileProviderCopyRepository
from gateway.repositories.files.file_repository import FileRepository


@dataclass(frozen=True)
class FileRepositories:
    """The files domain's repositories, all on one Unit of Work."""

    files: FileRepository
    provider_copies: FileProviderCopyRepository

    @classmethod
    def on(cls, uow: UnitOfWork) -> Self:
        """Build every repository on this Unit of Work."""
        return cls(files=FileRepository(uow), provider_copies=FileProviderCopyRepository(uow))
