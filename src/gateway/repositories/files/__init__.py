"""Data access for the file rows the Files API serves."""

from gateway.repositories.files.file_provider_copy_repository import FileProviderCopyRepository
from gateway.repositories.files.file_repository import (
    FilePageQuery,
    FileRepository,
    OutputFileRow,
    could_name_a_file,
)
from gateway.repositories.files.files_repositories import FileRepositories

__all__ = [
    "FilePageQuery",
    "FileProviderCopyRepository",
    "FileRepositories",
    "FileRepository",
    "OutputFileRow",
    "could_name_a_file",
]
