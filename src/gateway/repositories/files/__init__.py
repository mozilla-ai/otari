"""Data access for the file rows the Files API serves."""

from gateway.repositories.files.file_repository import (
    FilePageQuery,
    FileRepository,
    OutputFileRow,
    could_name_a_file,
)
from gateway.repositories.files.files_repositories import FileRepositories

__all__ = [
    "FilePageQuery",
    "FileRepositories",
    "FileRepository",
    "OutputFileRow",
    "could_name_a_file",
]
