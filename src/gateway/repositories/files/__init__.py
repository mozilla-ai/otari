"""Data access for the file rows the Files API serves."""

from gateway.repositories.files.file_repository import FileRepository, OutputFileRow
from gateway.repositories.files.files_repositories import FileRepositories

__all__ = [
    "FileRepositories",
    "FileRepository",
    "OutputFileRow",
]
