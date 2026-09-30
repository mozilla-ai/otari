"""The files domain: uploaded files, their lifecycle, and the sweep that gives their storage back."""

from gateway.services.files._metadata import expiry_for, guess_mime_type
from gateway.services.files._provider_files import produced_files_for
from gateway.services.files._provider_uploads import provider_holds_copies
from gateway.services.files._sandbox_bridge import SandboxFileBridge
from gateway.services.files._service import (
    DEFAULT_LIST_LIMIT,
    MAX_LIST_LIMIT,
    FileBackends,
    FileContent,
    FileDialect,
    FileListing,
    FilePage,
    FileScope,
    FileService,
    NewFile,
    NewOutput,
    SweepBatch,
)
from gateway.services.files._staging import CODE_EXECUTION_OUTPUT_PURPOSE, StagedFile, sandbox_path_for
from gateway.services.files._sweeper import run_file_sweeper
from gateway.types.provider_file import ProviderFile

__all__ = [
    "CODE_EXECUTION_OUTPUT_PURPOSE",
    "DEFAULT_LIST_LIMIT",
    "MAX_LIST_LIMIT",
    "FileBackends",
    "FileContent",
    "FileDialect",
    "FileListing",
    "FilePage",
    "FileScope",
    "FileService",
    "NewFile",
    "NewOutput",
    "ProviderFile",
    "SandboxFileBridge",
    "StagedFile",
    "SweepBatch",
    "expiry_for",
    "guess_mime_type",
    "produced_files_for",
    "provider_holds_copies",
    "run_file_sweeper",
    "sandbox_path_for",
]
