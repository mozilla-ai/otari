"""The files domain: uploaded files, their lifecycle, and the sweep that gives their storage back."""

from gateway.services.files._metadata import expiry_for
from gateway.services.files._sandbox_bridge import SandboxFileBridge
from gateway.services.files._service import (
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
from gateway.services.files._staging import CODE_EXECUTION_OUTPUT_PURPOSE, StagedFile
from gateway.services.files._sweeper import run_file_sweeper
from gateway.types.provider_file import ProviderFile

__all__ = [
    "CODE_EXECUTION_OUTPUT_PURPOSE",
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
    "run_file_sweeper",
]
