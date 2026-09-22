"""File-domain services that are not part of the original flat modules.

``file_service.py``, ``file_store.py`` and ``file_extractors.py`` next door
still serve uploads and document understanding; this package holds what has
been written since the layout rule took effect.
"""

from gateway.services.files.file_sweeper import SweepBatch, run_file_sweeper, sweep_files
from gateway.services.files.provider_files import ProviderFile, produced_files_for
from gateway.services.files.sandbox_bridge import SandboxFileBridge

__all__ = [
    "ProviderFile",
    "SandboxFileBridge",
    "SweepBatch",
    "produced_files_for",
    "run_file_sweeper",
    "sweep_files",
]
