"""Discover provider operations from any-llm, without constructing SDK clients."""

from any_llm import AnyLLM
from any_llm.exceptions import UnsupportedProviderError
from any_llm.types.files import FileOperation

from gateway.services.provider_files.contracts import FileAccount, FileMetadata, FilesError


def require_file_operation(provider: str, operation: FileOperation) -> type[AnyLLM]:
    try:
        provider_class = AnyLLM.get_provider_class(provider)
    except (UnsupportedProviderError, ValueError, ImportError):
        raise FilesError(400, "Unknown file provider") from None
    if operation not in provider_class.SUPPORTED_FILE_OPERATIONS:
        raise FilesError(400, "Provider does not support this file operation")
    return provider_class


def check_file_account(account: FileAccount, provider: str) -> None:
    if account.provider != provider:
        raise FilesError(502, "Authorization service returned an invalid file account")


def require_download(provider: str, metadata: FileMetadata) -> None:
    require_file_operation(provider, "download")
    if metadata.downloadable is False:
        raise FilesError(400, "This file is not downloadable")
    # Unknown per-file permission is decided upstream, after local ownership authorization.
