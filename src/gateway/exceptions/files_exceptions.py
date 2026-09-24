"""Errors that the files domain may raise, each carrying the status it renders as."""

from fastapi import status

from gateway.exceptions import TenancyError, TenancyNotFoundError, TenancyValidationError


class FilesDisabledError(TenancyNotFoundError):
    """The deployment does not serve files.

    A 404 rather than a 403: a switched-off feature has no resource to refuse
    access to, and the paths behave as if they were never mounted.
    """

    def __init__(self) -> None:
        super().__init__("File uploads are disabled")


class FileNotServedError(TenancyNotFoundError):
    """No file under this ID is served to this caller.

    One status and one message for four facts: the ID names nothing, it names
    another user's or another workspace's file, the file is deleted, or it has
    expired. Telling them apart would make the response an existence oracle
    over other tenants' uploads.
    """

    def __init__(self) -> None:
        super().__init__("File not found")


class EmptyUploadError(TenancyValidationError):
    """The upload carried no bytes, which is not a file this API stores."""

    def __init__(self) -> None:
        super().__init__("Uploaded file is empty")


class UploadTooLargeError(TenancyError):
    """The upload ran past the deployment's size ceiling."""

    status_code = status.HTTP_413_CONTENT_TOO_LARGE

    def __init__(self, max_bytes: int) -> None:
        super().__init__(f"File exceeds maximum upload size of {max_bytes // (1024 * 1024)} MB")


class UnknownPageCursorError(TenancyValidationError):
    """The page token names no position this gateway issued."""

    def __init__(self) -> None:
        super().__init__("Invalid page token")


class FileStorageError(TenancyError):
    """The bytes could not be written, read back or removed.

    The caller is told only that the deployment failed. The message names the
    file, and reaches the operator through the log.
    """

    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR

    def __init__(self, message: str) -> None:
        super().__init__(message)


class ProviderAttachmentError(TenancyError):
    """A file a request attached cannot be given to the provider's own code execution.

    Each of these refuses the request rather than answering without the file,
    because the request asked for code to be run over that file and the code
    would not find it.
    """

    status_code = status.HTTP_400_BAD_REQUEST


class AttachedFileUnavailableError(ProviderAttachmentError):
    """The deployment holds no usable file under the ID the request attached."""

    def __init__(self) -> None:
        super().__init__("An attached file does not exist, or holds no bytes here")


class AttachedFileExpiresTooSoonError(ProviderAttachmentError):
    """The file has too little left for a provider to hold a copy no longer than Otari does."""

    def __init__(self) -> None:
        super().__init__("An attached file expires too soon to be copied to the provider")


class ProviderUploadDisabledError(ProviderAttachmentError):
    """The deployment does not upload a copy of an attached file to a provider."""

    def __init__(self) -> None:
        super().__init__("This deployment does not upload attached files to a provider")


class ProviderUploadFailedError(ProviderAttachmentError):
    """A copy of the attached file could not be put at the provider.

    A 502 because nothing the caller sent is wrong: the provider refused the
    copy, would not be reached, or the bytes could not be read back.
    """

    status_code = status.HTTP_502_BAD_GATEWAY

    def __init__(self) -> None:
        super().__init__("An attached file could not be made available to the provider's code execution")
