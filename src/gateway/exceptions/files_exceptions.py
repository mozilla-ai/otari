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
