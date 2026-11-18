"""Errors that the usage-and-telemetry domain may raise.

Each carries the HTTP status it renders as.
"""

from fastapi import status

from gateway.exceptions import TenancyValidationError


class UsageTagFilterError(TenancyValidationError):
    """A tag filter that is not written ``key:value``."""

    status_code = status.HTTP_422_UNPROCESSABLE_CONTENT

    def __init__(self) -> None:
        super().__init__("A tag filter must be written key:value")


class UsageSeriesTooLongError(TenancyValidationError):
    """A grouped series whose window holds more buckets than one response may carry."""

    status_code = status.HTTP_422_UNPROCESSABLE_CONTENT

    def __init__(self, limit: int, bucket: str) -> None:
        super().__init__(f"window spans more than {limit} {bucket} buckets; use bucket=day or narrow the range")
