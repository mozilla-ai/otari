"""Errors that the inference domain may raise, each carrying the status it renders as."""

from fastapi import status

from gateway.exceptions import TenancyError, TenancyNotFoundError, TenancyValidationError


class EmbeddingOptionUnsupportedError(TenancyValidationError):
    """An embedding option was sent for a provider whose embedding call does not read it.

    Refused rather than dropped, because the embedding would otherwise come back
    as if the option had not been sent.
    """

    def __init__(self, option: str, selector: str) -> None:
        # The selector the caller sent, never the resolved provider, so an alias
        # does not reveal its target.
        super().__init__(f"'{option}' is not supported for model '{selector}'")


class CountryDatabaseInvalidError(TenancyValidationError):
    """An uploaded file is no MaxMind DB country file. The file stored before stays in place."""


class CountryDatabaseTooLargeError(TenancyError):
    """An uploaded file is larger than a country file can be."""

    status_code = status.HTTP_413_CONTENT_TOO_LARGE

    def __init__(self, limit_bytes: int) -> None:
        super().__init__(f"The file is larger than {limit_bytes // (1024 * 1024)} MiB")


class CountryDatabaseNotFoundError(TenancyNotFoundError):
    """No country file is stored."""

    def __init__(self) -> None:
        super().__init__("No country file is stored")
