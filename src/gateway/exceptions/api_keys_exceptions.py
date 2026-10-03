"""Errors raised by the API-keys domain."""

from gateway.exceptions import TenancyNotFoundError


class ApiKeyNotFoundError(TenancyNotFoundError):
    """No managed key under this ID is visible to the caller."""

    def __init__(self, key_id: str) -> None:
        super().__init__(f"API key with id '{key_id}' not found")
