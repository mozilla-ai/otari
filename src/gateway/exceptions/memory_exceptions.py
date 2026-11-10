"""Memory's refusals, each with its HTTP status."""

from fastapi import status

from gateway.exceptions import TenancyError

SEALING_UNAVAILABLE = "Memory can't store anything right now. Try again later."


class MemorySealingUnavailableError(TenancyError):
    """No data key could be made or opened, so nothing is stored: memory never keeps text unsealed."""

    status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    def __init__(self) -> None:
        super().__init__(SEALING_UNAVAILABLE)
