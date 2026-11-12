"""Errors that the update-check domain may raise.

Each carries the HTTP status it renders as.
"""

from gateway.exceptions import TenancyConflictError


class UpdateCheckSetInConfigError(TenancyConflictError):
    """The deployment's configuration answers whether to check, so the dashboard does not."""

    def __init__(self) -> None:
        super().__init__(
            "The update check is decided by this deployment's configuration "
            "(update_check, OTARI_UPDATE_CHECK, DO_NOT_TRACK or CI); change it there"
        )
