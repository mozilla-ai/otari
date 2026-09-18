"""The gateway's error classes, presented at one import path."""

from gateway.exceptions._base import (
    TenancyConflictError,
    TenancyError,
    TenancyForbiddenError,
    TenancyNotFoundError,
    TenancyValidationError,
)

__all__ = [
    "TenancyConflictError",
    "TenancyError",
    "TenancyForbiddenError",
    "TenancyNotFoundError",
    "TenancyValidationError",
]
