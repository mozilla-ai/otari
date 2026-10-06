"""The gateway's error classes, presented at one import path.

Each class names the HTTP status its condition maps to, and one handler
registered in `gateway.main` renders it as FastAPI's own ``{"detail": ...}``
body. A service raises a domain error and its routes stay thin, with no
try/except around each one.
"""

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
