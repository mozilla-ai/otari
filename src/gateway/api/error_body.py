"""An error answered with a body of the API dialect's own, rather than FastAPI's ``{"detail": ...}``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import HTTPException

if TYPE_CHECKING:
    from collections.abc import Mapping


class BodyHTTPException(HTTPException):
    """An ``HTTPException`` whose ``body`` is the whole response body.

    Anthropic's SDKs read ``{"type": "error", "error": {...}}`` from the top of an error
    body, which FastAPI's ``{"detail": ...}`` would bury one level down.
    """

    def __init__(self, status_code: int, body: dict[str, Any], headers: Mapping[str, str] | None = None) -> None:
        super().__init__(status_code=status_code, detail=body, headers=headers)
        self.body = body
