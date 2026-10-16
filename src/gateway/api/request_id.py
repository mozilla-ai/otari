"""The ``Otari-Request-ID`` a completion is answered under, on an error as on a success.

FastAPI copies the headers of the response it injects into a route only onto a
response the route returns. An exception the route raises is rendered from the
exception alone, so the id is also kept on the ASGI scope, where the exception
handler finds it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gateway.core.config import REQUEST_ID_HEADER

if TYPE_CHECKING:
    from starlette.requests import Request
    from starlette.responses import Response

# Namespaced ASGI scope key holding the id the request is answered under.
_SCOPE_KEY = "otari.request_id"


def set_request_id(request: Request, response: Response, request_id: str) -> None:
    """Answer ``request`` under ``request_id``, whether its route returns ``response`` or raises."""
    response.headers[REQUEST_ID_HEADER] = request_id
    request.scope[_SCOPE_KEY] = request_id


def with_request_id(request: Request, response: Response) -> Response:
    """``response`` carrying the id ``request`` is answered under, when it has one and names none itself."""
    request_id: str | None = request.scope.get(_SCOPE_KEY)
    if request_id is not None:
        response.headers.setdefault(REQUEST_ID_HEADER, request_id)
    return response
