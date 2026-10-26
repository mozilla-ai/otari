"""The ``Otari-Request-ID`` a request is answered under, on an error as on a success.

FastAPI copies the headers of the response it injects into a route only onto the
response it builds from the route's return value. A Response the route returns
itself never gets them, so a streamed completion passes the header to its
StreamingResponse. Nor does an exception the route raises, which is rendered from
the exception alone, so the id is also kept on the ASGI scope, where the exception
handler finds it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gateway.core.config import REQUEST_ID_HEADER

if TYPE_CHECKING:
    from starlette.requests import Request
    from starlette.responses import Response

_SCOPE_KEY = "otari.request_id"


def set_request_id(request: Request, response: Response | None, request_id: str) -> None:
    """Answer ``request`` under ``request_id``, whether its route returns ``response`` or raises.

    A route that builds its own response passes ``None`` and sets the header on it.
    """
    if response is not None:
        response.headers[REQUEST_ID_HEADER] = request_id
    request.scope[_SCOPE_KEY] = request_id


def with_request_id(request: Request, response: Response) -> Response:
    """``response`` carrying the id ``request`` is answered under, when it has one and names none itself."""
    request_id: str | None = request.scope.get(_SCOPE_KEY)
    if request_id is not None:
        response.headers.setdefault(REQUEST_ID_HEADER, request_id)
    return response
