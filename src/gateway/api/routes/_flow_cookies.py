"""The cookies that bind each sign-in in flight to the browser that started it.

A browser holds one cookie per flow, named for the flow and holding its browser token.
A response sets only its own flow's token, so sign-ins that start together all keep theirs.
It expires other flows only to bound how many a browser holds.
"""

import hashlib
import re
from dataclasses import dataclass

from fastapi import Request, Response

from gateway.models.tenancy import OAUTH_STATE_TTL_SECONDS
from gateway.services.dashboard_session_service import request_is_https

# The most flows a start leaves of those its request carried, and the most it expires.
# An abandoned sign-in keeps its cookie until the cookie expires.
# A browser past its own per-site limit evicts cookies of its choosing, the session cookie included.
# Each expiry is a response header, and a proxy refuses a response whose headers outgrow its buffer.
MAX_LIVE_FLOWS = 8


@dataclass(frozen=True)
class FlowCookies:
    """The flow cookies under one name prefix and path, each holding the browser token of one sign-in in flight.

    NOTE: ``path`` must cover every route that starts or finishes a flow, or the browser withholds the cookies there.
    """

    prefix: str
    path: str

    def start(self, request: Request, response: Response, *, state: str, browser_token: str) -> None:
        """Give the browser ``browser_token`` to hold for the flow ``state`` names.

        A request that carries ``MAX_LIVE_FLOWS`` flows or more has the first it lists expired,
        at most ``MAX_LIVE_FLOWS`` of them, so a browser converges on holding that many.

        NOTE: RFC 6265 recommends that a browser list the oldest first, and one that does not loses another flow.
        """
        # Only a name this class could have written goes back into a response header.
        held = [name for name in request.cookies if self._names_a_flow(name)]
        surplus = max(0, len(held) + 1 - MAX_LIVE_FLOWS)
        for cookie_name in held[: min(surplus, MAX_LIVE_FLOWS)]:
            self._expire(request, response, cookie_name)
        response.set_cookie(
            self._cookie_name(state),
            browser_token,
            max_age=OAUTH_STATE_TTL_SECONDS,
            httponly=True,
            secure=request_is_https(request),
            samesite="lax",
            path=self.path,
        )

    def browser_token(self, request: Request, state: str) -> str | None:
        """The browser token ``request`` carries for ``state``, or ``None`` where it carries none.

        This selects which token to present and verifies nothing about it.
        """
        return request.cookies.get(self._cookie_name(state))

    def finish(self, request: Request, response: Response, state: str) -> None:
        """Take the token of the flow ``state`` names back out of the browser."""
        self._expire(request, response, self._cookie_name(state))

    def _cookie_name(self, state: str) -> str:
        """The name of the cookie for ``state``.

        The name is a digest, so no character of a state a request supplied reaches a response header.
        """
        return f"{self.prefix}_{hashlib.sha256(state.encode()).hexdigest()}"

    def _names_a_flow(self, cookie_name: str) -> bool:
        """Whether ``cookie_name`` is one ``_cookie_name`` could have written."""
        return re.fullmatch(rf"{re.escape(self.prefix)}_[0-9a-f]{{64}}", cookie_name) is not None

    def _expire(self, request: Request, response: Response, cookie_name: str) -> None:
        response.delete_cookie(
            cookie_name, path=self.path, httponly=True, secure=request_is_https(request), samesite="lax"
        )
