"""Ask a peer control plane what a workspace may do, and with which credential.

A hybrid gateway serves the data plane and owns no tenancy, so every policy and
credential question goes to a control plane elsewhere. This is the one place
that asks, and it speaks `docs/hybrid-mode-protocol.md`.

It asks questions and carries no customer traffic. ``ResolveEndpoint`` is a
closed set for that reason: a caller cannot supply a path, so nothing here can
grow into a route for the data plane's own work.
"""

from enum import StrEnum
from http import HTTPStatus
from typing import Any

import httpx

from gateway.core.config import GatewayConfig
from gateway.exceptions.control_plane_exceptions import (
    ControlPlaneNotConfiguredError,
    ControlPlaneRefusedError,
    ControlPlaneUnavailableError,
)
from gateway.services.control_plane import transport

# A refusal the peer wrote for the caller, forwarded under its own status. 400
# belongs here because this peer's 400s are hand-written and caller-safe, such
# as a BYO key whose auth shape cannot travel through a gateway. 422 does not,
# because a framework validation error would describe the request's shape. A 421
# says the token belongs to another region and its detail names the host the
# caller must go to, so it forwards rather than collapsing.
_FORWARDED_STATUSES = frozenset(
    {
        HTTPStatus.BAD_REQUEST,
        HTTPStatus.UNAUTHORIZED,
        HTTPStatus.PAYMENT_REQUIRED,
        HTTPStatus.FORBIDDEN,
        HTTPStatus.NOT_FOUND,
        HTTPStatus.MISDIRECTED_REQUEST,
        HTTPStatus.TOO_MANY_REQUESTS,
    }
)

NOT_CONFIGURED_DETAIL = "Hybrid mode is misconfigured"
UNAVAILABLE_DETAIL = "Authorization service unavailable"


class ResolveEndpoint(StrEnum):
    """A question the control plane answers about a workspace.

    A member is its path and the detail a caller sees when the peer refuses
    without one of its own. The wording belongs to the question rather than to
    whichever module asks it, and pairing them here means a member cannot exist
    without one.
    """

    def __new__(cls, path: str, refusal_detail: str) -> "ResolveEndpoint":
        member = str.__new__(cls, path)
        member._value_ = path
        member.refusal_detail = refusal_detail
        return member

    refusal_detail: str

    CODE_EXECUTION = ("/gateway/code-execution/resolve", "Code execution resolution failed")
    MCP_SERVERS = ("/gateway/mcp-servers/resolve", "MCP server resolution failed")
    PROVIDER_KEYS = ("/gateway/provider-keys/resolve", "Authorization request rejected")
    WEB_SEARCH = ("/gateway/web-search/resolve", "Web search resolution failed")


def _safe_detail(response: httpx.Response, fallback: str) -> str:
    """The peer's own ``detail`` when it is a plain string, else ``fallback``."""
    try:
        payload = response.json()
    except ValueError:
        return fallback

    detail = payload.get("detail") if isinstance(payload, dict) else None
    return detail if isinstance(detail, str) else fallback


async def resolve(config: GatewayConfig, *, user_token: str, endpoint: ResolveEndpoint, body: dict[str, Any]) -> Any:
    """Ask ``endpoint`` about ``body`` and return the parsed answer.

    Raises:
        ControlPlaneNotConfiguredError: no control plane address is set.
        ControlPlaneRefusedError: the peer refused, carrying its status, its
            detail where that is a safe string, and its ``Retry-After``.
        ControlPlaneUnavailableError: a timeout, a network failure, an
            unreadable body or any other status, so nothing about the peer's
            internals reaches the caller.
    """
    base_url = config.platform.get("base_url")
    if not base_url:
        raise ControlPlaneNotConfiguredError(NOT_CONFIGURED_DETAIL)

    timeout_ms = int(config.platform.get("resolve_timeout_ms", 5000))
    headers = {
        "X-Gateway-Token": config.platform_token or "",
        "X-User-Token": user_token,
    }

    try:
        response = await transport.post(
            url=transport.control_plane_url(base_url, endpoint.value),
            headers=headers,
            body=body,
            timeout_seconds=timeout_ms / 1000,
        )
    except (httpx.TimeoutException, httpx.NetworkError):
        raise ControlPlaneUnavailableError(UNAVAILABLE_DETAIL) from None

    if response.status_code == HTTPStatus.OK:
        try:
            return response.json()
        except ValueError:
            raise ControlPlaneUnavailableError(UNAVAILABLE_DETAIL) from None

    if response.status_code in _FORWARDED_STATUSES:
        rate_limited = response.status_code == HTTPStatus.TOO_MANY_REQUESTS
        raise ControlPlaneRefusedError(
            _safe_detail(response, endpoint.refusal_detail),
            status_code=response.status_code,
            retry_after=response.headers.get("Retry-After") if rate_limited else None,
        )

    raise ControlPlaneUnavailableError(UNAVAILABLE_DETAIL)
