"""Ask a peer control plane what a workspace may do, and with which credential.

A hybrid gateway serves the data plane and owns no tenancy, so every policy and
credential question goes to a control plane elsewhere. This is the one place
that asks, and it speaks `docs/hybrid-mode-protocol.md`.

It asks questions and carries no customer traffic. ``ResolveEndpoint`` is a
closed set for that reason: a caller cannot supply a path, so nothing here can
grow into a route for the data plane's own work.
"""

from enum import StrEnum
from typing import Any

import httpx

from gateway.core.config import GatewayConfig
from gateway.exceptions.control_plane_exceptions import (
    ControlPlaneNotConfiguredError,
    ControlPlaneRefusedError,
    ControlPlaneUnavailableError,
)
from gateway.services.control_plane import _transport
from gateway.services.control_plane._transport import control_plane_url

# A refusal the peer wrote for the caller, forwarded under its own status. 400
# belongs here because this peer's 400s are hand-written and caller-safe, such
# as a BYO key whose auth shape cannot travel through a gateway. 422 does not,
# because a framework validation error would describe the request's shape. A 421
# says the token belongs to another region and its detail names the host the
# caller must go to, so it forwards rather than collapsing.
_FORWARDED_STATUSES = frozenset({400, 401, 402, 403, 404, 421, 429})

_RATE_LIMITED = 429

UNAVAILABLE_DETAIL = "Authorization service unavailable"
NOT_CONFIGURED_DETAIL = "Hybrid mode is misconfigured"


class ResolveEndpoint(StrEnum):
    """A question the control plane answers about a workspace."""

    PROVIDER_KEYS = "/gateway/provider-keys/resolve"
    MCP_SERVERS = "/gateway/mcp-servers/resolve"
    WEB_SEARCH = "/gateway/web-search/resolve"
    CODE_EXECUTION = "/gateway/code-execution/resolve"


def _safe_detail(response: httpx.Response, fallback: str) -> str:
    """The peer's own ``detail`` when it is a plain string, else ``fallback``."""
    try:
        payload = response.json()
    except ValueError:
        return fallback

    detail = payload.get("detail") if isinstance(payload, dict) else None
    return detail if isinstance(detail, str) else fallback


async def resolve(
    config: GatewayConfig,
    *,
    user_token: str,
    endpoint: ResolveEndpoint,
    body: dict[str, Any],
    client_error_detail: str,
) -> Any:
    """Ask ``endpoint`` about ``body`` and return the parsed answer.

    Owns what every question shares: the address guard, the gateway and user
    token headers, the bounded POST and the outcome ladder.

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
        response = await _transport.post(
            url=control_plane_url(base_url, endpoint.value),
            headers=headers,
            body=body,
            timeout_seconds=timeout_ms / 1000,
        )
    except (httpx.TimeoutException, httpx.NetworkError):
        raise ControlPlaneUnavailableError(UNAVAILABLE_DETAIL) from None

    if response.status_code == 200:
        try:
            return response.json()
        except ValueError:
            raise ControlPlaneUnavailableError(UNAVAILABLE_DETAIL) from None

    if response.status_code in _FORWARDED_STATUSES:
        raise ControlPlaneRefusedError(
            _safe_detail(response, client_error_detail),
            status_code=response.status_code,
            retry_after=response.headers.get("Retry-After") if response.status_code == _RATE_LIMITED else None,
        )

    raise ControlPlaneUnavailableError(UNAVAILABLE_DETAIL)
