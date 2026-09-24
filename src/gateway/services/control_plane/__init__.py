"""How a deployment talks to the control plane a peer runs for it."""

from gateway.services.control_plane._resolve import (
    NOT_CONFIGURED_DETAIL,
    UNAVAILABLE_DETAIL,
    ResolveEndpoint,
    resolve,
)
from gateway.services.control_plane._transport import control_plane_url

__all__ = [
    "NOT_CONFIGURED_DETAIL",
    "UNAVAILABLE_DETAIL",
    "ResolveEndpoint",
    "control_plane_url",
    "resolve",
]
