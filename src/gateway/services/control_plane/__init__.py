"""How a deployment talks to the control plane a peer runs for it."""

from gateway.services.control_plane._resolve import (
    NOT_CONFIGURED_DETAIL,
    UNAVAILABLE_DETAIL,
    ResolveEndpoint,
    resolve,
)

__all__ = [
    "NOT_CONFIGURED_DETAIL",
    "UNAVAILABLE_DETAIL",
    "ResolveEndpoint",
    "resolve",
]
