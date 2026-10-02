"""The bounded POST a deployment makes to its peer control plane.

NOTE: callers must reach ``post`` through this module rather than binding the
name, so that replacing it here reaches them. A bound name keeps the original.
"""

from typing import Any

import httpx

# Bumped when the resolve/usage wire shapes in docs/hybrid-mode-protocol.md
# change in a way the platform needs to branch on. Sent as X-Otari-Protocol-
# Version on every resolve and usage call; unread by this gateway itself.
PLATFORM_PROTOCOL_VERSION = 1


def control_plane_url(base_url: str, path: str) -> str:
    """Join ``path`` onto the control plane's base URL."""
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


async def post(
    *, url: str, headers: dict[str, str], body: dict[str, Any], timeout_seconds: float
) -> httpx.Response:
    """POST ``body`` and return the peer's response, whatever its status."""
    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        return await client.post(url, headers=headers, json=body)
