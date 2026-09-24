"""The bounded POST a deployment makes to its peer control plane.

NOTE: callers reach ``post`` through this module rather than binding the name,
because a test double replaces it here. A bound name keeps the real one.
"""

from typing import Any

import httpx


def control_plane_url(base_url: str, path: str) -> str:
    """Join ``path`` onto the control plane's base URL."""
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


async def post(
    *, url: str, headers: dict[str, str], body: dict[str, Any], timeout_seconds: float
) -> httpx.Response:
    """POST ``body`` and return the peer's response, whatever its status."""
    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        return await client.post(url, headers=headers, json=body)
