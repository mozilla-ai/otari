"""The one request a check sends to otari.ai, and the reply it reads back."""

import asyncio
import uuid
from datetime import timedelta

import httpx
from pydantic import ValidationError

from gateway.schemas.update_check import UpdateCheckReply

TIMEOUT_SECONDS = 10
# A version, a link and at most fifty advisories. Anything longer is not a reply.
MAX_REPLY_BYTES = 256 * 1024


class UpdateCheckFailedError(Exception):
    """The check got no usable reply. ``retry_after`` is the server's own ask, when it made one."""

    def __init__(self, reason: str, retry_after: timedelta | None = None) -> None:
        super().__init__(reason)
        self.retry_after = retry_after


def _retry_after(response: httpx.Response) -> timedelta | None:
    value = response.headers.get("Retry-After", "")
    if value.isascii() and value.isdecimal() and len(value) <= 6:
        return timedelta(seconds=min(int(value), 86_400))
    return None


async def send_check(
    url: str, *, version: str, install_id: uuid.UUID, transport: httpx.AsyncBaseTransport | None = None
) -> UpdateCheckReply:
    """POST the version and the install id, and nothing else. Raises ``UpdateCheckFailedError``.

    A POST rather than a GET so the install id stays out of every access log and
    proxy on the way, and so no cache answers it in otari.ai's place.
    """
    try:
        async with (
            # httpx's timeout bounds each read; this bounds the whole check, a reply trickled byte by byte included.
            asyncio.timeout(TIMEOUT_SECONDS * 2),
            httpx.AsyncClient(
                timeout=TIMEOUT_SECONDS, follow_redirects=False, trust_env=False, transport=transport
            ) as client,
            client.stream("POST", url, json={"version": version, "install_id": str(install_id)}) as response,
        ):
            if response.status_code != httpx.codes.OK:
                raise UpdateCheckFailedError(f"status {response.status_code}", _retry_after(response))
            body = b""
            async for chunk in response.aiter_bytes():
                body += chunk
                if len(body) > MAX_REPLY_BYTES:
                    raise UpdateCheckFailedError("reply too large")
    except (httpx.HTTPError, TimeoutError) as exc:
        raise UpdateCheckFailedError(type(exc).__name__) from None
    try:
        return UpdateCheckReply.model_validate_json(body)
    except ValidationError:
        raise UpdateCheckFailedError("reply not understood") from None
