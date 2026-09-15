"""Stateless client for the trusted Files authority; never transfers file bytes."""

from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from gateway.services.provider_files.contracts import FilesError

T = TypeVar("T", bound=BaseModel)


class PlatformFilesClient:
    def __init__(self, base_url: str, gateway_token: str, user_token: str | None, *, timeout: float = 5) -> None:
        self.base_url = base_url.rstrip("/")
        self._gateway_token = gateway_token
        self._user_token = user_token
        self.timeout = timeout

    async def post(self, path: str, body: dict[str, Any], result_type: type[T]) -> T:
        headers = {"X-Gateway-Token": self._gateway_token}
        if self._user_token is not None:
            headers["X-User-Token"] = self._user_token
        try:
            async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=False) as client:
                response = await client.post(f"{self.base_url}/gateway/files/{path}", headers=headers, json=body)
        except httpx.HTTPError:
            raise FilesError(502, "Authorization service unavailable") from None
        if response.status_code == 404:
            # A supporting peer marks all Files responses, including missing bindings.
            if response.headers.get("X-Otari-Files-Protocol") != "1":
                raise FilesError(502, "Authorization service does not support provider-native Files")
            raise FilesError(404, "File or provider account unavailable")
        if response.status_code in {400, 401, 403, 409, 413, 429}:
            details = {
                400: "Invalid file operation",
                401: "Invalid authentication token",
                403: "File operation forbidden",
                409: "File operation conflict",
                413: "File size limit exceeded",
                429: "File operation limit exceeded",
            }
            raise FilesError(
                response.status_code,
                details[response.status_code],
                {"Retry-After": response.headers["Retry-After"]} if "Retry-After" in response.headers else None,
            )
        if response.status_code != 200:
            raise FilesError(502, "Authorization service unavailable")
        try:
            return result_type.model_validate(response.json())
        except (ValueError, ValidationError):
            raise FilesError(502, "Authorization service returned an invalid file response") from None

    async def retry(self, path: str, body: dict[str, Any], result_type: type[T]) -> T:
        """Bound retries to idempotent finalization and cleanup reports only."""
        for attempt in range(3):
            try:
                return await self.post(path, body, result_type)
            except FilesError as exc:
                if exc.status_code != 502 or attempt == 2:
                    raise
        raise AssertionError("unreachable")
