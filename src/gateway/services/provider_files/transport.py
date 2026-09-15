"""Provider Files transport through any-llm's public interface."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from urllib.parse import urlsplit

from any_llm import AnyLLM
from httpx import AsyncClient

from gateway.services.provider_files.contracts import FileAccount, FilesError
from gateway.services.url_safety import UnsafeURLError, validate_provider_api_base


@asynccontextmanager
async def provider_client(account: FileAccount, *, idle_timeout: float = 30) -> AsyncIterator[Any]:
    base = account.api_base or "https://api.anthropic.com"
    parsed = urlsplit(base)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise FilesError(502, "Invalid provider file endpoint")
    try:
        await validate_provider_api_base(base)
    except UnsafeURLError:
        raise FilesError(502, "Invalid provider file endpoint") from None
    async with AsyncClient(timeout=idle_timeout, follow_redirects=False) as http_client:
        client = AnyLLM.create(
            "anthropic",
            api_key=account.api_key.get_secret_value(),
            api_base=base,
            http_client=http_client,
            max_retries=0,
            default_headers={"anthropic-workspace-id": account.workspace} if account.workspace else {},
        )
        if not hasattr(client, "aupload_file"):
            raise FilesError(502, "Provider-native Files require any-llm-sdk 1.28 or later")
        yield client


def provider_error(exc: Exception) -> FilesError:
    code = getattr(exc, "status_code", None)
    if code is None:
        original = getattr(exc, "original_exception", None)
        code = getattr(original, "status_code", None)
    source = getattr(exc, "original_exception", None) or exc
    response = getattr(source, "response", None)
    retry_after = getattr(response, "headers", {}).get("Retry-After")
    headers = None
    if isinstance(retry_after, str) and len(retry_after) <= 128 and not {"\r", "\n"} & set(retry_after):
        headers = {"Retry-After": retry_after}
    if code in {400, 404, 413, 429}:
        return FilesError(
            code,
            {
                400: "Provider rejected the file operation",
                404: "File unavailable",
                413: "File size limit exceeded",
                429: "Provider file rate limit exceeded",
            }[code],
            headers=headers if code == 429 else None,
        )
    return FilesError(502, "Provider file operation failed")
