"""Multipart admission limits and cleanup happen before provider upload."""

from collections.abc import AsyncIterator

import httpx
import pytest
from starlette.datastructures import Headers
from starlette.requests import Request

from gateway.api.routes.hybrid_files import file_headers
from gateway.services.provider_files.contracts import FilesError
from gateway.services.provider_files.transfers import UploadAdmission, receive_upload


@pytest.mark.parametrize("beta", ["files-api-2025-04-14", "other-beta, files-api-2025-04-14 "])
def test_legacy_beta_rejected_case_insensitive_header_name(beta: str) -> None:
    request = Request(
        {"type": "http", "headers": [(b"anthropic-version", b"2023-06-01"), (b"anthropic-beta", beta.encode())]}
    )
    with pytest.raises(FilesError, match="GA API"):
        file_headers(request)


def test_only_contract_headers_forwarded() -> None:
    request = Request(
        {
            "type": "http",
            "headers": [
                (b"anthropic-version", b"2023-06-01"),
                (b"x-api-key", b"caller-key"),
                (b"anthropic-workspace-id", b"foreign"),
                (b"anthropic-beta", b"other-beta"),
            ],
        }
    )
    assert file_headers(request) == {"anthropic-version": "2023-06-01", "anthropic-beta": "other-beta"}


@pytest.mark.asyncio
async def test_spooled_upload_closes_on_success_and_failure() -> None:
    for reject in (False, True):
        request = httpx.Request("POST", "https://test", files={"file": ("private.csv", b"col\nvalue", "text/csv")})
        body = request.read()

        async def chunks() -> AsyncIterator[bytes]:
            yield body

        upload = None
        try:
            async with receive_upload(Headers(request.headers), chunks(), max_bytes=100, idle_seconds=1) as (
                upload,
                duration,
            ):
                assert upload.file.read() == b"col\nvalue"
                assert duration is None
                if reject:
                    raise ValueError("intentional failure")
        except ValueError:
            assert reject
        assert upload is not None and upload.file.closed


@pytest.mark.asyncio
async def test_body_limit_before_multipart_completion() -> None:
    async def chunks() -> AsyncIterator[bytes]:
        yield b"x" * 70000

    with pytest.raises(FilesError) as error:
        async with receive_upload(
            Headers({"content-type": "multipart/form-data; boundary=test"}), chunks(), max_bytes=1, idle_seconds=1
        ):
            pytest.fail("Oversized upload was accepted")
    assert error.value.status_code == 413


@pytest.mark.asyncio
async def test_aggregate_admission_and_release() -> None:
    admission = UploadAdmission(100)
    async with admission.reserve(70):
        with pytest.raises(FilesError):
            async with admission.reserve(40):
                pytest.fail("Overcommitted temporary storage")
    assert admission.reserved == 0
    async with admission.reserve(100):
        assert admission.reserved == 100
